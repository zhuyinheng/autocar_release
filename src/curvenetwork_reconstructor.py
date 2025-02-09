"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import torch
import numpy as np
import rootutils
from typing import List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.camera import Camera
from src.geometry.multiview_image import MultiViewVideo
from src.geometry.curvenetwork import Curvenetwork
from src.geometry.deformation_graph import DeformationGraph
from src.geometry.correspondence import Correspondence


class DeformableCurvenetworkReconstructor:

    def __init__(
        self,
        device="cuda:0",
    ):
        pass
        self.multi_view_video = MultiViewVideo()
        self.deformable_curve_network: Dict[int, Curvenetwork] = {}
        self.deformation_graphs: Dict[Tuple[int, int], DeformationGraph] = {}
        self.device = device

    def init_time_step(
        self,
        time_idx: int,
        sparse_volume_coords: np.ndarray,
        sparse_volume_feats: np.ndarray,
        sparse_volume_voxel_spacing: np.ndarray,
        sparse_volume_min_bound: np.ndarray,
        cam_list: List[Camera],
        mask_list: List[np.ndarray],
    ):
        self.view_num = len(cam_list)
        for view_idx in range(self.view_num):
            self.multi_view_video.update_image_at_time_view(
                time_idx,
                view_idx,
                mask_list[view_idx],
                cam_list[view_idx],
                mask_list[view_idx],
            )

        self.deformable_curve_network[time_idx] = (
            Curvenetwork.init_from_sparse_volume(
                sparse_volume_coords,
                sparse_volume_feats,
                sparse_volume_min_bound,
                sparse_volume_voxel_spacing,
            )
        )

    def optimize(self, source_t: int, target_t: int):
        """
        Finding the optimal transformation from source_t to target_t,
        in other words, warp(Curvenetwork[source_t]) -> Curvenetwork[target_t].

        It corresponds to the matching-registration algorithm in the paper.

        It consist the following steps:

        1. Pre-conditioning: using 2D optical flow to warp the source points to the target points.
        2. Non-rigid matching-optimizing double loop:
            1. Matching: finding the correspondence between the source points and the target points.
            2. Optimizing: optimizing the deformation graph to minimize the matching loss.
            * Since efficient L-BFGS optimizer is used, aprrox. 10 steps optimization would be sufficient

        """

        assert (
            source_t in self.deformable_curve_network.keys()
            and target_t in self.deformable_curve_network.keys()
        )

        source_curvenetwork: Curvenetwork = self.deformable_curve_network[
            source_t
        ]
        target_curvenetwork: Curvenetwork = self.deformable_curve_network[
            target_t
        ]

        # Step1: Pre-conditionting, using optical flow

        paths = source_curvenetwork.dfs_paths
        source_points_np = source_curvenetwork.high_quality_graph_pts

        deform_graph = DeformationGraph(source_points_np).to(self.device)
        source_points = torch.tensor(
            source_points_np,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        mv_optflow_warped_source_points = {
            k: torch.tensor(
                v, dtype=torch.float32, device=self.device, requires_grad=False
            )
            for k, v in self.multi_view_video.warp_points_3d(
                source_points_np, source_t, target_t
            ).items()
        }

        optimizer = torch.optim.Adagrad(
            [{"params": deform_graph.parameters()}], lr=1e-2
        )

        for iter in range(100):
            optimizer.zero_grad()
            loss_dict = {}
            deformed_source_points = deform_graph.warp_points(source_points)
            mv_deformed_source_points = (
                self.multi_view_video.points_world2pix_fix_time_loop_view(
                    deformed_source_points, target_t
                )
            )
            for view_idx in range(self.view_num):
                deformed_source_points_2d = mv_deformed_source_points[view_idx]
                loss_dict[view_idx] = torch.norm(
                    (
                        deformed_source_points_2d
                        - mv_optflow_warped_source_points[view_idx]
                    ),
                    dim=-1,
                ).mean()
            loss_dict["reg"] = (
                deform_graph.reg_term()
                / (deform_graph.control_points_edges.shape[0])
                * 10
            )
            loss = 0
            for k, v in loss_dict.items():
                loss += v
            loss.backward()
            optimizer.step()

        # Step 2: Matching-optimization

        target_points = target_curvenetwork.points
        target_points_mv = self.multi_view_video.high_quality_centerline_points_fix_time_loop_view(
            target_t
        )
        optimizer = torch.optim.LBFGS(
            [{"params": deform_graph.parameters()}],
            line_search_fn="strong_wolfe",
        )
        # matching-optimzing double loop
        for matching_iter in range(3):

            corres_3d = Correspondence()
            corres_2ds = {}
            # matching
            with torch.no_grad():
                deformed_source_points = deform_graph.warp_points(
                    source_points
                )
                mv_deformed_source_points = (
                    self.multi_view_video.points_world2pix_fix_time_loop_view(
                        deformed_source_points, target_t
                    )
                )

                corres_3d.fit(deformed_source_points, paths, target_points)
                for view_idx in range(self.view_num):
                    deformed_source_points_2d = mv_deformed_source_points[
                        view_idx
                    ]
                    corres_2d = Correspondence()
                    corres_2d.fit(
                        deformed_source_points_2d,
                        paths,
                        target_points_mv[view_idx],
                    )
                    corres_2ds[view_idx] = corres_2d

            # optimization
            def _closure():
                optimizer.zero_grad()
                loss_dict = {}
                deformed_source_points = deform_graph.warp_points(
                    source_points
                )

                # 3d matching loss
                disp = (
                    deformed_source_points[corres_3d.fused_valid_mask, :]
                    - corres_3d.fused_dual_pts[corres_3d.fused_valid_mask, :]
                )
                tangent_projection = disp * (
                    corres_3d.fused_dual_tangent[corres_3d.fused_valid_mask, :]
                )
                normal_projection = disp - tangent_projection
                loss_dict["3d"] = (
                    torch.norm(normal_projection, dim=-1) ** 2
                ).mean() + (torch.norm(tangent_projection, dim=-1) ** 2).mean()

                # 2d matching loss
                mv_deformed_pts = (
                    self.multi_view_video.points_world2pix_fix_time_loop_view(
                        deformed_source_points, target_t
                    )
                )
                for view_idx in range(self.view_num):
                    corres_2d = corres_2ds[view_idx]
                    projected_deformed_pts_3d = mv_deformed_pts[view_idx]
                    disp = (
                        projected_deformed_pts_3d[
                            corres_2d.fused_valid_mask, :
                        ]
                        - corres_2d.fused_dual_pts[
                            corres_2d.fused_valid_mask, :
                        ]
                    )
                    tangent_projection = disp * (
                        corres_2d.fused_dual_tangent[
                            corres_2d.fused_valid_mask, :
                        ]
                    )
                    normal_projection = disp - tangent_projection

                    loss_dict[view_idx] = (
                        torch.norm(normal_projection, dim=-1) ** 2
                    ).mean() + (
                        torch.norm(tangent_projection, dim=-1) ** 2
                    ).mean()

                loss_dict["reg"] = deform_graph.reg_term() / (
                    deform_graph.control_points_edges.shape[0]
                )
                loss = 0
                for k, v in loss_dict.items():
                    loss += v
                loss.backward()
                # optimizer.step()
                return loss

            for _ in range(3):
                optimizer.step(_closure)

        # Step 3: save state
        deformed_source_points = (
            deform_graph.warp_points(source_points).detach().cpu().numpy()
        )

        self.deformation_graphs[(source_t, target_t)] = deform_graph.to(
            "cpu"
        ).eval()
        self.deformable_curve_network[target_t]._high_quality_graph = (
            self.deformable_curve_network[source_t]._high_quality_graph
        )
        self.deformable_curve_network[target_t]._high_quality_graph_pts = (
            deformed_source_points
        )

    @torch.no_grad()
    def warp_points_3d(self, points, source_t, target_t):
        assert (source_t, target_t) in self.deformation_graphs.keys()
        deform_graph = self.deformation_graphs[(source_t, target_t)].to(
            "cuda:0"
        )
        return (
            deform_graph.warp_points(
                torch.tensor(points, dtype=torch.float32, device="cuda:0")
            )
            .detach()
            .cpu()
            .numpy()
        )
