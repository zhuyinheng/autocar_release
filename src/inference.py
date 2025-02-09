"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import hydra
import imageio
import numpy as np
import rootutils
import torch
import tqdm
import trimesh
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.curvenetwork_reconstructor import DeformableCurvenetworkReconstructor
from src.modules.autocar import AutoCAR
from src.utils import RankedLogger, extras
from src.utils.camera import Camera, cam_list_to_tensor
from src.utils.napari_utils import (
    napari_add_mesh_video,
    napari_add_point_cloud_video,
)

log = RankedLogger(__name__, rank_zero_only=True)


class AutoCARReconstructor:
    """
    voxel_spacing: 3, float32, [dx, dy, dz]
    min_bound: 3, float32, [x, y, z]
    view_num: int, number of views
    time_num: int, number of time steps

    coordinate system: world_points=(voxel_idx+0.5) * voxel_spacing + min_bound
    """

    recon_network: AutoCAR
    deformable_reconstructor: DeformableCurvenetworkReconstructor
    voxel_spacing: np.ndarray
    min_bound: np.ndarray
    cfg: DictConfig
    view_num: int
    time_num: int
    vis_during_reconstruction: True

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.deformable_reconstructor = DeformableCurvenetworkReconstructor(
            device=cfg.inferece_device
        )
        self.voxel_spacing = np.array(
            [self.cfg.model.recon_net.ray_casting.LODs[-1]] * 3
        )
        self.min_bound = np.array(
            self.cfg.model.recon_net.ray_casting.bbox_min
        )
        self.view_num = cfg.view_num
        self.time_num = cfg.time_num
        self.device = cfg.inferece_device
        self.selected_view_pair = (
            cfg.selected_view_pair
        )  # you can modify it in configs/eval.yaml

    def reconstruct(self):
        log.info("Reconstructing...")
        self.data_loading()
        self.network_loading()
        self.network_forward()
        for t_idx in tqdm.tqdm(
            range(self.time_num), "init deformable curve network"
        ):
            self.deformable_reconstructor.init_time_step(
                t_idx,
                self.sparse_volume_C[t_idx],
                self.sparse_volume_F[t_idx],
                self.voxel_spacing,
                self.min_bound,
                self.cameras[t_idx],
                self.imgs[t_idx],
            )
        for t_idx in tqdm.tqdm(range(self.time_num - 1), "optimize"):
            self.deformable_reconstructor.optimize(t_idx, t_idx + 1)

        log.info("Reconstruction Finished")

        log.info(f"Saving Results in {self.cfg.paths.output_dir} - 3D points")
        pts_time = []
        pts_t = self.deformable_reconstructor.deformable_curve_network[
            0
        ].high_quality_graph_pts
        for t_idx in tqdm.tqdm(
            range(self.time_num), "Saving Results - 3D points"
        ):

            trimesh.PointCloud(pts_t).export(
                f"{self.cfg.paths.output_dir}/centerlines_{t_idx}.ply"
            )
            pts_time.append(pts_t)
            if t_idx == self.time_num - 1:
                break
            pts_t = self.deformable_reconstructor.warp_points_3d(
                pts_t, t_idx, t_idx + 1
            )
        if self.vis_during_reconstruction:
            log.info("Visualizing Point Cloud Video")
            import napari

            viewer = napari.Viewer()
            for t in range(self.t_num):
                viewer.add_points(
                    self.deformable_reconstructor.deformable_curve_network[
                        t
                    ].points,
                    name=f"static_t{t}",
                    size=1,
                )
                viewer.add_points(pts_time[t], name=f"optimized_t{t}", size=1)
            napari.run()

        log.info(
            f"Saving Results in {self.cfg.paths.output_dir}- Surface Mesh"
        )
        mesh = self.deformable_reconstructor.deformable_curve_network[0].mesh
        meshs_time = [mesh]
        for t_idx in tqdm.tqdm(
            range(self.time_num - 1), "Saving Results - Surface Mesh"
        ):

            log.info("Visualizing Point Cloud Video")
            v_t = self.deformable_reconstructor.warp_points_3d(
                meshs_time[t_idx].vertices, t_idx, t_idx + 1
            )
            mesh_t = meshs_time[t_idx].copy()
            mesh_t.vertices = v_t

            mesh_t.export(f"{self.cfg.paths.output_dir}/surface_{t_idx}.ply")
            meshs_time.append(mesh_t)

        if self.vis_during_reconstruction:
            import napari

            viewer = napari.Viewer()
            napari_add_point_cloud_video(
                viewer, pts_time, "centerlines", use_psudo_color=True
            )
            napari_add_mesh_video(
                viewer, meshs_time, "Surface", use_psudo_color=True
            )
            napari.run()

    @torch.no_grad()
    def network_forward(self):
        log.info("Forwarding...")
        self.sparse_volume_C = []
        self.sparse_volume_F = []
        self.sparse_volume_C_world_coordinates = []
        for t_idx in tqdm.tqdm(range(self.time_num), "network_forward"):
            masks = np.stack(self.imgs[t_idx], 0)[None] / 255.0
            masks = torch.tensor(
                masks, dtype=torch.float32, device=self.device
            )
            world2pix4x4 = cam_list_to_tensor(self.cameras[t_idx], self.device)

            pred, world_coordinates = self.recon_network.forward(
                masks, world2pix4x4[None, ...]
            )
            pred.F[:, 0] = torch.sigmoid(pred.F[:, 0])
            # viewing_sparse_volume(pred)
            self.sparse_volume_C.append(pred.C[:, 1:].cpu().numpy())
            self.sparse_volume_F.append(pred.F.cpu().numpy())
            self.sparse_volume_C_world_coordinates.append(
                world_coordinates.cpu().numpy()
            )

    def data_loading(self):
        log.info("Loading data...")
        testset_case_root = self.cfg.testset_case_root
        select_view_pair = self.select_view_pair
        cams = []
        imgs = []
        for t_idx in range(self.time_num):
            _cams = []
            _imgs = []
            for v_idx in select_view_pair:
                _imgs.append(
                    imageio.imread(
                        f"{testset_case_root}/mask_{t_idx:02d}_{v_idx:02d}.png"
                    )
                )
                _cams.append(
                    Camera.fromFILE(
                        f"{testset_case_root}/cam_{t_idx:02d}_{v_idx:02d}.json"
                    )
                )
            cams.append(_cams)
            imgs.append(_imgs)

        self.imgs = imgs
        self.cameras = cams

    def network_loading(self):
        log.info("Loading network...")
        self.recon_network = AutoCAR(self.cfg.model.recon_net)
        assert self.cfg.ckpt_path
        state_dict = torch.load(self.cfg.ckpt_path, map_location="cpu")[
            "state_dict"
        ]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.find("recon_net.") != -1:
                new_key = key.replace("recon_net.", "")
                new_state_dict[new_key] = value
        self.recon_network.load_state_dict(new_state_dict)
        self.recon_network.eval()
        self.recon_network.to(self.device)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for inferece.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    reconstructor = AutoCARReconstructor(cfg)
    reconstructor.reconstruct()


if __name__ == "__main__":
    main()
