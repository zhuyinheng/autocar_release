"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

from typing import List

import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes

from src.utils.camera import Camera


class MeshRenderer(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def _get_setting_mask_only(self, image_size):
        return RasterizationSettings(
            image_size=image_size,
            blur_radius=0,
            faces_per_pixel=1,
        )

    def compose_cam_list(self, cam_list: List[Camera]):

        pytorch3d_cams = PerspectiveCameras(
            R=torch.stack(
                [torch.from_numpy(cam.R_pytorch3d) for cam in cam_list], 0
            ),
            T=torch.stack(
                [torch.from_numpy(cam.t_pytorch3d) for cam in cam_list], 0
            ),
            K=torch.stack(
                [torch.from_numpy(cam.K_pytorch3d) for cam in cam_list], 0
            ),
            image_size=torch.stack(
                [
                    torch.tensor([cam.rows, cam.columns], dtype=torch.int32)
                    for cam in cam_list
                ],
                0,
            ),
            in_ndc=False,
        )
        return pytorch3d_cams

    @torch.no_grad()
    def render_mask_depth_multi_mesh_multi_cam(self, meshes, cams_list):
        import numpy as np

        mask = []
        depth = []
        for t in range(len(meshes)):
            _mask, _depth = self.render_mask_depth(meshes[t], cams_list)
            _mask = _mask.cpu().numpy()
            _depth = _depth.cpu().numpy()
            mask.append(_mask[0])
            depth.append(_depth[0])
        mask = np.stack(mask, 0)
        depth = np.stack(depth, 0)
        return mask, depth

    @torch.no_grad()
    def render_mask_depth(self, meshes: Meshes, cam_list: List[Camera]):
        """
        cam_list: B * view_num [A_0,B_0,A_1,B_1]
        """
        batch_num = meshes._N
        assert batch_num == 1
        image_num = len(cam_list)
        meshes = meshes.extend(image_num)
        # view_num = image_num // 2
        cams = self.compose_cam_list(cam_list)
        cams = cams.to(meshes.device)
        raster_settings = self._get_setting_mask_only(512)

        rasterizer = MeshRasterizer(
            cameras=cams, raster_settings=raster_settings
        )

        fragments = rasterizer(meshes)
        depth = fragments.zbuf[..., 0].permute(0, 2, 1).flip([1, 2])
        rows, columns = depth.shape[1:]
        depth = depth.view([batch_num, image_num, rows, columns])
        mask = depth > 0
        return mask, depth
