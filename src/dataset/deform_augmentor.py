"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import torch
import numpy as np
import trimesh
from typing import List
from pytorch3d.structures import Meshes


class SinDeformAugmentor(torch.nn.Module):
    """
    Deform the input mesh with a sin function.
    """

    amplitude: float  # the amplitude of the sin function
    period: float  # @ the period of the sin function
    dummy_param: (
        torch.nn.Parameter
    )  # a dummy parameter to make the module detecting the device

    def __init__(self, amplitude=0.05, period=1) -> None:
        super().__init__()
        self.amplitude = amplitude
        self.period = period
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def _deform(self, mesh: trimesh.Trimesh, t: float):
        if self.amplitude == 0:
            return mesh
        else:
            new_mesh = mesh.copy()
            new_mesh.vertices *= 1 + self.amplitude * np.sin(
                2 * np.pi * t / self.period
            )
            return new_mesh

    def _trimesh2pytorch3d(self, mesh_list: List[trimesh.Trimesh]):
        device = self.dummy_param.device
        meshes = Meshes(
            verts=[
                torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
                for mesh in mesh_list
            ],
            faces=[
                torch.tensor(mesh.faces, dtype=torch.int32, device=device)
                for mesh in mesh_list
            ],
        )
        return meshes

    @torch.no_grad()
    def multi_time_forward(
        self, mesh: trimesh.Trimesh, t_start: float, t_end: float, n_steps: int
    ):
        multi_time_meshs = []
        for t in np.linspace(t_start, t_end, n_steps):
            multi_time_meshs.append(self._deform(mesh, t))
        return multi_time_meshs

    @torch.no_grad()
    def forward(self, mesh_list: List[trimesh.Trimesh], t: float = 0):
        deformed_mesh_list: List[trimesh.Trimesh] = [
            self._deform(mesh, t) for mesh in mesh_list
        ]
        return self._trimesh2pytorch3d(mesh_list), self._trimesh2pytorch3d(
            deformed_mesh_list
        )
