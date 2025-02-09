"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import numpy as np
import rootutils
import torch
from scipy.spatial import KDTree

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.so3 import so3_exp_map


class DeformationGraph(torch.nn.Module):
    """
    pytorch implementation of deformation graph
    Reference:
    [1] https://github.com/hwdong/deformation_graph
    [2] https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
    """

    def __init__(self, points: np.ndarray, sigma=5, k=4):
        torch.nn.Module.__init__(self)
        self.sigma = sigma
        self.points = points.astype(np.float32)
        # N x 3
        self.build_deformation_graph(sigma, k)

    def build_deformation_graph(self, sigma=5, k=4):

        self.control_points = (
            np.unique(np.ceil(self.points / sigma).astype(np.int32), axis=0)
            * sigma
        ).astype(np.float32)
        # M x k
        M = self.control_points.shape[0]
        node_tree = KDTree(self.control_points)
        _, self.points_to_control_points = node_tree.query(self.points)
        # self.points_to_control_points: N
        _, control_points_knn = node_tree.query(self.control_points, k)
        # control_points_edges_u: M x K
        control_points_edges_u = np.linspace(
            0,
            self.control_points.shape[0] - 1,
            self.control_points.shape[0],
            endpoint=True,
            dtype=np.int64,
        )[:, None].repeat(k, axis=-1)
        # control_points_edges_u: M x k
        control_points_edges_v = control_points_knn
        control_points_edges = np.stack(
            [control_points_edges_u, control_points_edges_v], -1
        )
        self.control_points_edges = control_points_edges.reshape([
            M * k,
            2,
        ]).astype(np.int64)
        # M*k x 2
        self.rot_vec = torch.nn.Parameter(
            torch.zeros(
                [self.control_points.shape[0], 3], dtype=torch.float32
            ),
            requires_grad=True,
        )
        self.t = torch.nn.Parameter(
            torch.zeros(
                [self.control_points.shape[0], 3], dtype=torch.float32
            ),
            requires_grad=True,
        )
        self.control_points_pth = torch.nn.Parameter(
            torch.tensor(self.control_points),
            requires_grad=False,
        )
        self.control_points_edges_pth = torch.nn.Parameter(
            torch.tensor(self.control_points_edges).type(torch.int32),
            requires_grad=False,
        )

    def parameters(self, recurse: bool = True):

        for name, param in self.named_parameters(recurse=recurse):
            if name == "rot_vec" or name == "t":
                yield param

    def warp_points(self, points: torch.Tensor):

        local_points = points[:, None, :] - self.control_points_pth[None, :, :]
        # local_points: N x M x 3, points in local coordinate at each control point
        skinning_weights = torch.exp(
            -((local_points**2).sum(-1)) / (2 * self.sigma * self.sigma)
        )
        # skinning_weights: N x M, gaussian kernel according to paper

        skinning_weights = skinning_weights / (
            skinning_weights.sum(-1)[:, None] + 1e-5
        )
        # N x M

        warped_points = (
            torch.einsum(
                "nij,pnj -> pni", so3_exp_map(self.rot_vec), local_points
            )
            + self.t[None, :, :]
            + self.control_points_pth[None, :, :]
        )
        # R@local_points + t', t' = self.t +self.control_points_pth. it's a residual form
        # N x M x 3

        warped_points = (warped_points * (skinning_weights[..., None])).sum(1)
        # N x 3
        return warped_points

    def reg_term(self):
        control_points_i = self.control_points_pth[
            self.control_points_edges[:, 0], :
        ]
        control_points_j = self.control_points_pth[
            self.control_points_edges[:, 1], :
        ]
        t_i = self.t[self.control_points_edges[:, 0], :]
        t_j = self.t[self.control_points_edges[:, 1], :]
        # M x 3
        return (
            (
                torch.einsum(
                    "mij,mj -> mi",
                    so3_exp_map(
                        self.rot_vec[self.control_points_edges[:, 0], :]
                    ),
                    control_points_j - control_points_i,
                )
                + control_points_i
                + t_i
                - (control_points_j + t_j)
            )
            ** 2
        ).sum()
