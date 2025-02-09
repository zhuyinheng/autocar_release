"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

from typing import List
import numba
import numpy as np
import torch
from scipy.spatial import KDTree


@numba.jit(
    "void(float32[:,:],int32[:,:],float32[:,:],float32[:,:,:])",
    nopython=True,
)
def dp_func(
    f: np.ndarray,
    past_choice: np.ndarray,
    uni_dist: np.ndarray,
    pair_dist: np.ndarray,
):
    """
    numba implementation of following code snippet
    ```
    for i in range(1, N):

        cost = (
            f[i - 1 : i, :]  # 1 x k1, 0,1,2
            + uni_dist[i, None]  # k2 x 1, 1,2,3
            + pair_dist[i]  # k2 x k1, 1,2,3 x 0,1,2
        )
        # cost: k,i x k,i-1
        # cost[k1,k2]: cost of matching k1 in i to k2 in i-1,
        f[i] = cost.min(-1)

        # f[i,k1]: cost of matching k1 in i to any k2 in i-1
        past_choice[i] = np.argmin(cost, -1)
        # past_choice[i,k1]: index of k2 in i-1, that minimize cost of matching k1 in i to k2 in i-1
    ```
    """
    N = f.shape[0]
    K = f.shape[1]
    for i in range(1, N):
        cost = (
            f[i - 1 : i, :]  # 1 x k1, 0,1,2
            + uni_dist[i, None]  # k2 x 1, 1,2,3
            + pair_dist[i]  # k2 x k1, 1,2,3 x 0,1,2
        )
        # cost: k,i x k,i-1
        # cost[k1,k2]: cost of matching k1 in i to k2 in i-1,
        for j in range(K):
            f[i, j] = cost[i].min()
            # f[i,k1]: cost of matching k1 in i to any k2 in i-1
            past_choice[i, j] = np.argmin(cost[j])
            # past_choice[i,k1]: index of k2 in i-1, that minimize cost of matching k1 in i to k2 in i-1


@numba.jit(nopython=True)
def backtrace_func(c, path, past_choice, pts1_knn_in_pts2):
    """
    numba implementation of following code snippet
    ```
    for i in np.linspace(N - 1, 0, N, dtype=np.int32):
        path[i] = pts1_knn_in_pts2[i, c]
        c = past_choice[i, c]
    ```
    """
    for i in range(path.shape[0] - 1, -1, -1):
        path[i] = pts1_knn_in_pts2[i, c]
        c = past_choice[i, c]


def curve_scatter_matching(
    curve_points: np.ndarray,
    scatter_points: np.ndarray,
    k=10,
    numba_speed_up=False,
):
    """
    Matching curve points to scatter points, with dynamic programming.

    pts1: N x D, oredered D-dimentional points in curve
    pts2: M x D, unordered D-dimentional points in scatter
    k: int, top-k nearest neighbor to search in scatter points for each curve point
    numba_speed_up: bool, whether to use numba to speed up the dynamic programming

    Tips: making sure the correct matched points in pts2, is top-k nearest neighbor of pts1 in pts2
    """
    N = curve_points.shape[0]
    if isinstance(scatter_points, KDTree):
        scatter_points_kdtree = scatter_points
        scatter_points = scatter_points.data
    else:
        scatter_points_kdtree = KDTree(scatter_points)
    # NOTE: building KDTree is time consuming, should be reused if possible.

    _, scatter_idxs_near_curve_points = scatter_points_kdtree.query(
        curve_points, k=k
    )
    # N x k, for each point in curve_points, k nearest neighbor in scatter_points
    # e.g. scatter_points[scatter_idxs_near_curve_points[0]] is the k nearest neighbor of curve_points[0]
    # if scatter_idxs_near_curve_points[i,j] == M, means no neighbor found

    # NOTE: An alternative way is use query_ball_point,
    # meaning that the neighbor is defined by radius instead of ranking.
    # radius query is hard to vectorize, since the number of neighbor is not fixed.

    scatter_points_cat = np.concatenate(
        [scatter_points, np.ones_like(scatter_points[0:1, :]) * 1e9], 0
    )
    # M+1 x D, add a dummy point to avoid out of index

    dual_points = scatter_points_cat[scatter_idxs_near_curve_points, :]
    # N x k x D
    # dual points is the k nearest neighbor of each curve points in scatter points

    uni_dist = np.linalg.norm(
        curve_points[:, None, :] - dual_points,
        axis=-1,
    ).astype(np.float32)
    # N x k

    # Caching the distance between each pair of points
    curve_points_diff = np.diff(curve_points, axis=0)[:, None, None, :]
    # N-1 x 1 x 1 x D

    # Cache the difference between each pair of dual points
    dual_point_pair_diff = (
        dual_points[1:, :, None, :] - dual_points[:-1, None, :, :]
    )
    # N-1 x k x k x D
    # dual_point_pair_diff[i,k1,k2] = dual_points[i+1,k1] - dual_points[i,k2]

    pair_dist = np.linalg.norm(
        curve_points_diff - dual_point_pair_diff,
        axis=-1,
    )
    pair_dist = np.concatenate(
        [np.zeros_like(pair_dist[0:1]), pair_dist], 0
    ).astype(np.float32)
    # pair_dist: N x k2 x k1

    f = np.ones([N, k], dtype=np.float32) * 1e9
    past_choice = -np.ones([N, k], dtype=np.int32)
    f[0] = uni_dist[0]
    if numba_speed_up:
        dp_func(f, past_choice, uni_dist, pair_dist)
    else:
        for i in range(1, N):

            cost = (
                f[i - 1 : i, :]  # 1 x k1, 0,1,2
                + uni_dist[i, None]  # k2 x 1, 1,2,3
                + pair_dist[i]  # k2 x k1, 1,2,3 x 0,1,2
            )
            # cost: k x k
            # cost[k1,k2]: cost of matching k1 in i to k2 in i-1,
            f[i] = cost.min(-1)

            # f[i,k1]: cost of matching k1 in i to any k2 in i-1
            past_choice[i] = np.argmin(cost, -1)
            # past_choice[i,k1]: index of k2 in i-1, that minimize cost of matching k1 in i to k2 in i-1

    c = np.argmin(f[N - 1])
    path = np.zeros(N, dtype=np.int32)
    if numba_speed_up:
        backtrace_func(c, path, past_choice, scatter_idxs_near_curve_points)
    else:
        for i in np.linspace(N - 1, 0, N, dtype=np.int32):
            path[i] = scatter_idxs_near_curve_points[i, c]
            c = past_choice[i, c]
    path_pts = scatter_points[path, :]
    return path, path_pts


def curve_tangent(curve_points):

    from scipy.ndimage import gaussian_filter1d

    tangent = np.gradient(
        gaussian_filter1d(curve_points, sigma=3, axis=0), axis=0
    )
    return tangent / (np.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-6)


class Correspondence:

    dist_quality_threshold: float = 10
    tangent_quality_threshold: float = 1e-3

    fused_forward_map = None
    fused_dual_pts = None
    fused_dual_tangent = None
    fused_valid_mask = None

    def __init__(self) -> None:
        pass

    def fit(self, curve_points, paths: List[List[int]], scatter_points):
        """Finding correcpondence between a set of path and scatter points.
        the set of paths is a reprensetation of graph for computation.
        one curve point may exist in multiple paths.
        one curve point may be matched to multiple scatter points, and vice versa.

        curve_points: N x D,
        paths: P x L_i, P paths, each path has L_i curve points
        scatter_points: M x D
        """
        cache_torch = False
        if isinstance(curve_points, torch.Tensor):
            cache_torch = True
            cache_torch_device = curve_points.device
            curve_points = curve_points.detach().cpu().numpy()
        if isinstance(scatter_points, torch.Tensor):
            scatter_points = scatter_points.detach().cpu().numpy()

        self._fit(curve_points, paths, scatter_points)
        if cache_torch:
            self.convert_torch(cache_torch_device)

    def _fit(
        self,
        curve_points: np.ndarray,
        paths: List[List[int]],
        scatter_points: np.ndarray,
    ):
        N = curve_points.shape[0]
        # M = scatter_points.shape[0]
        P = len(paths)
        D = curve_points.shape[1]
        forward_map = -np.ones([N, P], dtype=np.int32)
        # forward_map[i,j]: index of matched scatter point in scatter_points, for curve point i in path j
        # forward_map[i,j] = -1, means no matched point

        dist_quality = np.ones([N, P]) * 1e9
        # dist_quality[i,j]: quality of matching curve point i in path j to scatter point
        # dist_quality[i,j] = 1e9, means no matched point

        dual_points = np.zeros([N, P, D])
        dual_tangent = np.zeros([N, P, D])
        # dual_pts[i,j]: matched scatter point in scatter_points, for curve point i in path j
        # dual_tangent[i,j]: tangent of matched scatter point in scatter_points, for curve point i in path j

        # Looping paths
        for path_idx in range(P):
            path = np.array(paths[path_idx])
            path_pts = curve_points[path]

            # path_pts quality check
            path_arc_length = np.linalg.norm(
                np.diff(path_pts, axis=0), axis=-1
            ).sum()
            if path_arc_length < 1e-3:
                continue

            matched_path, matched_path_pts = curve_scatter_matching(
                path_pts,
                scatter_points,
            )

            forward_map[path, path_idx] = np.array(matched_path)
            dual_points[path, path_idx] = matched_path_pts
            dist_quality[path, path_idx] = np.linalg.norm(
                path_pts - matched_path_pts, axis=-1
            )
            dual_tangent[path, path_idx] = curve_tangent(matched_path_pts)

        # Merging attributes from multiple paths into one
        # NOTE: one node may exist in many paths, we need to merge the attributes from different paths

        forward_map[dist_quality > self.dist_quality_threshold] = -1

        valid_path_num = (forward_map >= 0).sum(-1)
        # N, number of valid path for each curve point

        matched_curve_points_mask = valid_path_num > 0
        # N, whether the curve point is matched in any path

        dual_points_path_mean = (
            dual_points * (forward_map[:, :, None] >= 0)
        ).sum(1)
        dual_points_path_mean[matched_curve_points_mask, :] = (
            dual_points_path_mean[matched_curve_points_mask, :]
            / ((valid_path_num[matched_curve_points_mask, None]) + 1e-6)
        )
        # dual_points_path_mean: N x 3, mean of multiple matched scatter points(from different paths)
        # of each curve point
        dual_tangent_path_mean = (
            dual_tangent * (forward_map[:, :, None] >= 0)
        ).sum(1)
        dual_tangent_path_mean[matched_curve_points_mask, :] = (
            dual_tangent_path_mean[matched_curve_points_mask, :]
            / ((valid_path_num[matched_curve_points_mask, None]) + 1e-6)
        )
        # dual_tangent_path_mean: N x 3, mean of multiple matched scatter points(from different paths)

        dual_pts_path_std = (
            np.linalg.norm(
                dual_points - dual_points_path_mean[:, None, :], axis=-1
            )
            * (forward_map >= 0)
        ).sum(-1) / (valid_path_num + 1e-6)
        # dual_pts_path_std: N, std of multiple matched scatter points(from different paths)
        forward_map[dual_pts_path_std > 15, :] = -1
        forward_map[np.isnan(dual_tangent_path_mean).sum(-1) > 0, :] = -1

        self.fused_forward_map = forward_map.max(-1)
        self.fused_dual_pts = dual_points_path_mean
        self.fused_dual_tangent = dual_tangent_path_mean
        self.fused_valid_mask = self.fused_forward_map >= 0

    def convert_torch(self, device: torch.device):
        self.fused_forward_map = torch.tensor(
            self.fused_forward_map,
            dtype=torch.int32,
            device=device,
            requires_grad=False,
        )
        self.fused_dual_pts = torch.tensor(
            self.fused_dual_pts,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        self.fused_dual_tangent = torch.tensor(
            self.fused_dual_tangent,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        self.fused_valid_mask = (
            torch.tensor(
                self.fused_valid_mask,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )
            > 0
        )


if __name__ == "__main__":
    """
    Example of curve scatter matching showing Naive distance-based matching and DP-based matching
    """
    N = 100
    pts1 = np.stack([np.zeros(N), np.linspace(0, 100, N)], -1)
    pts2 = np.concatenate(
        [pts1 + np.array([[1, 0]]), pts1 + np.array([[-1, 0]])]
    )
    pts2 = KDTree(pts2)
    path, path_pts = curve_scatter_matching(
        pts1, pts2, numba_speed_up=True, k=5
    )
    pts2 = pts2.data
    import matplotlib.pyplot as plt

    plt.scatter(pts1[:, 0], pts1[:, 1], label="curve")
    plt.scatter(pts2[:, 0], pts2[:, 1], label="scatter")
    plt.scatter(path_pts[:, 0], path_pts[:, 1], label="matched scatter")
    plt.show()
