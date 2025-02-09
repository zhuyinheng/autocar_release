"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import numpy as np
import torch
import json


def cam_list_to_tensor(cam_list, device):

    world2pix4x4 = []
    for cam in cam_list:
        world2pix4x4.append(cam.world2pix4x4)
    world2pix = np.stack(world2pix4x4, 0)
    return torch.tensor(world2pix, dtype=torch.float32, device=device)


class Camera:
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    rows: int
    columns: int
    d_rows: float
    d_columns: float
    """
    world frame:
    1. origin: isotencenter
    2. x-axis: to foot
    3. y-axis: to back
    4. z-axis: to right
    """

    def __init__(
        self,
        primary_angle,
        secondary_angle,
        SID,
        SOD,
        d_rows,
        d_columns,
        rows,
        columns,
    ):
        """
        primary_angle: rotation angle around the x-axis in degree
        secondary_agnle: rotation angle around the z-axis in degree
        self.R@P_world + self.t = P_camera
        """
        primary_angle = np.deg2rad(primary_angle)
        secondary_angle = -np.deg2rad(secondary_angle)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(primary_angle), -np.sin(primary_angle)],
            [0, np.sin(primary_angle), np.cos(primary_angle)],
        ])
        R_z = np.array([
            [np.cos(secondary_angle), -np.sin(secondary_angle), 0],
            [np.sin(secondary_angle), np.cos(secondary_angle), 0],
            [0, 0, 1],
        ])
        R_active = R_x @ R_z
        R_passive = R_active.T
        self.t = np.array([0, 0, SOD])
        swap_axis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.R = swap_axis @ R_passive
        self.K = np.array([
            [SID / d_rows, 0, rows / 2],
            [0, SID / d_columns, columns / 2],
            [0, 0, 1],
        ])
        self.d_rows = d_rows
        self.d_columns = d_columns
        self.rows = rows
        self.columns = columns

    def init_from_dicom_meta(
        self,
        primary_angle,
        secondary_angle,
        SID,
        SOD,
        d_rows,
        d_columns,
        rows,
        columns,
    ):
        self.__init__(
            primary_angle,
            secondary_angle,
            SID,
            SOD,
            d_rows,
            d_columns,
            rows,
            columns,
        )

    @property
    def world2camera4x4(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def world2pix4x4(self):
        T = self.world2camera4x4
        K = self.K_pytorch3d
        return K @ T

    @property
    def camera2world4x4(self):
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, 3] = -self.R.T @ self.t
        return T

    @property
    def R_pytorch3d(self):

        swap_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # swap_axis = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        R = swap_axis @ self.R
        R = R.T
        return R.astype(np.float32)

    @property
    def t_pytorch3d(self):
        return self.t.astype(np.float32)

    @property
    def K_pytorch3d(self):
        """
        according to pytorch 3d

        K = [
                [fx,   0,   px,   0],
                [0,   fy,   py,   0],
                [0,    0,    0,   1],
                [0,    0,    1,   0],
        ]
        """
        K = np.zeros([4, 4], dtype=np.float32)
        K[:3, :3] = self.K.astype(np.float32)
        K[2, 2] = 0
        K[3, 2] = 1
        K[2, 3] = 1
        return K

    def toJSON(self):
        datadict = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.__dict__.items()
        }
        return json.dumps(datadict, indent=4)

    def save(self, fn):
        with open(fn, "w") as f:
            f.write(self.toJSON())

    @classmethod
    def dummy_cam(cls):
        return cls(0, 0, 1200, 800, 0.25, 0.25, 512, 512)

    @classmethod
    def fromJSON(cls, s):
        cam = cls.dummy_cam()
        for k, v in cam.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(cam, k, np.array(s[k]))
            else:
                setattr(cam, k, s[k])
        return cam

    @classmethod
    def fromFILE(cls, fn):
        with open(fn, "r") as f:
            s = json.load(f)
        return cls.fromJSON(s)

    def _points_world2camera_np(self, points_world):
        """
        points_world: N x 4
        """
        if points_world.ndim == 1:
            points_world = points_world[None, :]
        if points_world.shape[1] == 3:
            points_world = np.concatenate(
                [points_world, np.ones((points_world.shape[0], 1))], axis=1
            )

        assert points_world.ndim == 2
        assert points_world.shape[1] == 4

        return np.einsum("ij,nj->ni", self.world2camera4x4, points_world)

    def points_world2camera(self, points_world):
        return self._points_world2camera_np(points_world)

    def points_world2pix(self, points_world):
        if isinstance(points_world, np.ndarray):
            return self._points_world2pix_np(points_world)
        elif isinstance(points_world, torch.Tensor):
            return self._points_world2pix_torch(points_world)

    def _points_world2pix_torch(self, points_world: torch.Tensor):
        world2pix4x4 = torch.tensor(
            self.world2pix4x4, dtype=torch.float32, device=points_world.device
        )

        _points_world = points_world

        if _points_world.dim == 1:
            _points_world = _points_world[None, :]
        if _points_world.shape[1] == 3:
            _points_world = torch.cat(
                [_points_world, torch.ones_like(_points_world[:, :1])], dim=1
            )

        _points_pix = torch.einsum("ij,nj->ni", world2pix4x4, _points_world)
        _points_pix = _points_pix[:, :2] / _points_pix[:, -1:]

        return _points_pix

    def _points_world2pix_np(self, points_world):
        if points_world.ndim == 1:
            points_world = points_world[None, :]
        if points_world.shape[1] == 3:
            points_world = np.concatenate(
                [points_world, np.ones((points_world.shape[0], 1))], axis=1
            )
        points_camera = self.points_world2camera(points_world)
        points_pix = self._points_camera2pix_np(points_camera)
        return points_pix[:, :2] / points_pix[:, -1:]

    def _points_camera2pix_np(self, points_camera):
        """
        points_camera: N x 3
        """

        if points_camera.ndim == 1:
            points_camera = points_camera[None, :]
        if points_camera.shape[1] == 4:
            points_camera = points_camera[:, :3] / (points_camera[:, 3:])

        return np.einsum("ij,nj->ni", self.K, points_camera)

    def points_camera2pix(self, points_camera):
        return self._points_camera2pix_np(points_camera)

    def image_world2pix(self, points_world):

        if points_world.ndim == 1:
            points_world = points_world[None, :]
        if points_world.shape[1] == 3:
            points_world = np.concatenate(
                [points_world, np.ones((points_world.shape[0], 1))], axis=1
            )
        # pix = (self.world2pix4x4 @ (points_world.T)).T
        pix = np.einsum(
            "ij,nj->ni",
            self.world2pix4x4,
            points_world,
        )
        z = pix[:, -1] / pix[:, -2]
        pix = pix[:, :2] / pix[:, -1:]

        pix = pix.round().astype(np.int32)
        return self.pix2image(pix, z)

    def pix2image(self, points_pix, z):
        grid = np.zeros([self.rows, self.columns], dtype=np.float32)

        points_pix_mask = np.ones_like(points_pix[:, 0]) > 0
        points_pix_mask &= (points_pix[:, 0] >= 0) & (
            points_pix[:, 0] < self.rows
        )
        points_pix_mask &= (points_pix[:, 1] >= 0) & (
            points_pix[:, 1] < self.columns
        )
        points_pix = points_pix[points_pix_mask]
        grid[points_pix[:, 0], points_pix[:, 1]] = z[points_pix_mask]
        return grid

    def image_world2depth(self, points_world):
        points_camera = self.points_world2camera(points_world)
        points_pix = self.points_camera2pix(points_camera)
        z = points_pix[:, -1]
        points_pix = points_pix / points_pix[:, -1:]
        points_pix = points_pix.round().astype(np.int32)[:, :2]
        return self.pix2image(points_pix, z)

    def image_world_axis(self):
        line = np.linspace(0, 100, 256)
        x_axis = np.stack([line, np.zeros_like(line), np.zeros_like(line)], -1)
        y_axis = np.stack([np.zeros_like(line), line, np.zeros_like(line)], -1)
        z_axis = np.stack([np.zeros_like(line), np.zeros_like(line), line], -1)
        x = self.image_world2depth(x_axis) > 0
        y = self.image_world2depth(y_axis) > 0
        z = self.image_world2depth(z_axis) > 0
        return (np.stack([x, y, z], -1) * 255).astype(np.uint8)

    def point_world_camera(self, point_world):
        return self.R @ point_world + self.t

    def point_camera_world(self, point_camera):
        return self.R.T @ (point_camera - self.t)

    def camera_position(self):
        return self.point_camera_world(np.array([0, 0, 0]))

    def camera_frame(self):
        o = self.camera_position()
        x = self.point_camera_world(np.array([1, 0, 0]))
        y = self.point_camera_world(np.array([0, 1, 0]))
        z = self.point_camera_world(np.array([0, 0, 1]))
        return o, x - o, y - o, z - o

    def opengl_camera_pose(self):
        T = np.eye(4)
        swap_axis = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])
        R = swap_axis @ self.R
        T[:3, :3] = R.T
        T[:3, 3] = self.camera_position()
        return T
