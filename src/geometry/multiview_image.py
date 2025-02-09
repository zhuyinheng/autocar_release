"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

from typing import Dict, Tuple

import numpy as np
import rootutils
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.geometry.image_optical_flow_estimator import ImageOpticalFlowEstimator
from src.utils.camera import Camera


class PosedImage:
    """
    PosedImage is a class that computes and stores the image, camera, and centerline information.
    """

    image: np.ndarray = None
    camera: Camera = None
    mask: np.ndarray = None
    quality_checking_radius = 5
    quality_checking_min_nn_num = 3
    high_quality_thred = 0.95
    _centerline_image = None
    _centerline_points = None
    _centerline_points_quality = None
    _centerline_points_tangent = None

    def __init__(
        self,
        image: np.ndarray,
        camera,
        mask=None,
        quality_checking_radius=5,
        quality_checking_min_nn_num=3,
        high_quality_thred=0.95,
    ):
        if image.ndim == 2:

            image = image[:, :, None].repeat(3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = image.repeat(3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
                pass
        self.image = image
        self.camera = camera
        if mask is None:
            self.mask = self._seg_mask_from_image()
        else:
            self.mask = mask > 0
        self.quality_checking_radius = quality_checking_radius
        self.quality_checking_min_nn_num = quality_checking_min_nn_num
        self.high_quality_thred = high_quality_thred

    @property
    def centerline_image(self):
        if getattr(self, "_centerline_image", None) is None:
            self._cache_centerline_image()
        return self._centerline_image

    @property
    def centerline_points(self):
        if getattr(self, "_centerline_points", None) is None:
            self._cache_centerline_points()
        return self._centerline_points

    @property
    def centerline_points_tangent(self):
        if getattr(self, "_centerline_points_tangent", None) is None:
            self._cache_centerline_points_quality_tangent()
        return self._centerline_points_tangent

    @property
    def centerline_points_quality(self):
        if getattr(self, "_centerline_points_quality", None) is None:
            self._cache_centerline_points_quality_tangent()
        return self._centerline_points_quality

    @property
    def centerline_points_high_quality_mask(self):
        if getattr(self, "_centerline_points_quality", None) is None:
            self._cache_centerline_points_quality_tangent()
        return self.centerline_points_quality > self.high_quality_thred

    @property
    def high_quality_centerline_points(self):
        return self.centerline_points[
            self.centerline_points_high_quality_mask, :
        ]

    @property
    def high_quality_centerline_points_tangent(self):
        return self.centerline_points_tangent[
            self.centerline_points_high_quality_mask, :
        ]

    def _cache_centerline_points_quality_tangent(self):
        """
        Python Reimplementation of Tricks in Vid2Curve
        """
        N = self.centerline_points.shape[0]
        centerline_points = self.centerline_points
        centerline_points_kdtree = KDTree(centerline_points)

        centerline_points_with_nn = centerline_points_kdtree.query_ball_tree(
            centerline_points_kdtree, self.quality_checking_radius
        )
        # centerline_points_with_nn: [[idx in centerline_points,],],
        # [idx in centerline_points,] is the neighbor of the point,
        # len(centerline_points_with_nn) == centerline_points.shape[0]

        self._centerline_points_quality = np.zeros(N)
        self._centerline_points_tangent = np.zeros([N, 2])
        for anchor_i in range(N):
            M = len(centerline_points_with_nn[anchor_i])
            if M < self.quality_checking_min_nn_num:
                continue
            points_around_anchor_point = centerline_points[
                np.array(centerline_points_with_nn[anchor_i])
            ]
            _, s, vt = np.linalg.svd(
                points_around_anchor_point
                - points_around_anchor_point.mean(0, keepdims=True)
            )
            s /= M
            score = (s[0] ** 2 + 1e-3) / (s[0] ** 2 + s[1] ** 2 + 1e-3)
            if np.isnan(score):
                continue
            self._centerline_points_quality[anchor_i] = score
            self._centerline_points_tangent[anchor_i] = vt[0]

    def _cache_centerline_image(self):
        self._centerline_image = skeletonize(self.mask)
        # self._update_pts_from_skele()

    def _cache_centerline_points(self):
        self._centerline_points = np.stack(np.where(self.centerline_image), -1)

    def _seg_mask_from_image(self):
        mask = self.image.mean(-1)
        mask = mask > mask.mean()
        return mask

    def vis(self, fn=None):
        import matplotlib.pyplot as plt

        plt.imshow(self.image)
        plt.imshow(self.centerline_image, cmap="Reds", alpha=0.3)

        plt.quiver(
            self.high_quality_centerline_points[:, 1],
            self.high_quality_centerline_points[:, 0],
            self.high_quality_centerline_points_tangent[:, 1],
            self.high_quality_centerline_points_tangent[:, 0],
            angles="xy",
            alpha=0.5,
        )
        if fn is None:
            plt.show()
        else:
            plt.savefig(fn)
            plt.close()


class MultiViewVideo:
    """
    MultiViewVideo is a class that stores the Multi-view,Multi-time PosedImage.
    It also provide related functions including optical flow estimation and point warping.
    """

    _posed_images: Dict[Tuple[int, int], PosedImage] = {}
    _optical_flows: Dict[
        Tuple[int, int, int, int], ImageOpticalFlowEstimator
    ] = {}

    def __init__(self):

        pass

    def update_image_at_time_view(
        self,
        time_idx: int,
        view_idx: int,
        *args,
    ):
        if isinstance(args[0], PosedImage):
            self._update_from_posed_image(time_idx, view_idx, args[0])
        elif isinstance(args[0], np.ndarray) and isinstance(args[1], Camera):
            self._update_from_image_camera(
                time_idx,
                view_idx,
                *args,
            )
        else:
            raise NotImplementedError

    @property
    def unique_view_idxs(self):
        t_v_keys = list(self._posed_images.keys())
        return np.unique([t_v[1] for t_v in t_v_keys])

    def high_quality_centerline_points_fix_time_loop_view(self, time_idx):
        points = {}
        for view_idx in self.unique_view_idxs:
            points[view_idx] = self._posed_images[
                (time_idx, view_idx)
            ].high_quality_centerline_points
        return points

    def high_quality_centerline_points_tangent_fix_time_loop_view(
        self, time_idx
    ):
        tangent = {}
        for view_idx in self.unique_view_idxs:
            tangent[view_idx] = self._posed_images[
                (time_idx, view_idx)
            ].high_quality_centerline_points_tangent
        return tangent

    def points_world2pix_fix_time_loop_view(self, points, time_idx):
        pixs = {}
        for view_idx in self.unique_view_idxs:
            pixs[view_idx] = self._posed_images[
                (time_idx, view_idx)
            ].camera.points_world2pix(points)
        return pixs

    def warp_points_3d(
        self, pts_3d: np.ndarray, source_time_idx: int, target_time_idx: int
    ):
        """
        pts_3d: N x 3
        """
        multi_view_warped_pts_2d = {}
        for view_idx in self.unique_view_idxs:
            if (
                self._posed_images.get((source_time_idx, view_idx), None)
                is None
            ):
                continue
            if (
                self._posed_images.get((target_time_idx, view_idx), None)
                is None
            ):
                continue
            multi_view_warped_pts_2d[view_idx] = self.warp_pts_3d_single_view(
                pts_3d, source_time_idx, target_time_idx, view_idx
            )
        return multi_view_warped_pts_2d

    def warp_pts_3d_single_view(
        self,
        pts_3d: np.ndarray,
        source_time_idx: int,
        target_time_idx: int,
        view_idx: int = None,
    ):
        """ """

        posed_image = self._posed_images[(source_time_idx, view_idx)]
        pts_2d = posed_image.camera.points_world2pix(pts_3d)
        optical_flow = self.get_optical_flow(
            source_time_idx, view_idx, target_time_idx, view_idx
        )
        warped_pts_2d = optical_flow.warp_points(pts_2d)
        return warped_pts_2d

    def get_optical_flow(
        self,
        source_time_idx,
        source_view_idx,
        target_time_idx,
        target_view_idx,
    ):
        if (
            self._optical_flows.get(
                (
                    source_time_idx,
                    source_view_idx,
                    target_time_idx,
                    target_view_idx,
                ),
                None,
            )
            is None
        ):
            self._cache_optical_flow(
                source_time_idx,
                source_view_idx,
                target_time_idx,
                target_view_idx,
            )
        return self._optical_flows[(
            source_time_idx,
            source_view_idx,
            target_time_idx,
            target_view_idx,
        )]

    def _cache_optical_flow(
        self,
        source_time_idx,
        source_view_idx,
        target_time_idx,
        target_view_idx,
    ):
        source_image = self._posed_images[
            (source_time_idx, source_view_idx)
        ].image
        target_image = self._posed_images[
            (target_time_idx, target_view_idx)
        ].image
        optical_flow = ImageOpticalFlowEstimator()
        optical_flow.estimate_flow(source_image, target_image)
        self._optical_flows[(
            source_time_idx,
            source_view_idx,
            target_time_idx,
            target_view_idx,
        )] = optical_flow

    def _update_from_posed_image(
        self, time_idx: int, view_idx: int, posed_image: PosedImage
    ):
        self._posed_images[(time_idx, view_idx)] = posed_image

    def _update_from_image_camera(
        self,
        time_idx: int,
        view_idx: int,
        image: np.ndarray,
        camera: Camera,
        *args,
    ):
        self._posed_images[(time_idx, view_idx)] = PosedImage(
            image, camera, *args
        )


if __name__ == "__main__":
    import imageio
    import matplotlib.pyplot as plt

    from src.utils.camera import Camera

    image = imageio.imread("src/tests/sample_mask.png")
    print(image.shape)
    camera = Camera(0, 0, 1200, 800, 0.5, 0.5, 512, 512)
    mvi = PosedImage(image, camera)
    mvi.vis()
    plt.show()
