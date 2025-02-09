"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import numpy as np
from src.utils.camera import Camera


class CameraPairSampler:
    """
    CameraPairSampler is a class that samples a pair of cameras with random positions.
    The distribution of DSA camera posed is private.
    here we use the fitted distribution available in the paper.
    """

    cate_center: np.ndarray  # 9 x 2,float32, the center of the 9 categories
    sigma: 10  # the sigma of the gaussian distribution
    random_state = np.random.RandomState(888)

    def __init__(self):
        self.cate_center = (
            np.stack(
                np.meshgrid(
                    np.linspace(-1, 1, 3, endpoint=True),
                    np.linspace(-1, 1, 3, endpoint=True),
                    indexing="ij",
                ),
                -1,
            ).reshape(-1, 2)
            * 60
        )

    def sample(self):
        sampled_cate = self.random_state.choice(
            self.cate_center.shape[0], 2, replace=False
        )

        mean = self.cate_center[sampled_cate, :]
        # mean: 2 x 2
        std_gaussian = self.random_state.randn(2, 2)
        pa_sa = self.sigma * std_gaussian + mean
        cam1 = Camera(
            pa_sa[0, 0], pa_sa[0, 1], 1200, 800, 0.25, 0.25, 512, 512
        )
        cam2 = Camera(
            pa_sa[1, 0], pa_sa[1, 1], 1200, 800, 0.25, 0.25, 512, 512
        )
        return [cam1, cam2]

    def sample_batch(self, batch_size):
        cam_list = []
        for i in range(batch_size):
            cam_list.append(self.sample())
        return cam_list
