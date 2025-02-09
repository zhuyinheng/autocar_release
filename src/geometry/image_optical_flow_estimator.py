"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import cv2
import numpy as np


class ImageOpticalFlowEstimator:
    """
    Estimate, store the optical flow between two images, and provide the warping function.

    Esitmation direction: A -> B
    Warping direction: B = warp(A)

    Reference: https://answers.opencv.org/question/186403/how-to-apply-optical-flow-to-initial-image/
    """

    def __init__(self):

        self.flow_estimator = cv2.DISOpticalFlow.create(
            cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
        )

    def estimate_flow(self, A, B):
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
        self.flow = self.flow_estimator.calc(A, B, None)

    def warp_image(self, A):

        h, w = self.flow.shape[:2]
        flow = -self.flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(A, flow, None, cv2.INTER_LINEAR)
        return res

    def warp_points(self, points, ordered=True):
        points = (
            points.round().astype(np.int32).clip(0, self.flow.shape[0] - 1)
        )
        if ordered:
            warped_pts = (
                points + self.flow[points[:, 0], points[:, 1]][:, ::-1]
            )
        else:
            # Reference Vid2Curve
            points = (
                points.round().astype(np.int32).clip(0, self.flow.shape[0] - 1)
            )
            image: np.ndarray[np.Any, np.dtype[np.floating[np._64Bit]]] = (
                np.zeros([self.flow.shape[0], self.flow.shape[1]])
            )
            image[points[:, 0], points[:, 1]] = 1
            warped_pts = np.stack(np.where(self.warp_image(image) > 0), -1)
        return warped_pts
