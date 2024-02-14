#!/usr/lib/bin/env/ python

import numpy as np
import cv2
from matplotlib import pyplot as plt

fc = 718.8560
pp = (607.1928, 185.2157)


def featureTracking(image_1, image_2, points_1):
    # Set Lucas-Kanade Params
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # Calculate Optical Flow
    points_2, status, err = cv2.calcOpticalFlowPyrLK(
        image_1, image_2, points_1, None, **lk_params
    )
    status = status.reshape(status.shape[0])
    points_1 = points_1[status == 1]
    points_2 = points_2[status == 1]

    # Return Tracked Points
    return points_1, points_2


def featureDetection():
    # Detect FAST Features
    thresh = dict(threshold=25, nonmaxSuppression=True)
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast


def getK():
    return np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )


def monoVO(frame_0, frame_1, MIN_NUM_FEAT):
    # Input: Two image frames
    # Returns: Rotation matrix and translation vector, boolean validity, ignore pose if false
    image_1 = frame_0
    image_2 = frame_1

    detector = featureDetection()
    kp1 = detector.detect(image_1)
    points_1 = np.array([ele.pt for ele in kp1], dtype="float32")
    points_1, points_2 = featureTracking(image_1, image_2, points_1)

    K = getK()

    E, mask = cv2.findEssentialMat(points_2, points_1, fc, pp, cv2.RANSAC, 0.999, 1.0)
    _, R, t, mask = cv2.recoverPose(E, points_2, points_1, focal=fc, pp=pp)

    if len(points_2) < MIN_NUM_FEAT:
        return R, t, False

    return R, t, True
