#!/usr/lib/bin/env/ python

# References
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://stackoverflow.com/questions/6773584/how-are-glob-globs-return-values-ordered

import cv2
import numpy as np
import glob
import rospy
import os

IMAGES_PATH = os.path.dirname(__file__) + "/../images/"
# print(images_path)

rospy.init_node("lab4_cal", anonymous=True)
# print(sorted(glob.glob(images_path + "*.jpeg"), key=os.path.getmtime))

images = [
    cv2.resize(cv2.imread(file_name), (0, 0), fx=0.5, fy=0.5)
    for file_name in glob.glob(IMAGES_PATH + "*.jpeg")
]

input_image_shape = images[0].shape  # (1024, 768, 3)

row_corners = 6
col_corners = 9

# object points
object_points = np.zeros((row_corners * col_corners, 3), np.float32)
object_points[:, 0:2] = np.mgrid[0:col_corners, 0:row_corners].T.reshape(-1, 2)

# print(np.mgrid[0:col_corners, 0:row_corners].T.shape) = (6, 9, 2)
# print(np.mgrid[0:col_corners, 0:row_corners].shape) = (2, 9, 6)
# print(np.mgrid[0:col_corners, 0:row_corners].T.reshape(-1, 2).shape) = (54, 2)

object_points_list = []  # 3D points
image_points_list = []  # 2D points

for image in images:
    # convert image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find the position of the internal corners of the chessboard
    retval, corners = cv2.findChessboardCorners(gray_img, (col_corners, row_corners))

    # If corners are found, add 3D and 2D points
    if retval == True:
        object_points_list.append(object_points)
        image_points_list.append(corners)


_, matrix, _, _, _ = cv2.calibrateCamera(
    object_points_list,
    image_points_list,
    (input_image_shape[0], input_image_shape[1]),
    None,
    None,
)

print("----------------------------------------------------------")
print("Intrinsic Parameters : \n", matrix)
print("----------------------------------------------------------")
# print(matrix.shape) = (3, 3)
