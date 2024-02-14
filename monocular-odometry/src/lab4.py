#!/usr/lib/bin/env/ python

# Import Monocular Visual Odometry
import mono_vo as mv

# You may only use the functions imported below
import numpy as np
import cv2
import rospy
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import time

bridge = CvBridge()

MIN_FEATURES = (
    750  # Minimum number of features for reliable relative pose between frames
    # 1500  # Minimum number of features for reliable relative pose between frames  # affects the performance
)

IMAGE_TOPIC = "kitti/image"
POSE_TOPIC = "kitti/pose"
MARKER_TOPIC = "visualization_marker_2"
KITTI_FRAME = "kitti_frame"
GROUND_TRUTH_POSES = []  # TwistStamped list
ESTIMATED_POSES = [(0, [0, 0, 0])]  # (sequence_id, 3*1 vectors)
IMAGES = []  # ROS Image
TRANSFORMATIONS = []  # (rotation, translation)


def plotTraj():
    rviz_publisher = rospy.Publisher(name=MARKER_TOPIC, data_class=Marker, queue_size=1)
    points = Marker()
    points.header.frame_id = KITTI_FRAME
    points.header.stamp = rospy.Time.now()
    points.type = Marker.POINTS
    points.action = Marker.ADD
    points.scale.x = 3
    points.scale.y = 4
    points.color.a = 1
    points.color.r = 0
    points.color.g = 0
    points.color.b = 1
    points.lifetime = rospy.Duration()

    # points trajector works fine, no need to use linestrip
    # print(ESTIMATED_POSES)

    for j in range(len(ESTIMATED_POSES)):
        point = Point()
        point.x = float(ESTIMATED_POSES[j][1][0])
        point.y = float(ESTIMATED_POSES[j][1][2])
        points.points.append(point)

    rviz_publisher.publish(points)
    return


def computeError():
    # implement RMSE here
    # error is due to lack of scale information in monocular VO
    # error is euclidean distance in 3D between two positions
    squared_sum = 0
    num_poses = len(ESTIMATED_POSES)
    for i in range(num_poses):
        # calculate error for correct sequence id
        sequence_id = ESTIMATED_POSES[i][0]
        x1 = ESTIMATED_POSES[i][1][0]
        z1 = ESTIMATED_POSES[i][1][2]

        truth = GROUND_TRUTH_POSES[sequence_id - 1]
        x2 = truth.twist.linear.x
        z2 = truth.twist.linear.z
        squared_sum += (x1 - x2) ** 2 + (z1 - z2) ** 2

    # 149.27721463374706
    print("Absolute Trajectory Error = ", math.sqrt(squared_sum / num_poses))
    return


def image_callback(image_msg):
    # sequence_id = image_msg.header.seq
    # print("image: ", sequence_id)

    latest_frame = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    if not IMAGES:  # empty list
        # one time call
        IMAGES.append(latest_frame)
    else:
        frame_0 = IMAGES[-1]
        frame_1 = latest_frame  # taking the latest image as most recent frame

        # pass two most recent images to getRT()
        # begin = time.time()
        R, t, valid = monoVO(frame_0, frame_1, MIN_NUM_FEAT=MIN_FEATURES)
        # R, t, valid = mv.monoVO(frame_0, frame_1, MIN_NUM_FEAT=MIN_FEATURES)
        # end = time.time()

        # NOTE: if the pose is invalid, then the function runtime is higher than the publishing rate
        # which causes our subscriber misses few images and we never get a valid pose again
        # print("Total runtime for ", valid, " pose is ", round(end - begin, 3))

        # print(valid) # fails at 250 for 1500 features
        # when valid is not true, callback is not able to keep up with publisher

        if valid:
            # if frame is valid, then transform relative pose change to get absolute pose in
            # the initial coordinate frame

            if not TRANSFORMATIONS:  # empty list
                TRANSFORMATIONS.append((R, t))
                ESTIMATED_POSES.append((image_msg.header.seq, t))
            else:
                # scale not used here
                rotation_matrix = TRANSFORMATIONS[-1][0]
                translation_vector = TRANSFORMATIONS[-1][1]

                translation_vector = translation_vector + rotation_matrix.dot(t)
                rotation_matrix = rotation_matrix.dot(R)

                TRANSFORMATIONS.append((rotation_matrix, translation_vector))
                ESTIMATED_POSES.append((image_msg.header.seq, translation_vector))

            # plot xz here
            plotTraj()

            # add only valid frame
            IMAGES.append(frame_1)

        # If getRT() returns invalid results, the most recent frame must be ignored and dropped.
        # no processing done
    return


def pose_callback(pose_msg):
    GROUND_TRUTH_POSES.append(pose_msg)
    # print("pose seq: ", pose_msg.header.seq)
    # part 3 is already plotting on rviz, no need to plot this one
    return


def subscriber():
    # initialize node
    rospy.init_node("monocular_VO", anonymous=True)

    # subscribe to image topic as well as the ground truth pose
    rospy.Subscriber(
        name=IMAGE_TOPIC, data_class=Image, callback=image_callback, queue_size=10
    )
    rospy.Subscriber(name=POSE_TOPIC, data_class=TwistStamped, callback=pose_callback)
    return


# mono_vo
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
    points_2, status, _ = cv2.calcOpticalFlowPyrLK(
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


def monoVO(frame_0, frame_1, MIN_NUM_FEAT):
    # Input: Two image frames
    # Returns: Rotation matrix and translation vector, boolean validity, ignore pose if false
    image_1 = frame_0
    image_2 = frame_1

    detector = featureDetection()
    kp1 = detector.detect(image_1)
    points_1 = np.array([ele.pt for ele in kp1], dtype="float32")
    points_1, points_2 = featureTracking(image_1, image_2, points_1)

    E, _ = cv2.findEssentialMat(points_2, points_1, fc, pp, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, points_2, points_1, focal=fc, pp=pp)

    if len(points_2) < MIN_NUM_FEAT:
        return R, t, False

    return R, t, True


if __name__ == "__main__":
    try:
        subscriber()
        while len(IMAGES) < 1000:
            print("trajectory not yet complete", len(IMAGES))
            time.sleep(10)
            # time taken ~ 1000/fps, 100s, 5 iterations
        print("TRAJECTORY COMPLETE")
        computeError()
    except rospy.ROSInterruptException:
        pass
