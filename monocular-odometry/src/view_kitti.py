#!/usr/lib/bin/env/ python

# reads the raw data
# publishes them in appropriate topics

import rospy
import os
import cv2
import glob
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


IMAGES_PATH = os.path.dirname(__file__) + "/../dataset/sequence/image_0/"
POSE_PATH = os.path.dirname(__file__) + "/../dataset/pose/00.txt"
IMAGE_TOPIC = "kitti/image"
POSE_TOPIC = "kitti/pose"
MARKER_TOPIC = "visualization_marker"
KITTI_FRAME = "kitti_frame"
BRIDGE = CvBridge()


def publish_kitti(fps):
    # publish the images present
    # inside sequence/00/image 0 to the topic kitti/image

    image_publisher = rospy.Publisher(name=IMAGE_TOPIC, data_class=Image, queue_size=10)

    # corresponding ground truth
    # poses from pose folder to topic kitti/pose
    # geometry msgs/Twist

    pose_publisher = rospy.Publisher(
        name=POSE_TOPIC, data_class=TwistStamped, queue_size=10
    )

    # print(sorted(glob.glob(IMAGES_PATH + "*.png")))

    images = [
        cv2.imread(file_name) for file_name in sorted(glob.glob(IMAGES_PATH + "*.png"))
    ]
    poses = get_poses()

    rate = rospy.Rate(fps)

    for i in range(len(images)):
        # header for checking the sequence
        header = Header()
        header.stamp = rospy.Time.now()
        header.seq = i + 1
        header.frame_id = KITTI_FRAME

        # 1. publish image
        image_message = BRIDGE.cv2_to_imgmsg(images[i], encoding="bgr8")
        image_message.header = header
        image_publisher.publish(image_message)
        
        # print("published image: ", header.seq)

        # 2. publish pose
        twist_stamped_msg = poses[i]
        twist_stamped_msg.header = header
        pose_publisher.publish(twist_stamped_msg)

        # print("published_pose: ", header.seq)

        # 3. plot on rviz
        plot_on_rviz(poses, i)

        rate.sleep()

    return


def get_poses():
    poses = []
    with open(POSE_PATH) as f:
        lines = f.readlines()
        for line in lines:
            # break into twelve columns and get 3 and 11 value
            values = line.split()
            x = float(values[3])
            z = float(values[11])
            twist_stamped_msg = TwistStamped()
            twist_stamped_msg.twist.linear.x = x
            twist_stamped_msg.twist.linear.z = z
            poses.append(twist_stamped_msg)
    return poses


def plot_on_rviz(poses, i):
    rviz_publisher = rospy.Publisher(name=MARKER_TOPIC, data_class=Marker, queue_size=10)
    points = Marker()
    points.header.frame_id = KITTI_FRAME
    points.header.stamp = rospy.Time.now()
    points.type = Marker.POINTS
    points.action = Marker.ADD
    points.scale.x = 3
    points.scale.y = 4
    points.color.a = 1
    points.color.r = 1
    points.color.g = 0
    points.color.b = 0
    points.lifetime = rospy.Duration()

    for j in range(i):
        point = Point()
        point.x = poses[j].twist.linear.x
        point.y = poses[j].twist.linear.z
        points.points.append(point)

    rviz_publisher.publish(points)
    return


if __name__ == "__main__":
    try:
        rospy.init_node("kitti_publisher", anonymous=True)
        fps = rospy.get_param("~fps")
        publish_kitti(fps)
    except rospy.ROSInterruptException:
        pass
