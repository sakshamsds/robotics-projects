#!/usr/bin/env python

import numpy as np
import rospy
import math
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import random


def init_publisher():
    global publisher
    marker_topic = "visualization_marker"
    publisher = rospy.Publisher(name=marker_topic, data_class=Marker, queue_size=1)


def callback(msg):
    # print(len(msg.ranges)) = 720
    # minimum angle =  -134.99974079443422 = ranges[719]
    # maximum angle =  134.99974079443422 = ranges[0]
    # increment angle =  0.37552083903996564
    # msg.ranges[359] # front
    points = get_coordinates(msg.ranges)
    ransac(points)
    return


def get_coordinates(ranges):
    points = []
    increment_angle = -270 / 720
    start_angle = 135

    for i in range(len(ranges)):
        if ranges[i] > 0.15 and ranges[i] < 30.0:
            angle = math.radians(start_angle + increment_angle * i)
            x = ranges[i] * math.cos(angle)
            y = -ranges[i] * math.sin(angle)
            points.append((x, y))

    return points


def ransac(points):

    iterations = 20  # num iterations to find single line
    distance_threshold = 0.01
    min_points = 100

    while len(points) > min_points:

        inliers_count = []
        inliers_list = []

        num_points = len(points)
        # print(num_points)
        # draw single line
        for i in range(iterations):

            # select two points randomly
            p1_idx = random.randint(0, num_points - 1)
            p2_idx = random.randint(0, num_points - 1)

            (x1, y1) = (points[p1_idx][0], points[p1_idx][1])
            (x2, y2) = (points[p2_idx][0], points[p2_idx][1])

            num_inliers = 0
            temp_inliers = []

            # calculate distance of each point from the line
            for j in range(num_points):
                current_x = points[j][0]
                current_y = points[j][1]
                distance = get_distance_from_line(x1, y1, x2, y2, current_x, current_y)

                # using a threshold t, count number of inliers
                if distance < distance_threshold:
                    num_inliers += 1
                    temp_inliers.append((current_x, current_y))

            inliers_count.append(num_inliers)
            inliers_list.append(temp_inliers)

        best_idx = np.argmax(inliers_count)
        inliers = inliers_list[best_idx]
        visualize_line(inliers)
        points = [point for point in points if point not in inliers]  # remove inliers

    return


def get_distance_from_line(x1, y1, x2, y2, x3, y3):
    return abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)) / np.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    )


# assert get_distance_from_line(0, 0, 0, 1, 5, 5) == 5


def visualize_line(inliers):

    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.a = 0.5
    marker.color.r = 0
    marker.color.g = 1
    marker.color.b = 0
    marker.lifetime = rospy.Duration()

    for (x, y) in inliers:
        point = Point()
        point.x = x
        point.y = y
        marker.points.append(point)

    publisher.publish(marker)
    return


if __name__ == "__main__":
    rospy.init_node("perception", anonymous=True)
    try:
        init_publisher()
        rospy.Subscriber("/front/scan", LaserScan, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
