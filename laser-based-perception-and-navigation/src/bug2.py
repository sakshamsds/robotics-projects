#!/usr/bin/env python

import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time


# fixed variables
goal_distance_tolerance = 0.1
line_distance_threshold = 0.1
obstacle_distance_threshold = 1
lv = 0.25
left_av = math.pi / 4


def publisher():
    global publisher
    velocity_topic = "/cmd_vel"
    publisher = rospy.Publisher(name=velocity_topic, data_class=Twist, queue_size=1)


def get_velocity(linear=0, angular=0):
    velocity = Twist()
    velocity.linear.x = linear
    velocity.angular.z = angular
    return velocity


def get_obstacles_orientation(msg):
    # print(len(msg.ranges)) = 720
    # minimum angle =  -134.99974079443422 = ranges[719]
    # maximum angle =  134.99974079443422 = ranges[0]
    # increment angle =  0.37552083903996564
    # msg.ranges[359] # front

    global obstacle_in_front, obstacle_on_right
    obstacle_on_right = True if min(msg.ranges[120:240]) < obstacle_distance_threshold else False
    obstacle_in_front = True if min(msg.ranges[240:480]) < obstacle_distance_threshold else False
    return


def get_current_position(msg):
    global current_orientation
    global current_position
    current_position = msg.pose.pose.position
    current_orientation = msg.pose.pose.orientation
    return


def goal_seek(x_final, y_final):
    if (
        get_distance_from_goal(current_position.x, current_position.y, x_final, y_final)
        <= goal_distance_tolerance
    ):
        # we have reached the goal, stop the robot
        publisher.publish(get_velocity(linear=0, angular=0))
        print(
            "distance from goal",
            get_distance_from_goal(
                current_position.x, current_position.y, x_final, y_final
            ),
        )
        return "reached_goal"
    else:
        # move towards the goal unless obstacle seen
        av = get_angle_to_goal(current_position.x, current_position.y, x_final, y_final)
        publisher.publish(get_velocity(linear=lv, angular=2 * av))

    return "wall_follow" if obstacle_in_front else "goal_seek"


def wall_follow(x_final, y_final):
    if obstacle_in_front:
        # turn left
        publisher.publish(get_velocity(linear=0, angular=left_av))
    else:
        if obstacle_on_right:
            # move straight
            publisher.publish(get_velocity(linear=lv, angular=0))
        else:
            # move towards obstacle, on right
            publisher.publish(get_velocity(linear=lv, angular=-left_av))

    # if back on start to goal line, change state to goal seek
    x_current = current_position.x
    y_current = current_position.y
    distance = get_distance_from_line(0, 0, x_final, y_final, x_current, y_current)
    # print("distance from line : ", distance)

    return "goal_seek" if distance < line_distance_threshold else "wall_follow"


def get_distance_from_line(x1, y1, x2, y2, x3, y3):
    return abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)) / np.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    )


def get_distance_from_goal(x_current, y_current, x_final, y_final):
    return math.sqrt((x_final - x_current) ** 2 + (y_final - y_current) ** 2)


def get_angle_to_goal(x_current, y_current, x_final, y_final):
    return math.atan2(y_final - y_current, x_final - x_current) - 2 * np.arcsin(
        current_orientation.z
    )


# fixes the issue of getting out of origin
# fixes my third and fourth test case
# when goal is behind bot
def face_towards_the_goal():
    for i in range(10):
        av = get_angle_to_goal(current_position.x, current_position.y, x_final, y_final)
        publisher.publish(get_velocity(linear=0, angular=av))
        time.sleep(0.3)
    return


def driver(x_final, y_final):
    # subscribe to laser scan
    rospy.Subscriber("/front/scan", LaserScan, get_obstacles_orientation)
    # this topic gives position wrt world frame
    rospy.Subscriber("/odometry/filtered", Odometry, get_current_position)
    publisher()

    global current_state
    current_state = "goal_seek"

    # let the current position be initialized
    time.sleep(1)

    # 3 secs runtime
    face_towards_the_goal()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print(current_state)

        if current_state == "goal_seek":
            # goal seek here
            current_state = goal_seek(x_final, y_final)
        elif current_state == "wall_follow":
            # wall follow here
            current_state = wall_follow(x_final, y_final)
        else:
            # goal reached
            print("----------GOAL REACHED-------------")
            break

        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("bug2", anonymous=True)
    x_final = rospy.get_param("~x")
    y_final = rospy.get_param("~y")
    print(x_final)  # <class 'float'>
    print(y_final)
    try:
        driver(x_final, y_final)
    except rospy.ROSInterruptException:
        pass
