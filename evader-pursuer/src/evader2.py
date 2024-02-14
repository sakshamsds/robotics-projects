#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import random


def publisher():
    global publisher
    velocity_topic = "/tb3_0/cmd_vel"
    publisher = rospy.Publisher(name=velocity_topic, data_class=Twist, queue_size=1)


def evade(msg):
    # min angle = 0
    # max angle = 6.28 = 2pi
    # print(len(msg.ranges)) = 360
    # ranges[0] = front, as idx increases, angle increases in anti clockwise direction
    # max scan range = 3.5
    min_range = min(msg.ranges[340:] + msg.ranges[:20])  # ~40 deg viewing angle
    threshold = 2

    # if the front object is within some threshold then randomly change direction -90 or +90
    if min_range < threshold:
        angular_velocity = random.choice([-1, 1]) * math.pi / 2
        publisher.publish(get_velocity(linear=0, angular=angular_velocity))
    else:
        # move forward at 1u/s
        # publisher.publish(get_velocity(linear=0, angular=0))
        publisher.publish(get_velocity(linear=1, angular=0))


def get_velocity(linear=0, angular=0):
    velocity = Twist()
    velocity.linear.x = linear
    velocity.angular.z = angular
    return velocity


def evader():
    # initialize the node
    rospy.init_node("evader2", anonymous=True)

    # subscribe to /front/scan topic for evading
    laser_topic = "/tb3_0/scan"
    rospy.Subscriber(name=laser_topic, data_class=LaserScan, callback=evade)

    # initialize publisher
    publisher()
    rospy.spin()


if __name__ == "__main__":
    try:
        evader()
    except rospy.ROSInterruptException:
        pass
