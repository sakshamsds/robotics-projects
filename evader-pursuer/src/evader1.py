#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import random


def publisher():
    global publisher
    velocity_topic = "/cmd_vel"
    publisher = rospy.Publisher(name=velocity_topic, data_class=Twist, queue_size=1)


def evade(msg):
    # print(len(msg.ranges)) = 720
    # minimum angle =  -134.99974079443422 = ranges[719]
    # maximum angle =  134.99974079443422 = ranges[0]
    # increment angle =  0.37552083903996564
    # msg.ranges[359] # front
    min_range = min(msg.ranges[300:420])  # ~45 deg viewing angle
    threshold = 2

    # if the front object is within some threshold then randomly change direction ot -90 or +90
    if min_range < threshold:
        angular_velocity = random.choice([-1, 1]) * math.pi / 2
        publisher.publish(get_velocity(linear=0, angular=angular_velocity))
    else:
        # move forward at 1u/s
        publisher.publish(get_velocity(linear=2, angular=0))


def get_velocity(linear=0, angular=0):
    velocity = Twist()
    velocity.linear.x = linear
    velocity.angular.z = angular
    return velocity


def evader():
    # initialize the node
    rospy.init_node("evader", anonymous=True)

    # subscribe to /front/scan topic for evading
    laser_topic = "/front/scan"
    rospy.Subscriber(name=laser_topic, data_class=LaserScan, callback=evade)
    publisher()
    rospy.spin()


if __name__ == "__main__":
    try:
        evader()
    except rospy.ROSInterruptException:
        pass
