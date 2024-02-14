#!/usr/bin/python

import rospy
import tf
import math
from geometry_msgs.msg import Twist
import time


if __name__ == "__main__":
    rospy.init_node("pursuer", anonymous=True)

    # publisher
    pursuer_velocity_topic = "/tb3_1/cmd_vel"
    publisher = rospy.Publisher(pursuer_velocity_topic, data_class=Twist, queue_size=1)

    listener = tf.TransformListener()
    # start pursuer after 4 seconds, so that we have the required transforms for t-2 seconds
    time.sleep(4)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            now = rospy.Time.now()
            past = now - rospy.Duration(2)

            listener.waitForTransformFull("tb3_1", now, "tb3_0", past, "world", rospy.Duration(1))
            (trans, rot) = listener.lookupTransformFull("tb3_1", now, "tb3_0", past, "world")

            # get latest transform
            # (trans, rot) = listener.lookupTransform(
            #     "tb3_1", "tb3_0", rospy.Time().now()
            # )
        except (
            tf.LookupException,
            tf.ExtrapolationException,
            tf.ConnectivityException,
        ):
            continue

        # print(trans)
        angular = math.atan2(trans[1], trans[0])
        linear = min(0.5, math.sqrt(trans[0] ** 2 + trans[1] ** 2))
        velocity = Twist()
        velocity.linear.x = linear
        velocity.angular.z = angular
        publisher.publish(velocity)

        rate.sleep()
