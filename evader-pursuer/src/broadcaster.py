#!/usr/bin/env python

import roslib

roslib.load_manifest("learning_tf")

from nav_msgs.msg import Odometry
import rospy
import tf

# publishes the coordinate frame of the robot with regards to the global frame
def broadcast_transform(msg, bot_name):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    br = tf.TransformBroadcaster()
    # pass position and orientation values as a tuple, z will be 0
    br.sendTransform(
        (position.x, position.y, position.z),
        (orientation.x, orientation.y, orientation.z, orientation.w),
        rospy.Time.now(),
        bot_name,
        "world",
    )


if __name__ == "__main__":
    rospy.init_node("broadcaster", anonymous=True)
    bot_name = rospy.get_param("~bot")
    print(bot_name)

    # The node subscribes to topic "turtleX/pose" and runs function handle_turtle_pose on every incoming message.
    rospy.Subscriber(
        "/{}/odom".format(bot_name), Odometry, broadcast_transform, bot_name
    )
    rospy.spin()
