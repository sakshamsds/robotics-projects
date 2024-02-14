# Before you begin

## To enable the LiDAR, run the following. 

export HUSKY_UST10_ENABLED='1'

export TURTLEBOT3_MODEL=waffle


## To verify, run roslaunch and echo the sensor topic

### For Evader

roslaunch husky_gazebo husky_playpen.launch

rostopic echo /front/scan

### For Pursuer

roslaunch turtlebot3_gazebo dual_turtlebot3.launch

rostopic echo /tb3_0/odom
