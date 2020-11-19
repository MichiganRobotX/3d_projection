# 3d_projection
This repo contains ros package for publishing 3D landmark pose, label and ID for robotX boat.

## Input topics:
/darknet_ros/bounding_boxes<br/> 
/wamv/sensors/lidars/lidar_wamv/points<br/> 

## Output topics:
/boat_frame_landmark_pose_with_id

## To Run the package
In catkin_ws/src
```
git clone https://github.com/MichiganRobotX/3d_projection.git
cd 3d_projection/landmark_detection
mv darknet_ros_msgs ../..
cd ../../..
catkin_make
source devel/setup.bash
rosrun landmark_detection landmark_detector.py
```
