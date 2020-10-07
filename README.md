# 3d_projection
This repo contains package for publishing 3D landmark pose and ID for robotX boat.

## Input topics:
/darknet_ros/bounding_boxes
/wamv/sensors/lidars/lidar_wamv/points
/wamv/sensors/cameras/front_left_camera/camera_info

## Output topics:
LandmarkPoseWithId

## To Run the package
In catkin_ws/src
```
git clone https://github.com/LuoXin0826/3d_projection.git
cd 3d_projection/landmark_detection
mv darknet_ros_msgs ../..
cd ../../..
catkin_make
source devel/setup.bash
rosrun 3d_projection landmark_detector.py
```
