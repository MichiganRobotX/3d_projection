# 3d_projection
This repo contains ros package for publishing 3D landmark pose, label and ID for robotX boat.

## Input topics:
```
bbox_topic = "/darknet_ros/bounding_boxes"
pcloud_topic = "/wamv/sensors/lidars/lidar_wamv/points"
localization_topic = "/wamv/robot_localization/odometry/filtered"
```
bbox_topic contains information of object bounding boxes and their labels.<br/>
pcloud_topic is for LiDAR point cloud.<br/>
localization_topic is given by SLAM team, it containes boat poses in map frame.

## Output topics:
```
landmarks_topic = "/landmark_detection/landmarks"
landmark_lla_topic = "/vrx/perception/landmark"
```
landmarks_topic publishes the landmark labels and landmark positions in base frame.<br/>
landmark_lla_topic publishes the landmark labels and landmark positions in LLA coordinate (which is required by the RobotX tasks).<br/>

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

## To Run the package with darknet_ros together:
In catkin_ws/src
```
roslaunch landmark_detection perception.launch
```
If shows the error ‘can not launch....’, run the following command lines first.
```
Chmod +x /path/to/landmark_detector.py
Chmod +x /path/to/perception.launch
```
