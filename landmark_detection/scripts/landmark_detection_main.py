#!/usr/bin/env python
import numpy as np

# Convert pointcloud to image plane
def from_lidar_to_image():
    pass

def search_3d_object_points():
    pass


# Requires:
# 1. LtoV: transformation matrix between lidar frame to vehicle frame
# 2. VtoM: transformation matrix between vehicle frame to map frame
# 3. lidar points: np array in homogenous coordinates 4xN
# Returns:
# 3d points: np array in map coordinate frame 4XN
def from_lidar_to_map(LtoV, VtoM, lidar_points):
    # pass
    return np.array(LtoV).dot(np.array(VtoM).dot(lidar_points))

# Requires:
# 1. lidar points: np array in homogenous coordinates 4xN
# 2. 
# 3. 
# Returns:
# 3d points: np array in map coordinate frame 4XN
def get_landmarks(lidar_points, ):
    pass

# Subscriber callbacks

if __name__ == '__main__':
    lidar_to_camera_tf_topic = ""
    vehicle_to_lidar_tf_topic = ""
    vechicle_pose_topic = ""
    camera_intrinsics = None
    object_detector_topic = "object_detector"
    bounding_box_topic = "bounding_boxes"
    lidar_point_cloud_topic = "/scan"


