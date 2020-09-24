#!/usr/bin/env python
import rospy
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs import msg
from BoundingBoxes.msg import BoundingBoxes
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation as R

# Define subscriber
obj_no_sub = Subscriber("????", Int8) # not sure the topic name of objects number
bbox_sub = Subscriber("????", BoundingBoxes) # not sure the topic name of bbox
pcl_sub = Subscriber("/vel1/velodyne_points", PointCloud2)

syc = ApproximateTimeSynchronizer([obj_no_sub, bbox_sub, pcl_sub], queue_size=5, slop=0.1)
syc.registerCallback(??????) # fill in the callback function

# Intrinsic information of front-left camera
K = [[762.7249337622711, 0.0, 640.5], [0.0, 762.7249337622711, 360.5], [0.0, 0.0, 1.0]]
P = [[762.7249337622711, 0.0, 640.5, -53.39074536335898], [0.0, 762.7249337622711, 360.5, 0.0], [0.0, 0.0, 1.0, 0.0]]

## Extrinsic information
# Front-left camera to base
p_fl2b = [0.75, 0.1, 1.5]
orien_fl2b = [0.56099, -0.56099, 0.43046, -0.43046]
R_fl2b = R.from_quat(orien_fl2b)
T_fl2b = np.concatenate((R_fl2b.as_matrix(), np.reshape(p_fl2b, (3,1))), axis=1)
T_fl2b = np.concatenate((T_fl2b, [0, 0, 0, 1]), axis=0)
# Lidar to base
p_ld2b = [0.7, 0, 1.8]
orien_ld2b = [0, 0, 0, 1]
R_ld2b = R.from_quat(orien_ld2b)
T_ld2b = np.concatenate((R_ld2b.as_matrix(), np.reshape(p_ld2b, (3,1))), axis=1)
T_ld2b = np.concatenate((T_ld2b, [0, 0, 0, 1]), axis=0)

# Lidar to Front-left Camera
T_ld2fl = np.dot(T_fl2b, np.linalg.inv(T_ld2b))

# Define publisher
ldmk_id_pubulish = rospy.Publisher('/landmark/id',????, queue_size=1) #Not sure landmark id type
pose_publish = rospy.Publisher('/landmark/pose', PoseStamped, queue_size=1)

# In callback function add the following lines:
# ldmk_id_pubulish.publish(id)
# pose_publish.publish(pose)


class Landmarks:
    # This class stores landmarks information
    def __init__(self):
        self.landmarks = []     # list of np arrays

    # Inputs:
    # pt1, pt2 : np.array of size 3x1
    # Outputs: euclidean distance
    def l2_dist(self, pt1, pt2):
        return np.linalg.norm(np.vstack((pt1, pt2)))

    # Inputs:
    # point: np.array of 3x1
    # Outputs:
    # landmark id: id if present otherwise -1
    def search_landmark(self, point):
        if len(self.landmarks) == 0:
            return -1

        min_dist = 0.254   # threshold of 10 inches = 0.254 m
        min_id = -1
        for i in range(len(self.landmarks)):
            dist = self.l2_dist(self.landmarks[i], point)
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id

    # Inputs:
    # point: np.array of 3x1
    def add_landmark(self, point):
        self.landmarks.append(point)

    # Outputs:
    # returns last landmark id, if it is first landmark then returns -1
    def get_last_id(self):
        return len(self.landmarks)-1

    def get_pose(self, ID):
        return self.landmarks[ID]


# Convert pointcloud to image plane
def from_lidar_to_image():
    pass


def search_3d_object_points():
    pass


# Input:
# 1. LtoV: transformation matrix between lidar frame to vehicle frame
# 2. VtoM: transformation matrix between vehicle frame to map frame
# 3. lidar points: np array in homogenous coordinates 4xN
# Outputs:
# 3d points: np array in map coordinate frame 4XN
def from_lidar_to_map(LtoV, VtoM, lidar_points):
    return np.array(VtoM).dot(np.array(LtoV).dot(lidar_points))


# Inputs:
# 1. lidar points: np array in homogenous coordinates 4xN
# 2. Landmarks: instantiated landmark class which stres landmarks
# Outputs:
# img_landmarks: np array Nx2 first colm: IDs second: pose
def get_landmarks(lidar_points, Landmarks):
    # pt : np.array = [x,y,z]
    # img_landmarks : np.array of np.arrays = [[id0,pt0],[id1,pt1],[id2,pt2]]
    img_landmarks = []
    for i in range(lidar_points.shape[1]):
        pt = lidar_points[:3, i]            # np array
        ID = Landmarks.search_landmark(pt)
        if ID == -1:
            Landmarks.add_landmark(pt)
            last_id = Landmarks.get_last_id()
            img_landmarks.append(np.array([last_id, pt]))
        else:
            img_landmarks.append(np.array([ID, Landmarks.get_pose(ID)]))

    return np.array(img_landmarks)


# Subscriber callbacks

if __name__ == '__main__':
    lidar_to_camera_tf_topic = ""
    vehicle_to_lidar_tf_topic = ""
    vechicle_pose_topic = ""
    camera_intrinsics = None
    object_detector_topic = "object_detector"
    bounding_box_topic = "bounding_boxes"
    lidar_point_cloud_topic = "/scan"

