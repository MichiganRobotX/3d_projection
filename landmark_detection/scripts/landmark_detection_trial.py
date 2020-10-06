#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
from std_msgs.msg import Header

class C:
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        self.xyz = None
        self.cloud_sub = rospy.Subscriber("/wamv/sensors/lidars/lidar_wamv/points", PointCloud2, self.callback)
        # Tf from flc to lidar
        self.flc_to_lidar = np.array([-0.07, -0.1, 0.13850])
        # Intrinsic Matrix
        self.flc_K =  np.array([[762.7249337622711, 0.0, 640.5],[0.0, 762.7249337622711, 360.5],[0.0, 0.0, 1.0]])



    # Convert pointcloud to image plane
    def callback(self, ros_point_cloud):
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        #self.lock.acquire()
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            if (x[0] <= 0.0):
                continue
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b v        self.from_lidar_to_image()alues in the 0-255 range
                        # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)

        self.xyz = xyz
        xyz_flc_cam = xyz + self.flc_to_lidar
        uv_flc_cam = np.dot(self.flc_K,xyz_flc_cam.T) # 3 x lamba
        uv_flc_cam = uv_flc_cam / uv_flc_cam[2,:]
        print(uv_flc_cam[:,20])
    
    def from_lidar_to_image(self):
        # Inspect pcl data from simulator
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
    # lidar_to_camera_tf_topic = ""
    # vehicle_to_lidar_tf_topic = ""
    # vechicle_pose_topic = ""
    # camera_intrinsics = None
    # object_detector_topic = "object_detector"
    # bounding_box_topic = "bounding_boxes"
    c = C()
    lidar_point_cloud_topic = "/wamv/sensors/lidars/lidar_wamv/points"
    
    rospy.spin()
    c.from_lidar_to_image()
    #from_lidar_to_image()

