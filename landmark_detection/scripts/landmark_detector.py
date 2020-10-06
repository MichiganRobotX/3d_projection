#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from darknet_ros_msgs.msg import BoundingBoxes
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
from std_msgs.msg import Header
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy import spatial
from landmark_detection.msg import LandmarkPoseWithId

class LandmarkDetector:
    # Setup topic names
    bbox_topic = "/darknet_ros/bounding_boxes"
    pcloud_topic = "/wamv/sensors/lidars/lidar_wamv/points"
    camera_info_topic = "/wamv/sensors/cameras/front_left_camera/camera_info"
    
    bboxes = None
    pcloud_pc2 = None

    pcloud_xyz = None
    camera_uv = None

    centers_xyz = None
    centers_uv = None

    nearest_uv = None
    nearest_idx_uv = None
    

    flc_to_lidar = np.array([-0.07, -0.1, 0.13850])
    lidar_to_base = np.array([0,0,0])
    flc_K = np.array([[762.7249337622711, 0.0, 640.5],[0.0, 762.7249337622711, 360.5],[0.0, 0.0, 1.0]])

    def __init__(self):
        # Initialize node
        rospy.init_node('landmark_detector', anonymous=True)
        rospy.loginfo("============= landmark_detector start =============")
        bbox_sub = Subscriber(self.bbox_topic, BoundingBoxes)
        pcl_sub = Subscriber(self.pcloud_topic, PointCloud2)
        syc = ApproximateTimeSynchronizer([bbox_sub, pcl_sub], queue_size=5, slop=0.1)
        syc.registerCallback(self.update_bbox_and_pcloud)

    def update_bbox_and_pcloud(self, bboxes, pcloud):
        # def update_bbox(self, bboxes):
        self.bboxes = bboxes
        self.get_center()
        print(self.centers_xyz)

        # def update_pcloud(self, pcloud):
        self.pcloud_pc2 = pcloud
        self.pc2_to_xyz()

        self.publish_landmark_info(landmark_id, landmark_pose, pcloud.header)

    def pc2_to_xyz(self):
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        #self.lock.acquire()
        gen = pc2.read_points(self.pcloud_pc2, skip_nans=True)
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

        self.pcloud_xyz = xyz

    def lidar_to_img_plane(self):
        xyz_flc_cam = self.pcloud_xyz + self.flc_to_lidar #Broadcasting (N x 3)
        uv_flc_cam = np.dot(self.flc_K, xyz_flc_cam.T) #3xN
        self.camera_uv = uv_flc_cam / uv_flc_cam[2, :] #3xN

    def center_from_bbox(self):
        num_bboxes = len(self.bboxes.bounding_boxes)
        self.centers_uv = np.zeros((2, num_bboxes)) # 2xN
        for i in range(num_bboxes):
            x_c = (self.bboxes.bounding_boxes[i].xmin + self.bboxes.bounding_boxes[i].xmax) / 2
            y_c = (self.bboxes.bounding_boxes[i].ymin + self.bboxes.bounding_boxes[i].ymax) / 2
            self.centers_uv[:,i] = np.array([x_c, y_c])
    
    def nearest_neighbor(self):
        num_bboxes = len(self.bboxes.bounding_boxes)
        self.nearest_idx_uv = np.zeros(num_bboxes, dtype=np.int32)
        tree = spatial.KDTree(self.camera_uv[:2, :])
        self.nearest_idx_uv = tree.query(self.centers_uv)[1]
        # for i in range(num_bboxes):
        #     center = [self.centers_uv[1,i], self.centers_uv[0,i], 1]
        #     self.nearest_idx_uv[i] = tree.query(center)[1]
        self.centers_xyz = self.pcloud_xyz[self.nearest_idx_uv]

    def lidar_to_base(self):
        pass

    def get_center(self):
        self.lidar_to_img_plane()
        self.center_from_bbox()
        self.nearest_neighbor()
    
    def publish_landmark_info(self, landmark_id, landmark_pose, header):
        landmark_info_msg = LandmarkPoseWithId()
        landmark_info_msg.header = header
        landmark_info_msg.id = landmark_id
        landmark_info_msg.pose = landmark_pose
        self.landmark_pub.publish(landmark_info_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    L = LandmarkDetector()
    L.run()
    