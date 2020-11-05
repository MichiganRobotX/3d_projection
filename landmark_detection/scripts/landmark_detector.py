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
from landmark_detection.msg import Landmarkmsg, Landmarksmsg
from Landmarks import Landmarks
from gazebo_msgs.msg import LinkStates
from scipy.spatial.transform import Rotation as R

class LandmarkDetector:
    # Setup topic names
    bbox_topic = "/darknet_ros/bounding_boxes"
    pcloud_topic = "/wamv/sensors/lidars/lidar_wamv/points"
    #camera_info_topic = "/wamv/sensors/cameras/front_left_camera/camera_info"
    #pose_topic = "/gazebo/link_states"

    bboxes = None
    pcloud_pc2 = None

    pcloud_xyz = None
    camera_uv = None

    centers_xyz = None
    centers_uv = None

    nearest_uv = None
    nearest_idx_uv = None

    # flc_to_lidar = np.array([-0.07, -0.1, 0.13850])
    base_to_lidar = np.array([0.7, 0, 1.8])
    base_to_flc_arm = np.array([0.73, 0.1, 2.26725])
    flc_arm_to_flc = np.array([ [0.9659258,   0.0000000,  0.2588190, 0.02 ],
                                [0.0000000,   1.0000000,  0.0000000, 0.0  ],
                                [-0.2588190,  0.0000000,  0.9659258, 0.027],
                                [0.0,         0.0,        0.0,       1.0  ] ])
    l_to_a = (base_to_flc_arm - base_to_lidar).reshape(-1,1)
    lidar_to_flc_arm = np.vstack((np.hstack((np.eye(3),l_to_a)), np.array([0, 0, 0, 1]).reshape(1,-1)))
    lidar_flc_fram_correction = np.array([[0,-1,0,0],
                                          [0,0,1,0],
                                          [-1,0,0,0],
                                          [0,0,0,1]])
    # lidar_to_flc = lidar_flc_fram_correction @ flc_arm_to_flc @ lidar_to_flc_arm
    lidar_to_flc = np.matmul(np.matmul(lidar_flc_fram_correction, flc_arm_to_flc) , lidar_to_flc_arm)
    flc_K = np.array([[762.7249337622711, 0.0, 640.5], [0.0, 762.7249337622711, 360.5], [0.0, 0.0, 1.0]])

    landmarks_list = Landmarks()
    landmark_id = None
    landmark_label = None

    def __init__(self):
        # Initialize node
        rospy.init_node('landmark_detector', anonymous=True)
        rospy.loginfo("============= landmark_detector start =============")
        bbox_sub = rospy.Subscriber(self.bbox_topic, BoundingBoxes, self.update_bbox)
        pcl_sub = rospy.Subscriber(self.pcloud_topic, PointCloud2, self.update_pcloud)
        # pose_sub = rospy.Subscriber(self.pose_topic, LinkStates, self.publish_landmark_info)
        # bbox_sub = Subscriber(self.bbox_topic, BoundingBoxes)
        # pcl_sub = Subscriber(self.pcloud_topic, PointCloud2)
        # rospy.loginfo("============= synchronizer start =============")
        # syc = ApproximateTimeSynchronizer([bbox_sub, pcl_sub], queue_size=10, slop=0.35)
        # rospy.loginfo("============= register callback start =============")
        # syc.registerCallback(self.update_bbox_and_pcloud)
        # rospy.loginfo("============= register callback end =============")
        self.landmark_pub = rospy.Publisher("/boat_frame_landmark_pose_with_id", Landmarksmsg, queue_size=1)

    # def update_bbox_and_pcloud(self, bboxes, pcloud):
    def update_bbox(self, bboxes):
        self.bboxes = bboxes
        # self.get_center()
        # print(self.centers_xyz)

    def update_pcloud(self, pcloud):
        self.pcloud_pc2 = pcloud
        self.pc2_to_xyz()
        current_bboxes = self.bboxes
        self.get_center(current_bboxes)

        self.publish_landmark_info(self.landmark_id, self.landmark_label, self.centers_xyz, pcloud.header)

    def pc2_to_xyz(self):
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        # self.lock.acquire()
        gen = pc2.read_points(self.pcloud_pc2, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            if (x[0] <= 0.0):
                continue
            test = x[3]
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f', test)
            i = struct.unpack('>l', s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)
            # prints r,g,b v self.from_lidar_to_image()alues in the 0-255 range
            # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)
            rgb = np.append(rgb, [[r, g, b]], axis=0)

        self.pcloud_xyz = xyz   # Nx3

    def lidar_to_img_plane(self):
        # xyz_flc_cam = self.lidar_to_flc @ np.hstack((self.pcloud_xyz, np.ones(len(self.pcloud_xyz)).reshape(-1,1))).T  # homogenize and matmul (4xN)
        xyz_flc_cam = np.matmul(self.lidar_to_flc, np.hstack((self.pcloud_xyz, np.ones(len(self.pcloud_xyz)).reshape(-1,1))).T) # homogenize and matmul (4xN)
        uv_flc_cam = np.dot(self.flc_K, xyz_flc_cam[:3, :])  # 3xN
        self.camera_uv = uv_flc_cam / uv_flc_cam[2, :]  # 3xN

    def center_from_bbox(self, bboxes):
        num_bboxes = len(bboxes.bounding_boxes)
        # num_bboxes = len(bboxes.bounding_boxes)
        self.centers_uv = np.zeros((2, num_bboxes))  # 2xN
        labels = []
        for i in range(num_bboxes):
            # print(f"xmin: {self.bboxes.bounding_boxes[i].xmin}, xmax: {self.bboxes.bounding_boxes[i].xmax}, ymin: {self.bboxes.bounding_boxes[i].ymin}, ymax: {self.bboxes.bounding_boxes[i].ymax}")
            # x_c = (self.bboxes.bounding_boxes[i].xmin + self.bboxes.bounding_boxes[i].xmax) / 2
            # y_c = (self.bboxes.bounding_boxes[i].ymin + self.bboxes.bounding_boxes[i].ymax) / 2
            x_c = (bboxes.bounding_boxes[i].xmin + bboxes.bounding_boxes[i].xmax) / 2
            y_c = (bboxes.bounding_boxes[i].ymin + bboxes.bounding_boxes[i].ymax) / 2
            label = bboxes.bounding_boxes[i].Class
            labels.append(label)
            self.centers_uv[:, i] = np.array([x_c, y_c])
        self.landmark_label = labels

    def nearest_neighbor(self, bboxes):
        # num_bboxes = len(self.bboxes.bounding_boxes)
        num_bboxes = len(bboxes.bounding_boxes)
        self.nearest_idx_uv = np.zeros(num_bboxes, dtype=np.int32)
        # print(self.camera_uv[:2, :])
        # print("================= End =============")
        tree = spatial.KDTree(self.camera_uv[:2, :].T)
        # print(self.centers_uv)
        # print("================= End =============")
        # out = tree.query(self.centers_uv.T)
        # print(out)
        self.nearest_idx_uv = tree.query(self.centers_uv.T)[1]
        # print(self.nearest_idx_uv)
        # for i in range(num_bboxes):
        #     center = [self.centers_uv[1,i], self.centers_uv[0,i], 1]
        #     self.nearest_idx_uv[i] = tree.query(center)[1]
        # print(self.camera_uv[:2, :][:, self.nearest_idx_uv])
        # print("================= End =============")
        self.centers_xyz = self.pcloud_xyz[self.nearest_idx_uv]

    def lidar_to_base(self, landmark_pose_lidar_frame):
        p_ld2b = [0.7, 0, 1.8]
        # orien_ld2b = [0, 0, 0, 1]
        # R_ld2b = R.from_quat(orien_ld2b)
        # T_ld2b = np.concatenate((R_ld2b.as_dcm(), np.reshape(p_ld2b, (3,1))), axis=1)
        # T_ld2b = np.concatenate((T_ld2b, np.array([[0, 0, 0, 1]])), axis=0)
        # landmark_pose_lidar_frame = np.append(landmark_pose_lidar_frame, np.array([1]))
        # landmark_pose_base_frame =  np.matmul(T_ld2b, np.reshape(landmark_pose_lidar_frame, (4,1)))
        landmark_pose_base_frame = landmark_pose_lidar_frame
        landmark_pose_base_frame[0] += p_ld2b[0]
        landmark_pose_base_frame[1] += p_ld2b[1]
        landmark_pose_base_frame[2] += p_ld2b[2]
        return landmark_pose_base_frame;

    def get_center(self, bboxes):
        self.lidar_to_img_plane()
        self.center_from_bbox(bboxes)
        self.nearest_neighbor(bboxes)

    def publish_landmark_info(self, landmark_id, landmark_label, landmark_pose, header):
        landmarks_info_msg = Landmarksmsg()
        landmarks_info_msg.header = header
        landmarks_info_msg.header.frame_id = '/landmarks'
        num_bboxes = len(landmark_pose)
        for i in range(num_bboxes):
            landmark_pose_base_frame = self.lidar_to_base(landmark_pose[i])
            landmark_info_msg = Landmarkmsg()
            # landmark_info_msg.id = landmark_id[i]
            landmark_info_msg.label = landmark_label[i]
            landmark_info_msg.pose.x = landmark_pose_base_frame[0]
            landmark_info_msg.pose.y = landmark_pose_base_frame[1]
            landmark_info_msg.pose.z = landmark_pose_base_frame[2]
            landmarks_info_msg.landmarks.append(landmark_info_msg)
        self.landmark_pub.publish(landmarks_info_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    L = LandmarkDetector()
    L.run()
