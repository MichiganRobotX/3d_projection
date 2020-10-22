import numpy as np


class Landmarks:
    # This class stores landmarks information
    def __init__(self):
        self.landmarks = []     # list of np arrays
        self.landmarks_classification = [] # id corr to self.landmarks

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
