import os
import cv2
class TumDataset:
    def __init__(self, dataset_path):
        self.left_folder = os.path.join(dataset_path, "cam0", "data")
        self.right_folder = os.path.join(dataset_path, "cam1", "data")
        self.left_images = sorted(os.listdir(self.left_folder))
        self.right_images = sorted(os.listdir(self.right_folder))
        self.length = len(self.left_images)
        print("Dataset loaded")
        print("Total frames:", self.length)
    def get_frame(self, i):
        left_path = os.path.join(self.left_folder, self.left_images[i])
        right_path = os.path.join(self.right_folder, self.right_images[i])
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        return left_img, right_img