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

        left = cv2.imread(left_path)
        right = cv2.imread(right_path)

        return left, right


dataset_path = "C:/Users/RoboticsLab/PycharmProjects/stereo_vo_project/dataset/dataset-room2_512_16/mav0"

dataset = TumDataset(dataset_path)

left, right = dataset.get_frame(0)

cv2.imshow("Left Camera", left)
cv2.imshow("Right Camera", right)

cv2.waitKey(0)
cv2.destroyAllWindows()