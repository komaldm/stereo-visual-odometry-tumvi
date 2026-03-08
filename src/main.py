import cv2
from dataset_loader import TumDataset
dataset_path = "C:/Users/RoboticsLab/PycharmProjects/stereo_vo_project/dataset/dataset-room2_512_16/mav0"
dataset=TumDataset(dataset_path)
left, right = dataset.get_frame(0)
cv2.imshow("left", left)
cv2.imshow("right", right)
cv2.waitKey(0)
cv2.destroyAllWindows()