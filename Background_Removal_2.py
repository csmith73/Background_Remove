import cv2
import numpy as np

img = cv2.imread("./test_data/Background_Removal/CM_1.jpg")
mask = cv2.imread("./test_data/Background_Removal/CM_1_Mask.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
transparent[:,:,0:3] = img
transparent[:, :, 3] = mask

cv2.imwrite("./test_data/Background_Removal/Results/Mav_Result_2.png", transparent)