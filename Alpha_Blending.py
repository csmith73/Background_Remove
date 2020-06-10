import cv2
from PIL import Image
import numpy as np

# Read the images
foreground = cv2.imread("./test_data/Alpha_Blending_Input/M_Input.jpg")

background = np.zeros([foreground.shape[0],foreground.shape[1],3],dtype=np.uint8)
background.fill(255)
#background = np.array(background)
#background = cv2.imread("./test_data/test_data/llama.jpg")
alpha = cv2.imread("./test_data/Alpha_Blending_Input/M_mask.png")

# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)

# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float) / 255
# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)

# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)
# Add the masked foreground and background.
outImage = cv2.add(foreground, background)

# Display image
#cv2.imshow("outImg", outImage / 255)
#cv2.waitKey(0)
cv2.imwrite("./test_data/u2net_results/Alpha_Blending/Alpha_Output2.png", outImage)
