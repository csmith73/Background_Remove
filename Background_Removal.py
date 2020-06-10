from skimage import io
import numpy as np
img = io.imread("./test_data/Background_Removal/Mav_1.jpg")
mask = io.imread("./test_data/Background_Removal/Mask_Mav_1.png")
mask2 = np.where((mask<200),0,1).astype('uint8')
img = img*mask2[:,:,0,np.newaxis]
io.imsave("./test_data/Background_Removal/Results/Mav_Result_1.png", img)
io.imshow(img)
io.show()


