import numpy as np
import cv2

#height then width
#uint8 = each number from 0-255
image = np.zeros((3,3,3), dtype=np.uint8)

#BGR
image[:,:] = [0,0,255]

print("BGR Image:")
print(image)

imageToHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cvt = convert

print("HSV Image:")
print(imageToHSV)