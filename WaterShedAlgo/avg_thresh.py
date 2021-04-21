
import numpy as np
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread('slide_9.png', cv2.IMREAD_COLOR)



#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([54, 38, 197])
upper_blue = np.array([250, 230, 252])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(img, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)

plt.imshow(img)
plt.show()

plt.imshow(mask)
plt.show()
plt.imshow(res)
plt.show()