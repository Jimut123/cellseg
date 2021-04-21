"""
Max red channel =  250
Min red channel =  54
Max green channel =  230
Min green channel =  38
Max blue channel =  252
Min blue channel =  197
"""

import numpy as np
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread('slide_1.png', cv2.IMREAD_COLOR)

avg = cv2.imread('Average.png', cv2.IMREAD_COLOR)

print(avg.shape)
plt.imshow(avg)
plt.show()

avg = np.array(avg)

print(avg[..., 0].min())
print(avg[..., 0].max())
print(avg[..., 1].min())
print(avg[..., 1].max())
print(avg[..., 2].min())
print(avg[..., 2].max())






# define range of blue color in HSV
lower_blue = np.array([113, 4, 123])
upper_blue = np.array([162, 49, 158])

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
