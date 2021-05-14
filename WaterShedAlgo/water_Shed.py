import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



img = cv.imread('slide_1.png')

print(img.shape)
img = cv.resize(img,(400, 300), cv.INTER_AREA)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# img = gray.copy()

# img -= img.min() 
# img /= img.max()

# img *= 255 # [0, 255] range

plt.imshow(gray,cmap='gray')
plt.show()



ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


plt.imshow(thresh)
plt.show()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
plt.imshow(opening)
plt.show()

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

plt.imshow(sure_bg)
plt.show()

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)

plt.imshow(sure_fg)
plt.show()

unknown = cv.subtract(sure_bg,sure_fg)

plt.imshow(unknown)
plt.show()



# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0


plt.imshow(markers)
plt.show()

markers = cv.watershed(img,markers)
img[markers == -1] = [0,255,0]


plt.imshow(img[:,:,::-1])
plt.axis('off')
plt.show()