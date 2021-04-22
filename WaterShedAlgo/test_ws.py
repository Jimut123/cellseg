
import numpy as np
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread('slide_9.png', cv2.IMREAD_COLOR)

print(img.shape)


img = cv2.resize(img,(667, 500), cv2.INTER_AREA)

shifted = cv2.pyrMeanShiftFiltering(img, 30, 60)
plt.title("shifted")
plt.imshow(shifted)
plt.show()

gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
plt.title("thresh")
plt.imshow(thresh)
plt.show()

threshold_bg = thresh.copy()


gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)



ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

plt.title("Thresh")
plt.imshow(thresh)
plt.show()

# noise removal
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

plt.title("Opening")
plt.imshow(opening)
plt.show()
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=1)


sure_fg = sure_bg.copy()
plt.title("Sure fg")
plt.imshow(sure_fg)
plt.show()

sure_bg = threshold_bg.copy()
plt.title("Sure bg")
plt.imshow(sure_bg)
plt.show()


# sure_bg = thresh.copy()

# Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)


unknown = cv2.subtract(sure_fg,sure_bg)

plt.title("Unknown")
plt.imshow(unknown)
plt.show()



# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

plt.title("Markers")
plt.imshow(markers)
plt.show()

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.title("Watershed")
plt.imshow(img)
plt.show()