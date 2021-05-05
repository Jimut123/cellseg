
import numpy as np
import cv2 
from matplotlib import pyplot as plt

# input the image
img = cv2.imread('slide_1.png', cv2.IMREAD_COLOR)

print(img.shape)

# resize it for faster and efficient computation, we can rescale it 
# afterwards
img = cv2.resize(img,(667, 500), cv2.INTER_AREA)




# norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# plt.title("Normalized Image")
# plt.imshow(norm_image)
# plt.show()

# The first option is image, then the size of the window which will 
# have the colour value and the last is the colour scale

shifted = cv2.pyrMeanShiftFiltering(img, 30, 30)
plt.title("shifted")
plt.imshow(shifted)
plt.show()



gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

plt.title("grayscale")
plt.imshow(gray, cmap='gray')
plt.show()

# use noise reduction via bluring
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

plt.title("blurred")
plt.imshow(blurred, cmap='gray')
plt.show()

thresh = cv2.threshold(blurred, blurred.mean(), 255,
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