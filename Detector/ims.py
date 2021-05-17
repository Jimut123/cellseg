import cv2
im = cv2.imread('basophil_1.jpg')
im = cv2.resize(im,(1000,1000))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 100, 100)
cv2.imshow("image", im)

cv2.resizeWindow('image', 1000, 1000)

cv2.waitKey(0)