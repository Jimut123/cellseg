#!/usr/bin/python

# Standard imports

import sys
import cv2 
import glob
import numpy as np
import matplotlib.pyplot as plt

# Read image
all_imgs = glob.glob('../samples/*')

im = cv2.imread(all_imgs[3], cv2.IMREAD_UNCHANGED)
im = cv2.resize(im, (360, 360))
src = im

scale = 1
delta = 0
fig = plt.figure()
counter = 0

sobel_imgs = []
for i_loop in range(10):
    for j_loop in range(10):
        if i_loop == j_loop and i_loop > 0:
            scale = i_loop
            delta = j_loop
            ddepth = cv2.CV_16S
            src = cv2.GaussianBlur(src, (3, 3), 0)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            # Gradient-Y
            # grad_y = cv2.Scharr(gray,ddepth,0,1)
            grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            counter += 1
            ax1 = fig.add_subplot(3,3,counter)
            sobel_imgs.append(grad)
            ax1.imshow(grad, cmap="gray")
plt.show()

fig = plt.figure()
counter = 1
total_thresh = np.zeros((360, 360))
for imgs in sobel_imgs:
    # mean = np.mean(imgs)
    # mean = np.max(imgs)
    print(imgs.shape)
    hist = cv2.calcHist([imgs],[0],None,[256],[0,256])
    mean = hist[128]
    print("mean = ",mean)
    ret, thres = cv2.threshold(imgs,0,mean,cv2.THRESH_OTSU)
    ax1 = fig.add_subplot(3,3,counter)
    ax1.imshow(thres, cmap="gray")
    total_thresh += imgs
    counter += 1
plt.show()

total_thresh = total_thresh/10
print("shape = ", total_thresh.shape)
print("max = ", np.max(total_thresh))
print("min = ", np.min(total_thresh))
fig = plt.figure()

blurred_thresh = cv2.GaussianBlur(total_thresh,(5,5),0)
# ax1 = fig.add_subplot(1,1,counter)
plt.imshow(blurred_thresh, cmap="gray")
plt.show()
# hist = cv2.calcHist([blurred_thresh],[0],None,[256],[0,256])
# max = hist[256]
maximum = np.amax(blurred_thresh)
coord = np.where(blurred_thresh == maximum)
print("coord = ",coord)
x, y =  coord[0][0], coord[1][0]
cv2.rectangle(blurred_thresh,(x-70,y-70),(x+100,y+100),(0,255,0),2)
plt.imshow(blurred_thresh, cmap="gray")
plt.show()








