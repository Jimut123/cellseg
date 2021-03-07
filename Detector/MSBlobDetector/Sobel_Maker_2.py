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






