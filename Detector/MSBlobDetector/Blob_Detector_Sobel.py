#!/usr/bin/python

# Standard imports

import sys
import cv2 
import glob
import numpy as np
import matplotlib.pyplot as plt


from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray



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



# https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html

imgs = sobel_imgs[1]
# imgs = sobel_imgs[4]
image = imgs
image_gray = imgs
blobs_log = blob_log(image_gray, max_sigma=90, num_sigma=10, threshold=0.25)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=80, threshold=1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=90, threshold=0.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
        'Determinant of Hessian']

sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()










