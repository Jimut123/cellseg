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

for im_name in all_imgs:
    # im = cv2.imread(all_imgs[3], cv2.IMREAD_UNCHANGED)
    img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (360, 360))
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html

    imgs = cv2.GaussianBlur(im,(5,5),0)
    
    image = imgs
    image_gray = imgs
    blobs_log = blob_log(image_gray, max_sigma=80, num_sigma=10, threshold=0.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    # print(blobs_log[:, 2])
    # max_log_rad = max(blobs_log[:, 2])
    # print("Max log rad = ",max_log_rad)

    blobs_dog = blob_dog(image_gray, max_sigma=80, threshold=1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    # max_dog_rad = max(blobs_dog)
    # print("Max log rad = ",max_dog_rad)

    blobs_doh = blob_doh(image_gray, max_sigma=80, threshold=0.001)
    # max_doh_rad = max(blobs_doh)
    # print("Max log rad = ",max_doh_rad)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
            'Determinant of Hessian']

    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image,cmap='gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()










