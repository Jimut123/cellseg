#!/usr/bin/python

# Standard imports

import sys
import cv2 
import glob
import math
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
    im = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
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

    # for imgs in sobel_imgs:
    imgs = sobel_imgs[2]
    imgs = cv2.GaussianBlur(imgs,(5,5),0)

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
    seg_map_conf = np.zeros((360,360))
    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            blank_img = np.zeros((360,360))
            color = (1, 1, 1, 0)
            cv2.circle(blank_img, (int(x), int(y)), int(r), color, thickness=-1)
            factor = math.exp(math.log2(r))*blank_img
            seg_map_conf += factor
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
    
    plt.tight_layout()
    plt.show()
    print(seg_map_conf.shape)
    plt.imshow(seg_map_conf)
    plt.show()
    flat=seg_map_conf.flatten()
    mean = np.mean(flat)
    print("mean = ",mean)
    print(flat.max())
    unique_set = list(set(flat))
    unique_set.sort()
    print("unique set = ",unique_set)
    set_val = int(len(unique_set)/1.2)
    print("set val = ",set_val)
    print("unique_set[set_val] = ",unique_set[set_val])
    # ret, thres = cv2.threshold(seg_map_conf,unique_set[set_val],unique_set[-1],cv2.THRESH_OTSU)
    # plt.imshow(thres)
    # plt.show()
    # seg_map_conf[seg_map_conf > unique_set[set_val]] = 1
    seg_map_conf[seg_map_conf < unique_set[set_val]] = 0
    plt.imshow(seg_map_conf)
    plt.show()
    kernel = np.ones((5,5),np.uint8) #cv2.getGaussianKernel(5, 0)
    erosion = cv2.erode(seg_map_conf,kernel,iterations = 6)
    erosion = cv2.dilate(erosion,kernel,iterations = 2)
    erosion[erosion > 0] = 1
    plt.imshow(erosion)
    plt.show()
    print("max ero = ",erosion.max())
    img_color_mask = src.copy()
    img_color_mask[:,:,0] = erosion
    img_color_mask[:,:,1] = src[:,:,1]
    img_color_mask[:,:,2] = src[:,:,2]
    # img_color_mask = img_color_mask*2
    # img_color_mask = np.clip(img_color_mask, 0, 1)
    plt.imshow(img_color_mask)
    plt.show()












