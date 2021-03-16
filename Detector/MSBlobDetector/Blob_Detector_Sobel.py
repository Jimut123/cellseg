#!/usr/bin/python

# A classical approach to Segmentation -- with almost 95% accuracy
# Jimut Bahan Pal -- 10th March 2021
# This file is suited for the PCB Dataset

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



# Read image names from the folder samples
all_imgs = glob.glob('../samples/*')

for im_name in all_imgs:
    # im = cv2.imread(all_imgs[3], cv2.IMREAD_UNCHANGED)
    # colour image value
    im = cv2.imread(im_name, cv2.IMREAD_COLOR)
    # resize to standard value
    im = cv2.resize(im, (360, 360))
    print("image shape = ",im.shape)
    src = im

    scale = 1
    delta = 0
    fig = plt.figure()
    counter = 0

    sobel_imgs = []
    # sobel scale selectors, select the 2nd
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
    # smoothen the image a bit
    imgs = cv2.GaussianBlur(imgs,(3,3),0)

    image = imgs
    image_gray = imgs
    # some selected values of Laplacian Of Gaussian
    blobs_log = blob_log(image_gray, max_sigma=80, num_sigma=10, threshold=0.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    # print(blobs_log[:, 2])
    # max_log_rad = max(blobs_log[:, 2])
    # print("Max log rad = ",max_log_rad)

    # some selected values of Difference Of Gaussian
    blobs_dog = blob_dog(image_gray, max_sigma=80, threshold=1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    # max_dog_rad = max(blobs_dog)
    # print("Max log rad = ",max_dog_rad)

    # some selected values of Difference Of Gaussian
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
            # get the circle and plot it for visualization
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            blank_img = np.zeros((360,360))
            color = (1, 1, 1, 0)
            # solid circle generated from the visualization
            cv2.circle(blank_img, (int(x), int(y)), int(r), color, thickness=-1)
            # a factor is multiplied so that the larger the circle, the more the value is added
            factor = math.exp(math.log2(r))*blank_img
            seg_map_conf += factor
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
    
    plt.tight_layout()
    plt.show()
    # check the shape
    print(seg_map_conf.shape)
    # show the segmented map of confidence generated
    plt.imshow(seg_map_conf)
    plt.show()
    # find the mean of the flattened values
    flat=seg_map_conf.flatten()
    mean = np.mean(flat)
    print("mean = ",mean)
    print(flat.max())
    # find the set of thresholds from the layers of map generated
    unique_set = list(set(flat))
    unique_set.sort()
    print("unique set = ",unique_set)
    # we select a threshold value of 1/1.17 so that we get the most important maps from the layers
    set_val = int(len(unique_set)/1.17)
    print("set val = ",set_val)
    print("unique_set[set_val] = ",unique_set[set_val])
    # ret, thres = cv2.threshold(seg_map_conf,unique_set[set_val],unique_set[-1],cv2.THRESH_OTSU)
    # plt.imshow(thres)
    # plt.show()
    #seg_map_conf[seg_map_conf > unique_set[set_val]] = unique_set[-1]
    # if the confidence of the layer is less than the threshold value selected then we make it to 0
    seg_map_conf[seg_map_conf < unique_set[set_val]] = 0
    # we set the highest confidence boundary values to the maximum confidence obtained
    seg_map_conf[seg_map_conf > 0] = unique_set[-1]
    plt.imshow(seg_map_conf)
    plt.show()
    # display the original image
    plt.imshow(src[:,:,::-1])
    plt.show()
    # we use Gaussian Kernel for the erosion
    kernel = np.ones((5,5),np.uint8) #cv2.getGaussianKernel(5, 0)
    erosion = cv2.erode(seg_map_conf,kernel,iterations = 5)
    print("max = ",np.amax(erosion))
    erosion = erosion/np.amax(erosion)
    print("max = ",np.amax(erosion))
    plt.imshow(erosion)
    plt.show()
    # overlay the map which is to be segmented
    img_color_mask = src.copy()
    img_color_mask[:,:,0] = erosion*254*0.5 + src[:,:,0]*0.5
    img_color_mask[:,:,1] = erosion*254*0.5 + src[:,:,1]*0.5
    img_color_mask[:,:,2] = erosion*254*0.5 + src[:,:,2]*0.5
    img_color_mask = img_color_mask
    img_color_mask = np.clip(img_color_mask, 0, 255)
    plt.imshow(img_color_mask[:,:,::-1])
    plt.show()












