import os
import glob
import cv2, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from google.colab.patches import cv2_imshow
# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
from tqdm import tqdm


sub_folders = glob.glob('PBC_dataset_normal_DIB/*')

total_files_list = []
folders_list = []
total_img_cnts = 0
for folders in sub_folders:
    print(folders)
    folders_list.append(folders.split('/')[1])
    files_from_folders = glob.glob('{}/*'.format(folders))
    total_img_cnts += len(files_from_folders)
    print(len(files_from_folders))
    print("Cumulative = ", total_img_cnts)
    for files in files_from_folders:
        total_files_list.append(files)

print("Total img cnts. = ",total_img_cnts)

print(folders_list)

for folder in folders_list:
    os.system('mkdir PBC_dataset_normal_DIB_cropped/{}'.format(folder))

def get_contour(img_bin):
    """Get connected domain

         :param img: input picture
         :return: Maximum connected domain
    """
    # Grayscale, binarization, connected domain analysis
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[2]

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v
list_ = [ 3329, 6224, 7438, 10555, 11975, 13526, 14744]
H_2 , W_2 = int(363/2), int(360/2)
print(H_2,W_2)

for image_names in tqdm(total_files_list):
    #print(image_names)
    image_name_to_save = image_names.split('/')[-1]
    folder_name = image_names.split('/')[1]
    #print(folder_name)
    save_path = 'PBC_dataset_normal_DIB_cropped/{}/{}'.format(folder_name,image_name_to_save)
    print("save path = ",save_path)
    image = cv2.imread(image_names)
    img = image.copy()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,20,20])
    upper_blue = np.array([300,245,245])
    
    # lower_blue = np.array([min_r_hsv,min_g_hsv,min_b_hsv])
    # upper_blue = np.array([int(max_r_hsv),int(max_g_hsv),int(max_b_hsv)])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    
    mask = cv2.dilate(mask, None, iterations=3)
    
    res = cv2.bitwise_and(img,img, mask= mask)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map

    D = ndimage.distance_transform_edt(mask)
    localMax = peak_local_max(D, indices=False, min_distance=20,
        labels=mask)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=mask)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    shifted = cv2.pyrMeanShiftFiltering(image, 50, 50)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    # loop over the unique labels returned by the Watershed
    # algorithm
    rad_circles_coords =  []

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask_1 = np.zeros(gray.shape, dtype="uint8")
        mask_1[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask_1.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        rad_circles_coords.append((x,y,r))
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    max_r = 0
    for iter in rad_circles_coords:
        x = iter[0]
        y = iter[1]
        r = iter[2]
        if r > max_r:
            max_r = int(r)
            max_x = int(x)
            max_y = int(y)
    # H_2 , W_2
    #crop = res[max_y-max_r:max_y+max_r,max_x-max_r:max_x+max_r].copy()
    crop = res[H_2-max_r:H_2+max_r,W_2-max_r:W_2+max_r].copy()
    cv2.imwrite(save_path,crop)
    # plt.imshow(crop[:,:,::-1])
    # plt.show()
    
