

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

import plotly.graph_objects as go


# Read image names from the folder samples
all_imgs = glob.glob('../samples/*')

for im_name in all_imgs:
    # im = cv2.imread(all_imgs[3], cv2.IMREAD_UNCHANGED)
    # colour image value
    print("name = ",im_name)
    im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
    # resize to standard value
    im = cv2.resize(im, (360, 360))
    print("image shape = ",im.shape)

    fig = go.Figure(data=[
        go.Surface(z=im)])
    fig.show()
    input("enter a key")
