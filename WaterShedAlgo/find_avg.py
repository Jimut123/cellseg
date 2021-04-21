import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# all_imgs = glob.glob('masks/*.png')

# avg_mask = np.zeros((200,200))
# total_mask = np.zeros((200,200,3))

# mask_name_count = 0



# for images in all_imgs:

#     im = cv2.imread(images,cv2.IMREAD_COLOR)
#     im = np.array(im)
#     avg_mask = im[0:200,0:200]
#     mask_name_count += 1

#     mask_name = "mask_{}.png".format(mask_name_count)
#     print(mask_name)
#     cv2.imwrite(mask_name, avg_mask[:,:,::-1])
#     # plt.imshow(avg_mask)
#     # plt.show()
#     total_mask = total_mask + np.array(avg_mask, dtype=np.float)/14.0
#     plt.imshow(total_mask)
#     plt.show()

# avg_mask = total_mask
# plt.imshow(avg_mask)
# plt.show()

import os, numpy, PIL
from PIL import Image

# Access all PNG files in directory
allfiles=os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(imlist[0]).size
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,3),numpy.float)

# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    imarr=numpy.array(Image.open(im),dtype=numpy.float)
    arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("Average.png")
out.show()


"""
Max red channel =  250
Min red channel =  54
Max green channel =  230
Min green channel =  38
Max blue channel =  252
Min blue channel =  197
"""

