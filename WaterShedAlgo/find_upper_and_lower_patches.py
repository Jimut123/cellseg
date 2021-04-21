import glob
import cv2
import numpy as np

all_imgs = glob.glob('masks/*.png')

r_min_list = []
r_max_list = []

b_min_list = []
b_max_list = []

g_min_list = []
g_max_list = []

for images in all_imgs:
    im = cv2.imread(images,cv2.IMREAD_COLOR)
    im = np.array(im)
    # red
    r_min = im[..., 0].min()
    r_max = im[..., 0].max()
    # green
    g_min = im[..., 1].min()
    g_max = im[..., 1].max()
    # blue
    b_min = im[..., 2].min()
    b_max = im[..., 2].max()
    # print(r_min, " ",r_max)
    # print(g_min, " ",g_max)
    # print(b_min, " ",b_max)

    r_min_list.append(r_min)
    r_max_list.append(r_max)

    g_min_list.append(g_min)
    g_max_list.append(g_max)

    b_min_list.append(b_min)
    b_max_list.append(b_max)

print("Max red channel = ",max(r_max_list))
print("Min red channel = ",min(r_min_list))
print("Max green channel = ",max(g_max_list))
print("Min green channel = ",max(g_min_list))
print("Max blue channel = ",max(b_max_list))
print("Min blue channel = ",max(b_min_list))



"""
Max red channel =  250
Min red channel =  54
Max green channel =  230
Min green channel =  38
Max blue channel =  252
Min blue channel =  197
"""