"""
Visualize Mask RCNN from JSON
Jimut Bahan Pal
"""

import cv2
import json
import random
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

vibrant_colors = [[0,0,255], [0,255,0], [0,255,255], [255,0,0], [255,0,255], [255,255,0]]


with open('data_test_smear_mixed/results.json') as json_data:
    all_json_annotations = json.load(json_data)
    json_data.close()


for data in all_json_annotations['data']:
    # get the result.json and parse through the data

    file_name = all_json_annotations['data'][data]['filename']
    height = all_json_annotations['data'][data]['height']
    width = all_json_annotations['data'][data]['width']
    valid_image_name = "data_test_smear_mixed/"+file_name
    
    img = Image.open(valid_image_name).convert("RGBA")
    # print("Max of Image = ",np.array(img).max())
    
    img = np.array(img)
    get_image = cv2.imread(valid_image_name,cv2.IMREAD_COLOR)
    imArray = np.asarray(img)
    print("filename = ",file_name)
    print("height = ",height)
    print("width = ",width)

    get_bbox_coords = [] # x, y, w, h, label, col
    get_all_masks = []

    for item in all_json_annotations['data'][data]['masks']:

        # print(item)
        class_name = item['class_name']
        print("Class Name = ", class_name)
        class_score = item['score']
        print("Class Score = ", class_score)
        vertices_list = item['vertices']

        x1 = item['bounding_box']['x1'] 
        x2 = item['bounding_box']['x2']   
        y1 = item['bounding_box']['y1']   
        y2 = item['bounding_box']['y2']   

        w = x2-x1
        h = y2-y1
        label = class_name
        
        print(" x1 = {}, x2 = {}, y1 = {}, y2 = {}".format(x1,x2,y1,y2))

        vertices_list = item['vertices']
        vert_list_m = []
        for ite_ in vertices_list:
            vert_list_m.append(tuple(ite_))
        # print(vertices_list)
        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(vert_list_m, outline=1, fill=1)

        maskIm = np.array(maskIm) * 255
        # plt.imshow(maskIm)
        # plt.show()
        print(maskIm.max())
        act_mask = np.zeros_like(get_image)
        
        act_mask[:,:,0] = maskIm
        act_mask[:,:,1] = maskIm
        act_mask[:,:,2] = maskIm
        print("--"*40,get_image.shape,", ",act_mask.shape)
        
        # plt.imshow(act_mask)
        # plt.show()

        green_mask = get_image.copy()
        col = vibrant_colors[random.randint(0,5)]

        green_mask[(act_mask==255).all(-1)] = col
        
        print("green mask shape = ",green_mask.shape)
        # plt.imshow(green_mask[:,:,::-1])
        # plt.show()

        get_all_masks.append(green_mask)
        get_bbox_coords.append([x1,y1,w,h,label,col])
    
    # plt.imshow(img[:,:,0])
    # plt.show()
    # plt.imshow(img[:,:,1])
    # plt.show()
    # plt.imshow(get_image)
    # plt.show()
    print("Shape of Get Image = ",get_image.shape)
    print(float(1/(len(get_all_masks)+1)))
    print("Max and Min of get_image = ",get_image.max(),get_image.min())
    final_masked_im = np.zeros(get_image.shape)
    final_masked_im = float(1/(len(get_all_masks)+1))*get_image
    #print("giu",np.unique(get_image)[:50])
    #print("fmiu",np.unique(final_masked_im)[:50])
    # print("Max and Min of final_masked_im = ",final_masked_im.max(),final_masked_im.min())
    
    # plt.imshow(final_masked_im.astype("uint8"))
    # plt.show()
    print("fff",final_masked_im.max())

    print("len get all mask = ",len(get_all_masks))
    for image in get_all_masks:
        # plt.imshow(image)
        # plt.show()
        print("max = ",image.max(),"min = ",image.min())
        final_masked_im = final_masked_im + float(1/(len(get_all_masks)+1))*image
        print("Final max = ",final_masked_im.max(),"min = ",final_masked_im.min())
        # plt.imshow(final_masked_im[:,:,::-1])
        # plt.show()
    print("fin = ",final_masked_im.max())

    for items in get_bbox_coords:
        x,y,w,h,name, col = int(items[0]), int(items[1]), int(items[2]), int(items[3]), items[4], items[5]
        cropped_img = np.zeros((w,h,3))
        cropped_img = img[y:y+h,x:x+w]
        # make DA preds here in the cropped image
        # plt.imshow(cropped_img[:,:,::-1])
        # plt.show()
        print(x,y,w,h,name)
        #col = vibrant_colors[random.randint(0,5)]
        cv2.rectangle(final_masked_im, (x,y), (x+w,y+h), col,15)
        # img, text, coord, type of font, size, col, thickness
        cv2.putText(final_masked_im, str(name), (x, y), 0, 3, [0,0,0], 10)
    save_im_name = valid_image_name.split('.')[0]+"_mask_rcnn.jpg"
    cv2.imwrite(save_im_name,final_masked_im[:,:,::-1])
    # plt.imshow(final_masked_im)
    # plt.show()
    # mask_gt_save_name = get_im_path.split('.')[0]+"_gt.jpg"
    # cv2.imwrite(mask_gt_save_name,final_masked_im)


        
