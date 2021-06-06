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


###########################################################################
######## LOAD MODEL AND STUFFS ############################################
###########################################################################

from tensorflow.keras.utils import to_categorical
from PIL import Image

import sys
from tensorflow.keras.utils import to_categorical
from PIL import Image

import tensorflow as tfpbc_8pbc_8
from keras.regularizers import l2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D,\
                                    GlobalMaxPool2D, Dropout, SpatialDropout2D, add, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Precision, Recall, SensitivityAtSpecificity, PrecisionAtRecall, \
                                     TruePositives, TrueNegatives, FalsePositives, FalseNegatives


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import time
import cv2
import os


rev_index = {0: 'lymphocyte', 1: 'monocyte', 2: 'myelocyte', 3: 'neutrophil', 4: 'blast', 5: 'promyelocyte', 6: 'metamyelocyte', 7: 'band', 8: 'basophil', 9: 'eosinophil'}
index = {'lymphocyte': 0, 'monocyte': 1, 'myelocyte': 2, 'neutrophil': 3, 'blast': 4, 'promyelocyte': 5, 'metamyelocyte': 6, 'band': 7, 'basophil': 8, 'eosinophil': 9}

H, W, C = 360, 360, 3
N_LABELS = len(index)
D = 1

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, GlobalAveragePooling2D

frozen = Xception(weights="imagenet", input_shape=(360,360,3), include_top=False)
frozen.summary()

trainable = frozen.output
trainable = GlobalAveragePooling2D()(trainable)
#print(trainable.shape)
trainable = Dense(128, activation="relu")(trainable)
trainable = Dense(32, activation="relu")(trainable)
trainable = Dense(N_LABELS, activation="softmax")(trainable)
model = Model(inputs=frozen.input, outputs=trainable)
model.summary()

# model.layers
# for layer in model.layers[:-4]:
#     layer.trainable = False
for layer in model.layers:
    print(layer, layer.trainable)

from keras.optimizers import Adam
opt = Adam(lr=1e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy',
            #experimental_run_tf_function=False,
            metrics = ['accuracy', AUC(curve="ROC"), Precision(), Recall(), \
            TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()]
            )

from tensorflow.keras.utils import to_categorical
from PIL import Image

from tensorflow import keras
model = keras.models.load_model('classification_Smear_Slides_DA_cropped_aug_8_xception_fine_tuned_100e.h5')

###########################################################################

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

    masks_act = []

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
        # plt.imshow(act_mask)
        # plt.show()
        masks_act.append(act_mask)
        print("--"*40,get_image.shape,", ",act_mask.shape)
        
        # plt.imshow(act_mask)
        # plt.show()

        colour_mask = np.zeros(get_image.shape) #get_image.copy()
        col = vibrant_colors[random.randint(0,5)]

        colour_mask[(act_mask==255).all(-1)] = col
        
        print("green mask shape = ",colour_mask.shape)
        # plt.imshow(green_mask[:,:,::-1])
        # plt.show()

        get_all_masks.append(colour_mask)
        # plt.imshow(colour_mask)
        # plt.show()
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
    # final_masked_im = float(1/(len(get_all_masks)+1))*get_image
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
        final_masked_im = final_masked_im + image
        print("Final max = ",final_masked_im.max(),"min = ",final_masked_im.min())
        # plt.imshow(final_masked_im[:,:,::-1].astype('uint8'))
        # plt.show()
    print("fin = ",final_masked_im.max())
    np.clip(final_masked_im, 0, 255, out=final_masked_im)
    # plt.imshow(final_masked_im.astype("uint8"))
    # plt.show()
    # plt.imshow(image)
    # plt.show()
    # final_masked_im = 0.4*final_masked_im +0.6*get_image
    final_masked_im = 0.4*final_masked_im +0.9*get_image
    for items,msks in zip(get_bbox_coords,masks_act):
        x,y,w,h,name, col = int(items[0]), int(items[1]), int(items[2]), int(items[3]), items[4], items[5]
        cropped_img = np.zeros((w,h,3))
        plt.imshow(msks[y:y+h,x:x+w])
        plt.show()
        cropped_img = get_image[y:y+h,x:x+w] #cv2.bitwise_and(get_image[y:y+h,x:x+w],get_image[y:y+h,x:x+w],mask = msks[y:y+h,x:x+w:,0]/255.0) #* /255.0
        plt.imshow(cropped_img)
        plt.show()
        # make DA preds here in the cropped image
        cropped_img = cv2.resize(cropped_img, (360,360))
        print("Shape = ",cropped_img.shape)
        y_pred = model.predict(cropped_img[np.newaxis, ...]/255.0)
        print("softmax = ",y_pred)
        prd_from_model = rev_index[int(tf.math.argmax(y_pred, axis=-1))]
        print("Prediction = ",prd_from_model)
        print(x,y,w,h,name)
        #col = vibrant_colors[random.randint(0,5)]
        cv2.rectangle(final_masked_im, (x,y), (x+w,y+h), col,15)
        # img, text, coord, type of font, size, col, thickness
        cv2.putText(final_masked_im, str(name), (x, y), 0, 3, [0,0,0], 10)
        cv2.putText(final_masked_im, str(prd_from_model), (x, y+h), 0, 3, [0,0,255], 10)
    save_im_name = valid_image_name.split('.')[0]+"_mask_rcnn.jpg"
    save_im_predname = valid_image_name.split('.')[0]+"_model_pred.jpg"
    max_fi =  final_masked_im.max()
    ratio = float(255/max_fi)
    print("ratio = ",ratio)
    cv2.imwrite(save_im_predname,final_masked_im[:,:,::-1])
    # plt.imshow((final_masked_im*ratio).astype('uint8'))
    # plt.show()
    # mask_gt_save_name = get_im_path.split('.')[0]+"_gt.jpg"
    # cv2.imwrite(mask_gt_save_name,final_masked_im)


        
