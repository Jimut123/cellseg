
import tensorflow as tf
from tensorflow import keras
import keras
import keras.utils
from keras import utils as np_utils
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D,\
                                    GlobalMaxPool2D, Dropout, SpatialDropout2D, add, concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import math
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

# Display

from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np

import cv2
import glob

import sys
sys.path.insert(0, '../')
from utils import get_img_array, make_gradcam_heatmap, get_jet_img



###########################################
#  LOAD MODEL

def Model_V2_Gradcam(H,W,C):
    N_LABELS = 8
    input_layer = tf.keras.Input(shape=(H, W, C))
    x_1 = tf.keras.layers.Conv2D(16, 3, activation='relu', strides=(1, 1), name="conv_16_1", padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(input_layer)
    x_2 = tf.keras.layers.Conv2D(16, 3, activation='relu', strides=(1, 1), name="conv_16_2", padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x_1)
    # x_4 = tf.keras.layers.Conv2D(16, 3, activation='relu', strides=(1, 1), name="conv_64_21", padding='same')(add([x_3,x_1]))
    x_3 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool3")(x_2)
    x_4 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="conv_32_1", padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x_3)
    x_5 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="conv_32_2", padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x_4)

    x_6 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool4")(x_5)
    x_7 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="conv_64_1", padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x_6)
    x_8 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="conv_64_2", padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x_7)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool5")(x_8)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(2, 2), name="conv_64_3", kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool6")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dropout(0.15, name="dropout_3")(x)
    x = tf.keras.layers.Dense(256, activation='relu', name="dense_64")(x)
    x = tf.keras.layers.Dense(N_LABELS, activation='softmax', name="output_layer")(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model

model = Model_V2_Gradcam(H=360, W=360, C=3)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics= ['accuracy'])
model.summary()


model = keras.models.load_model('classification_model_v2_blood_20epochs.h5')

print("MODEL LOADED!")

###########################################

index = {'platelet': 0, 'eosinophil': 1, 'lymphocyte': 2, 'monocyte': 3, 'basophil': 4, 'ig': 5, 'erythroblast': 6, 'neutrophil': 7}
rev_index = {0: 'platelet', 1: 'eosinophil', 2: 'lymphocyte', 3: 'monocyte', 4: 'basophil', 5: 'ig', 6: 'erythroblast', 7: 'neutrophil'}

all_files_samples = glob.glob('../samples/*.jpg')
img_size = (300,300)

print("All Samples : ",all_files_samples[:4])



for img_file in all_files_samples:

    all_img = []
    all_heatmap = []
    all_superimposed_img = []
    all_heatmap_ = []

    img_save_name, num = str(str(img_file.split('/')[2]).split('.')[0]).split('_')
    print("img_save_name = ", img_save_name, " num = ",num)
    # Prepare image
    # img_array = preprocess_input(get_img_array(img_path, size=img_size))

    im = Image.open(img_file)
    im = im.resize((360, 360))
    im = np.array(im) / 255.0
    get_img = np.array(im)

    #get_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

    img_array =  np.expand_dims(get_img, axis=0)
    print("Shape of input img = ",img_array.shape)
    # Make model
    # model = model_builder(weights="imagenet")

    # Print what the top predicted class is
    preds = model.predict(img_array)
    # print("Predicted:", decode_predictions(preds, top=1)[0])
    print("preds : ", preds)
    indx = int(np.argmax(preds))
    print(" Max = ", indx)
    predicted_name = rev_index[indx]
    print(" Predicted name = ", predicted_name)

    last_conv_layer_name = "conv_64_3" #"dense_2" 
    classifier_layer_names = [
        "max_pool6",         
        "flatten",         
        "dropout_3",       
        "dense_64",  
        "output_layer"
    ]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = get_img
    
    img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    print("--"*60,img.shape)
    all_img.append(img)
    all_heatmap.append(heatmap)
    all_heatmap_.append(heatmap_)
    all_superimposed_img.append(superimposed_img)
    

    # Save the superimposed image
    #save_path = "save/{}_{}.jpg".format(img_save_name, num)
    #superimposed_img.save(save_path)

    # Display Grad CAM


    last_conv_layer_name = "conv_64_2" #"dense_2" 
    classifier_layer_names = [
        "max_pool5",      
        "conv_64_3",    
        "max_pool6",         
        "flatten",         
        "dropout_3",       
        "dense_64",  
        "output_layer"
    ]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = get_img
    img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    print("--"*60,img.shape)
    all_img.append(img)
    all_heatmap.append(heatmap)
    all_heatmap_.append(heatmap_)
    all_superimposed_img.append(superimposed_img)
    

    last_conv_layer_name = "conv_64_1" 
    classifier_layer_names = [
       "conv_64_2",    
        "max_pool5",      
        "conv_64_3",    
        "max_pool6",         
        "flatten",         
        "dropout_3",       
        "dense_64",  
        "output_layer"
    ]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = get_img
    img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    print("--"*60,img.shape)
    all_img.append(img)
    all_heatmap.append(heatmap)
    all_heatmap_.append(heatmap_)
    all_superimposed_img.append(superimposed_img)
    
    

    last_conv_layer_name = "conv_32_2"
    classifier_layer_names = [
        "max_pool4",   
        "conv_64_1",    
        "conv_64_2",    
        "max_pool5",      
        "conv_64_3",    
        "max_pool6",         
        "flatten",         
        "dropout_3",       
        "dense_64",  
        "output_layer"
    ]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = get_img
    img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    print("--"*60,img.shape)
    all_img.append(img)
    all_heatmap.append(heatmap)
    all_heatmap_.append(heatmap_)
    all_superimposed_img.append(superimposed_img)

    last_conv_layer_name = "conv_32_1"
    classifier_layer_names = [
        "conv_32_2",    
        "max_pool4",   
        "conv_64_1",    
        "conv_64_2",    
        "max_pool5",      
        "conv_64_3",    
        "max_pool6",         
        "flatten",         
        "dropout_3",       
        "dense_64",  
        "output_layer"
    ]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = get_img
    img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    print("--"*60,img.shape)
    all_img.append(img)
    all_heatmap.append(heatmap)
    all_heatmap_.append(heatmap_)
    all_superimposed_img.append(superimposed_img)


    last_conv_layer_name = "conv_16_2"
    classifier_layer_names = [
        "max_pool3",     
        "conv_32_1",   
        "conv_32_2",    
        "max_pool4",   
        "conv_64_1",    
        "conv_64_2",    
        "max_pool5",      
        "conv_64_3",    
        "max_pool6",         
        "flatten",         
        "dropout_3",       
        "dense_64",  
        "output_layer"
    ]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = get_img
    img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    print("--"*60,img.shape)
    all_img.append(img)
    all_heatmap.append(heatmap)
    all_heatmap_.append(heatmap_)
    all_superimposed_img.append(superimposed_img)

    last_conv_layer_name = "conv_16_1"
    classifier_layer_names = [
       "conv_16_2",
        "max_pool3",     
        "conv_32_1",   
        "conv_32_2",    
        "max_pool4",   
        "conv_64_1",    
        "conv_64_2",    
        "max_pool5",      
        "conv_64_3",    
        "max_pool6",         
        "flatten",         
        "dropout_3",       
        "dense_64",  
        "output_layer"
    ]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = get_img
    img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    print("--"*60,img.shape)
    all_img.append(img)
    all_heatmap.append(heatmap)
    all_heatmap_.append(heatmap_)
    all_superimposed_img.append(superimposed_img)

    
    fig = plt.figure()
    all_thres = []
    count = 0

    ## Show the layer wise heatmaps here

    for img, heatmap, heatmap_, superimposed_img in zip(all_img, all_heatmap, all_heatmap_, all_superimposed_img):
        count += 1
        ax1 = fig.add_subplot(7,5,count)
        ax1.imshow(img)
        count += 1
        ax2 = fig.add_subplot(7,5,count)
        ax2.imshow(heatmap)
        count += 1
        ax3 = fig.add_subplot(7,5,count)
        ax3.imshow(heatmap_)
        count += 1
        # heatmap_gray = np.array(heatmap_)
        # heatmap_gray = cv2.cvtColor(np.array(heatmap_), cv2.COLOR_BGR2GRAY)
        # print("*"*50,np.array(heatmap_gray).shape)
        ret, thres = cv2.threshold(heatmap_,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        all_thres.append(thres)
        ax3 = fig.add_subplot(7,5,count)
        ax3.imshow(thres)
        count += 1
        ax4 = fig.add_subplot(7,5,count)
        ax4.imshow(superimposed_img)
    
    # show the added heatmap here after adding
    print("count = ",count)
    added_heatmap = np.zeros((360,360))
    for thres_ in all_thres:
        added_heatmap += thres_
    # plt.show()
    # plt.imshow(added_heatmap) 
    # plt.show()
    # ax5 = fig.add_subplot(8,5,36)
    # ax5.imshow(added_heatmap)
    
    # ret, binary_thresh = cv2.threshold(added_heatmap,maximum - 6,maximum,cv2.THRESH_BINARY)
    # plt.imshow(binary_thresh) 
    # plt.show()
    maximum = np.amax(added_heatmap)
    coord = np.where(added_heatmap == maximum)

    x, y = coord[0][0], coord[1][0]

    # all_x, all_y = coord[0], coord[1]
    # x_max = np.max(all_x)
    # x_min = np.min(all_x)
    # y_max = np.max(all_y)
    # y_min = np.min(all_y)
    # x, y = int((x_min+x_max)/2), int((y_min+y_max)/2)
    # print("Coord = ",x,y)

    print("Coord = ",x,y)
    cv2.rectangle(img,(x-100,y-100),(x+100,y+100),(0,255,0),2)
    # ax5 = fig.add_subplot(8,5,37)
    # ax5.imshow(img)
    plt.show()
    plt.imshow(added_heatmap) 
    plt.show()
    
    plt.imshow(img)
    if predicted_name == img_save_name:
        plt.title("Prediction = {}".format(predicted_name),fontsize=20).set_color('green')
    else:
        plt.title("Prediction = {}".format(predicted_name),fontsize=20).set_color('red')
    
    plt.xlabel('true: {}'.format(img_save_name),fontsize=20)
    plt.show()
    # w_thres = np.zeros((360,360))
    print("maximum = ",maximum)
    mean = np.mean(added_heatmap)
    print("mean = ", mean)
    # for i_ in range(360):
    #     for j_ in range(360):
    #         # plt.imshow(added_heatmap[i_:i_+30,j_:j_+30])
    #         # plt.show()
    #         if added_heatmap[i_:i_+30,j_:j_+30].any() >= mean:
    #             print("i_ = {}, j_ = {}".format(i_,j_))
    #             w_thres[i_:i_+30,j_:j_+30] = maximum
    #         j_+= 30
    #     i_+= 30
    # plt.imshow(w_thres)
    # plt.show()

    heatmap_thres = added_heatmap.copy()
    heatmap_thres[heatmap_thres < maximum] = 0
    grad_x = cv2.Sobel(heatmap_thres, cv2.CV_16S, 1, 0, ksize=3, scale=3, delta=3, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(heatmap_thres, cv2.CV_16S, 0, 1, ksize=3, scale=3, delta=3, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    kernel = np.ones((10,10),np.uint8) #cv2.getGaussianKernel(5, 0)
    
    dilation = cv2.dilate(grad,kernel,iterations = 5)
    dilation = dilation/np.max(dilation)
    plt.imshow(dilation)

    img_color_mask = get_img.copy()
    img_color_mask[:,:,0] = dilation*0.5 + im[:,:,0]*0.5
    img_color_mask[:,:,1] = dilation*0.5 + im[:,:,1]*0.5
    img_color_mask[:,:,2] = dilation*0.5 + im[:,:,2]*0.5
    img_color_mask = img_color_mask
    img_color_mask = np.clip(img_color_mask, 0, 255)
    plt.imshow(img_color_mask[:,:,::-1])
    plt.show()
    # import plotly.graph_objects as go
    # fig = go.Figure(data=[
    #     go.Surface(z=dilation)])
    # fig.show()
    # import plotly.graph_objects as go
    # fig = go.Figure(data=[
    #     go.Surface(z=added_heatmap)])
    # fig.show()
    # input("Enter a key")
    # 36 grid NMS
    # votes = np.zeros((6,6))
    # for i in range(300):
    #     for j in range(300):
    #         grid_1 = added_heatmap[i:i+60,j:j+60]
    #         plt.imshow(grid_1)
    #         plt.show()
    #         i+=60
    #         j+=60
    