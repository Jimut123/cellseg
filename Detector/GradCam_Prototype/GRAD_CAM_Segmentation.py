
import tensorflow as tf
from tensorflow import keras
import keras
import keras.utils
from keras import utils as np_utils
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model


# Display

from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
from utils import get_img_array, make_gradcam_heatmap, get_jet_img
import cv2
import glob

###########################################
#  LOAD MODEL

H, W, C = 360, 360, 3
N_LABELS = 8


input_layer = tf.keras.Input(shape=(H, W, C))
x = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(2, 2), name="conv_32")(input_layer)
x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool1")(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(2, 2), name="conv_64")(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool2")(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(2, 2), name="conv_64_2")(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool3")(x)
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dense(512, activation='relu', name="dense_512")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout_1")(x)
x = tf.keras.layers.Dense(512, activation='relu', name="dense_256")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout_2")(x)
x = tf.keras.layers.Dense(128, activation='relu', name="dense_64")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout_3")(x)
x = tf.keras.layers.Dense(N_LABELS, activation='softmax', name="output_layer")(x)
#x = tf.keras.layers.Reshape((1, N_LABELS))(x)
model = tf.keras.models.Model(inputs=input_layer, outputs=x)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics= ['accuracy'])
model.summary()

model = keras.models.load_model('classification_blood.h5')

print("MODEL LOADED!")

###########################################

index = {'platelet': 0, 'eosinophil': 1, 'lymphocyte': 2, 'monocyte': 3, 'basophil': 4, 'ig': 5, 'erythroblast': 6, 'neutrophil': 7}
rev_index = {0: 'platelet', 1: 'eosinophil', 2: 'lymphocyte', 3: 'monocyte', 4: 'basophil', 5: 'ig', 6: 'erythroblast', 7: 'neutrophil'}


all_files_samples = glob.glob('samples/*.jpg')
img_size = (300,300)



for img_file in all_files_samples:

    all_img = []
    all_heatmap = []
    all_superimposed_img = []
    all_heatmap_ = []

    img_save_name, num = str(str(img_file.split('/')[1]).split('.')[0]).split('_')
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

    last_conv_layer_name = "max_pool3" #"dense_2" 
    classifier_layer_names = [
        "flatten",
        "dense_512",
        "dropout_1",
        "dense_256",
        "dropout_2",
        "dense_64",
        "dropout_3",
        "output_layer",
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
        "max_pool3",
        "flatten",
        "dense_512",
        "dropout_1",
        "dense_256",
        "dropout_2",
        "dense_64",
        "dropout_3",
        "output_layer",
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
    

    last_conv_layer_name = "conv_64" 
    classifier_layer_names = [
        "max_pool2",
        "conv_64_2",
        "max_pool3",
        "flatten",
        "dense_512",
        "dropout_1",
        "dense_256",
        "dropout_2",
        "dense_64",
        "dropout_3",
        "output_layer",
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
    
    

    last_conv_layer_name = "conv_32"
    classifier_layer_names = [
        "max_pool1",
        "conv_64",
        "max_pool2",
        "conv_64_2",
        "max_pool3",
        "flatten",
        "dense_512",
        "dropout_1",
        "dense_256",
        "dropout_2",
        "dense_64",
        "dropout_3",
        "output_layer",
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
    for img, heatmap, heatmap_, superimposed_img in zip(all_img, all_heatmap, all_heatmap_, all_superimposed_img):
        count += 1
        ax1 = fig.add_subplot(4,5,count)
        ax1.imshow(img)
        count += 1
        ax2 = fig.add_subplot(4,5,count)
        ax2.imshow(heatmap)
        count += 1
        ax3 = fig.add_subplot(4,5,count)
        ax3.imshow(heatmap_)
        count += 1
        # heatmap_gray = np.array(heatmap_)
        # heatmap_gray = cv2.cvtColor(np.array(heatmap_), cv2.COLOR_BGR2GRAY)
        # print("*"*50,np.array(heatmap_gray).shape)
        ret, thres = cv2.threshold(heatmap_,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        all_thres.append(thres)
        ax3 = fig.add_subplot(4,5,count)
        ax3.imshow(thres)
        count += 1
        ax4 = fig.add_subplot(4,5,count)
        ax4.imshow(superimposed_img)
    
    print("count = ",count)
    added_heatmap = np.zeros((360,360))
    for thres_ in all_thres:
        added_heatmap += thres_
    plt.show()
    plt.imshow(added_heatmap) 
    plt.show()
    maximum = np.amax(added_heatmap)
    print("maximum = ",maximum)
    coord = np.where(added_heatmap == maximum)
    x, y = coord[0][0], coord[1][0]
    print("Coord = ",x,y)
    cv2.rectangle(img,(x-70,y-70),(x+100,y+100),(0,255,0),2)
    plt.imshow(img)
    if predicted_name == img_save_name:
        plt.title("Prediction = {}".format(predicted_name),fontsize=20).set_color('green')
    else:
        plt.title("Prediction = {}".format(predicted_name),fontsize=20).set_color('red')
    plt.xlabel('true: {}'.format(img_save_name),fontsize=20)
    plt.show()
    # 36 grid NMS
    # votes = np.zeros((6,6))
    # for i in range(300):
    #     for j in range(300):
    #         grid_1 = added_heatmap[i:i+60,j:j+60]
    #         plt.imshow(grid_1)
    #         plt.show()
    #         i+=60
    #         j+=60
    

    
    

