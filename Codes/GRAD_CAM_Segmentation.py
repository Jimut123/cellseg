
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
from GRAD_CAM import get_img_array, make_gradcam_heatmap
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




all_files_samples = glob.glob('samples/*.jpg')
img_size = (300,300)

for img_file in all_files_samples:

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

    # Display heatmap
    # plt.matshow(heatmap)
    # plt.show()

    # We load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)

    img = get_img

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    save_path = "save/{}_{}.jpg".format(img_save_name, num)
    superimposed_img.save(save_path)

    # Display Grad CAM
    #display(Image(save_path))
    fig = plt.figure(figsize=(10,30))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(get_img)
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(heatmap)
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(superimposed_img)
    plt.show()

