
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
x_1 = tf.keras.layers.Conv2D(16, 3, activation='relu', strides=(1, 1), name="conv_32", padding='same')(input_layer)
# x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool1")(x)
x_2 = tf.keras.layers.Conv2D(1, 3, activation='relu', strides=(1, 1), name="conv_64", padding='same')(x_1)
# x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool2")(x)
x_3 = tf.keras.layers.Conv2D(16, 3, activation='relu', strides=(1, 1), name="conv_64_2", padding='same')(concatenate([x_2, x_1]))
# x_4 = tf.keras.layers.Conv2D(16, 3, activation='relu', strides=(1, 1), name="conv_64_21", padding='same')(add([x_3,x_1]))
x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool3")(x_3)
x_4 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="conv_64_3", padding='same')(x)
x_5 = tf.keras.layers.Conv2D(1, 3, activation='relu', strides=(1, 1), name="conv_64_31", padding='same')(x_4)
x_6 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="conv_64_32", padding='same')(concatenate([x_5, x_4]))

x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool4")(x_6)
x_7 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="conv_64_4", padding='same')(x)
x_8 = tf.keras.layers.Conv2D(1, 3, activation='relu', strides=(1, 1), name="conv_64_41", padding='same')(x_7)
x_9 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="conv_64_42", padding='same')(concatenate([x_7, x_8]))
x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool5")(x_9)
x = tf.keras.layers.Conv2D(2, 3, activation='relu', strides=(2, 2), name="conv_64_5")(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool6")(x)
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dropout(0.15, name="dropout_3")(x)
x = tf.keras.layers.Dense(256, activation='relu', name="dense_64")(x)
x = tf.keras.layers.Dense(N_LABELS, activation='softmax', name="output_layer")(x)

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



img_size = (300,300)


from tensorflow import keras
model = keras.models.load_model('classification_model_v2_blood_150epochs.h5')