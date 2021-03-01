import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import time
import cv2
import os

dir = glob.glob('PBC_dataset_normal_DIB/*')
get_freq = {}
# count = 1
for item in dir:
  freq = len(glob.glob("{}/*".format(item)))
  print(freq)
  item_name  = item.split('/')[1]
  get_freq[item_name] = freq
  #get_freq[count] = freq
  #count += 1
  #get_freq.append(freq)


short_index = {}
total_img_names = []
short_labels = []
for item in dir:
  img_names = glob.glob("{}/*".format(item))[:5]
  short_name = str(img_names[0].split('.')[0]).split('/')[2].split('_')[0]
  short_index[short_name] = img_names[0].split('/')[1]
  short_labels.append(short_name)
  total_img_names.append(img_names)
print(total_img_names)
print(len(total_img_names))
print(short_labels)
print(short_index)


short_rev_index = {}
for item in short_index:
  short_rev_index[short_index[item]] = item
print(short_rev_index)

index = {}
rev_index = {}
count = 0
for item in get_freq:
  index[item] = count
  rev_index[count] = item
  count += 1 
print(index)
print(rev_index)

def parse_filepath(filepath):
    try:
        #path, filename = os.path.split(filepath)
        label = filepath.split('/')[1]
        #filename, ext = os.path.splitext(filename)
        #label, _ = filename.split("_")
        return label
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None

DATA_DIR = 'PBC_dataset_normal_DIB'  # 302410 images. validate accuracy: 98.8%
H, W, C = 360, 360, 3
N_LABELS = len(index)
D = 1

files = glob.glob("{}/*/*.jpg".format(DATA_DIR))
print("Total files = ",len(files))

# create a pandas data frame of images, age, gender and race
attributes = list(map(parse_filepath, files))

df = pd.DataFrame(attributes)
df['file'] = files
df.columns = ['label', 'file']
df = df.dropna()
df.head()

p = np.random.permutation(len(df))
train_up_to = int(len(df) * 0.95)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.95)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

print('train count: %s, valid count: %s, test count: %s' % (
    len(train_idx), len(valid_idx), len(test_idx)))


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

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

from tensorflow.keras.utils import to_categorical
from PIL import Image

def get_data_generator(df, indices, for_training, batch_size=16):
    images, labels = [], []
    while True:
        #print("indices = ",indices)    
        #print("len indices = ",len(indices))
        for i in indices:

            r = df.iloc[i]
            #print(" r = ", r, " i = ",i)
            file, label = r['file'], r['label']
            #print("file, label = ",file, label)
            im = Image.open(file)
            im = im.resize((360, 360))
            im = np.array(im) / 255.0
            #print(im.shape)
            images.append(im)
            #print(np.asarray([to_categorical(index[label], N_LABELS)]))
            #print(np.asarray([to_categorical(index[label], N_LABELS)]).shape)
            labels.append(to_categorical(index[label], N_LABELS))
            if len(images) >= batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
# batch_size = 100
# valid_batch_size = 32
batch_size = 40
valid_batch_size = 40
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]
# for storing logs into tensorboard
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=150,
                    callbacks=[tensorboard_callback,callbacks],
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)

import pandas as pd
hist_df = pd.DataFrame(history.history) 
hist_json_file = 'history_150e.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# download the model in computer for later use
model.save('classification_blood_150epochs.h5')



