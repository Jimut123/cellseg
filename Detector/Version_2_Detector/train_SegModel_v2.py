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
from keras.regularizers import l2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D,\
                                    GlobalMaxPool2D, Dropout, SpatialDropout2D, add, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

def Model_V2_Gradcam(H,W,C):

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

model.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics= ['accuracy'])
model.summary()






from tensorflow.keras.utils import to_categorical
from PIL import Image

def get_data_generator(df, indices, for_training, batch_size=16):
    images, labels = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r['file'], r['label']
            im = Image.open(file)
            im = im.resize((360, 360))
            im = np.array(im) / 255.0
            images.append(im)
            labels.append(to_categorical(index[label], N_LABELS))
            if len(images) >= batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
batch_size = 90
valid_batch_size = 90
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callbacks = [
    #ModelCheckpoint("./model_checkpoint", monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4)
]

# for storing logs into tensorboard
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=5,
                    callbacks=[tensorboard_callback,callbacks],
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)

import pandas as pd
hist_df = pd.DataFrame(history.history) 
hist_json_file = 'history_classification_model_v2_100e.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# download the model in computer for later use
model.save('classification_model_v2_blood_100epochs.h5')


from tensorflow import keras
model = keras.models.load_model('classification_model_v2_blood_100epochs.h5')


test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
res = dict(zip(model.metrics_names, model.evaluate(test_gen, steps=len(test_idx)//128)))
print(res)

with open("Output.txt", "w") as text_file:
    text_file.write(" Test Output : {}".format(res))


test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
x_test, y_test = next(test_gen)
y_pred = model.predict_on_batch(x_test)
y_true = tf.math.argmax(y_test, axis=-1)
y_pred = tf.math.argmax(y_pred, axis=-1)












