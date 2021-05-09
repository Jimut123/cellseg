
from tensorflow.keras.utils import to_categorical
from PIL import Image

import tensorflow as tf
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

import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import time
import cv2
import os
from tensorflow.keras.utils import to_categorical
from PIL import Image

import tensorflow as tf
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

import pandas as pd

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import time
import cv2
import os


dir = glob.glob('PBC_dataset_normal_DIB_cropped/*')
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
  print(item)
  img_names = glob.glob("{}/*".format(item))[:5]
  print("img names = ",img_names[:10])
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

DATA_DIR = 'PBC_dataset_normal_DIB_cropped'  # 302410 images. validate accuracy: 98.8%
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
df.tail()

np.random.seed(42)
p = np.random.permutation(len(df))
train_up_to = int(len(df) * 0.80)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.80)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

print('train count: %s, valid count: %s, test count: %s' % (
    len(train_idx), len(valid_idx), len(test_idx)))



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
            #experimental_run_tf_function=False,
            metrics = ['accuracy', AUC(curve="ROC"), Precision(), Recall(), \
            TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()]
            )
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
        # if not for_training:
        #     break


from tensorflow.keras.utils import to_categorical
from PIL import Image

def get_data_generator_custom(df, indices, for_training, batch_size=16):
    labels = []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r['file'], r['label']
            labels.append(to_categorical(index[label], N_LABELS))
            if len(images) >= batch_size:
                yield  np.array(labels)
                labels = []


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
# batch_size = 100
# valid_batch_size = 32
batch_size = 16
valid_batch_size = 16
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss'),
    #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4)
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
hist_json_file = 'history_pbc_8_v2_100e.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# download the model in computer for later use
model.save('classification_pbc_8_v2_100e.h5')

from tensorflow import keras
model = keras.models.load_model('classification_pbc_8_v2_100e.h5')

test_gen = get_data_generator(df, test_idx, for_training=False)
dict(zip(model.metrics_names, model.evaluate(test_gen, steps=len(test_idx))))

from tensorflow.keras.utils import to_categorical
from PIL import Image
from tqdm import tqdm
y_pred_list = []
y_test_list = []

for i in tqdm(test_idx):
    r = df.iloc[i]
    file_, label = r['file'], r['label']
    im = Image.open(file_)
    im = im.resize((360, 360))
    im = np.array(im) / 255.0
    # print(im[np.newaxis, ...].shape)
    y_pred = model.predict(im[np.newaxis, ...])
    y_pred_list.append(int(tf.math.argmax(y_pred, axis=-1)))
    #print(index[label])
    y_test_list.append(index[label])
    # print("This = ",rev_index[int(tf.math.argmax(y_pred, axis=-1))])
    # print(to_categorical(index[label], N_LABELS))
    # print(label)
    

from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(y_test_list, y_pred_list)
report = classification_report(y_test_list, y_pred_list)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='rocket_r')
    #plt.savefig(filename)
    plt.savefig('confusion_matrix.png')
    plt.savefig('confusion_matrix.eps')
    #plt.show()

cm_analysis(y_test_list, y_pred_list, [i for i in rev_index] , ymap=None, figsize=(10,10))


with open('report_pbc_8_v2_100e.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(report)
    #sys.stdout = original_stdout # Reset the standard output to its original value


