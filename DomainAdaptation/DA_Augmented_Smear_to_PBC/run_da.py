"""
Jimut Bahan Pal
05-06-21 
"""

# Domain Adaptation - Source Domain : Smear Slides 8, and Target Domain :  PBC 8 DA

from tensorflow.keras.utils import to_categorical
from PIL import Image

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



dir = glob.glob('classification_data_da_aug/*')
get_freq = {}
# count = 1
for item in dir:
  freq = len(glob.glob("{}/*".format(item)))
  print(freq)
  item_name  = item.split('/')[1]
  get_freq[item_name] = freq

short_index = []

for item in dir:
  name = item.split('/')[1]
  if ' ' in name:
      print(name)
      name = name.split(' ')[0] +"_"+ name.split(' ')[1]
  short_name = name
  short_index.append(short_name)

print(short_index)

count = 0
short_rev_index = {}
for item in short_index:
  short_rev_index[item] = count
  count += 1
# print(short_rev_index)

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
        label = filepath.split('/')[1]
        return label
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None
    

DATA_DIR = 'classification_data_da_aug'  # 302410 images. validate accuracy: 98.8%
H, W, C = 200, 200, 3
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


from tensorflow.keras.utils import to_categorical
from PIL import Image
from tqdm import tqdm

def get_data_generator(df, indices):
    images, labels = [], []
    for i in tqdm(indices):

        r = df.iloc[i]
        file, label = r['file'], r['label']
        im = Image.open(file)
        im = im.resize((H, W))
        im = np.array(im) / 255.0
        images.append(im)
        labels.append(index[label])
    return np.array(images), np.array(labels)


x_source_train,  y_source_train = get_data_generator(df, train_idx)
print(x_source_train.shape, y_source_train.shape)

x_source_val,  y_source_val = get_data_generator(df, valid_idx)
print(x_source_val.shape, y_source_val.shape)

x_source_test,  y_source_test = get_data_generator(df, test_idx)
print(x_source_test.shape, y_source_test.shape)


# PBC #####################################################################

dir = glob.glob('PBC_8_DA/*')
get_freq = {}
# count = 1
for item in dir:
  freq = len(glob.glob("{}/*".format(item)))
  print(freq)
  item_name  = item.split('/')[1]
  get_freq[item_name] = freq


DATA_DIR = 'PBC_8_DA'  # 302410 images. validate accuracy: 98.8%
H, W, C = 200, 200, 3
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


x_target_train,  y_target_train = get_data_generator(df, train_idx)
print(x_target_train.shape, y_target_train.shape)

x_target_val,  y_target_val = get_data_generator(df, valid_idx)
print(x_target_val.shape, y_target_val.shape)

x_target_test,  y_target_test = get_data_generator(df, test_idx)
print(x_target_test.shape, y_target_test.shape)



# Code for Model

from tensorflow.keras.layers import MaxPool2D, Conv2D, Dropout, Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
# from keras.models import Model
from tensorflow.keras.optimizers import Adam
#from keras.layers import Dense, Flatten, GlobalAveragePooling2D


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(name="grl")

    def call(self, x):
        return grad_reverse(x)

def get_adaptable_network(input_shape=x_source_train.shape[1:]):

    #inputs = Input(shape=input_shape)
    frozen = VGG16(weights="imagenet", input_shape=input_shape, include_top=False)
    frozen.summary()

    trainable = frozen.output
    trainable = Dense(512, activation="relu")(trainable)
    features = Flatten(name='flatten_1')(trainable)
    x = Dense(512, activation='relu', name='dense_digits_1')(features)
    x = Dense(512, activation='relu', name='dense_digits_2')(x)
    digits_classifier = Dense(10, activation="softmax", name="digits_classifier")(x)

    domain_branch = Dense(512, activation="relu", name="dense_domain")(GradReverse()(features))
    domain_classifier = Dense(1, activation="sigmoid", name="domain_classifier")(domain_branch)
    
    return Model(inputs=frozen.input, outputs=[digits_classifier, domain_classifier])

model = get_adaptable_network()
model.summary()

######################################
batch_size = 10
epochs = 200
######################################


d_source_train = np.ones_like(y_source_train)
d_source_val = np.ones_like(y_source_val)

source_train_generator = tf.data.Dataset.from_tensor_slices(
    (x_source_train, y_source_train, d_source_train)).batch(batch_size)

d_target_train = np.zeros_like(y_target_train)

target_train_generator = tf.data.Dataset.from_tensor_slices(
    (x_target_train, d_target_train)
).batch(batch_size)


from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import Mean, Accuracy
import collections

optimizer = Adam(learning_rate=1e-4)

cce = SparseCategoricalCrossentropy()
bce = BinaryCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=[cce, bce],
    metrics=["accuracy", "accuracy"]
)

count_dummy = 0

history_da = {}
history_da['source_image_loss'] = {}
history_da['source_accuracy'] = {}
history_da['source_domain_loss'] = {}
history_da['target_domain_loss'] = {}

###################### Train the Model here

def train_epoch(source_train_generator, target_train_generator):
    global lambda_factor, global_step, history_da, count_dummy

    # Keras provide helpful classes to monitor various metrics:
    epoch_source_digits = tf.keras.metrics.Mean()
    epoch_source_domains = tf.keras.metrics.Mean()
    epoch_target_domains = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Fetch all trainable variables but those used uniquely for the digits classification:
    variables_but_classifier = list(filter(lambda x: "digits" not in x.name, model.trainable_variables))

    loss_record = collections.defaultdict(list)

    for i, data in tqdm(enumerate(zip(source_train_generator, target_train_generator))):
        
        source_data, target_data = data
        # Training digits classifier & domain classifier on source:
        x_source, y_source, d_source = source_data

        with tf.GradientTape() as tape:
            digits_prob, domains_probs = model(x_source)
            digits_loss = cce(y_source, digits_prob)
            domains_loss = bce(d_source, domains_probs)
            source_loss = digits_loss + 0.2 * domains_loss

        gradients = tape.gradient(source_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_source_digits(digits_loss)
        epoch_source_domains(domains_loss)
        epoch_accuracy(y_source, digits_prob)

        # Training domain classifier on target:
        x_target, d_target = target_data
        with tf.GradientTape() as tape:
            _, domains_probs = model(x_target)
            target_loss = 0.2 * bce(d_target, domains_probs)

        gradients = tape.gradient(target_loss, variables_but_classifier)
        optimizer.apply_gradients(zip(gradients, variables_but_classifier))

        epoch_target_domains(target_loss)
        # print(f"\r ETA of {epoch} epoch: [{i}/{len(source_train_generator)}] ", end="")

    print("\nSource image loss = {}, Source Accuracy = {}, Source domain loss = {}, Target domain loss = {}".format(
        epoch_source_digits.result(), epoch_accuracy.result(),
        epoch_source_domains.result(), epoch_target_domains.result()))
    history_da['source_image_loss'][count_dummy] = float(epoch_source_digits.result())
    history_da['source_accuracy'][count_dummy] = float(epoch_accuracy.result())
    history_da['source_domain_loss'][count_dummy] = float(epoch_source_domains.result())
    history_da['target_domain_loss'][count_dummy] = float(epoch_target_domains.result())
    count_dummy += 1

for epoch in range(epochs):
    print("Epoch: {}/{}".format(epoch,epochs), end=" ")
    loss_record = train_epoch(source_train_generator, target_train_generator)


# download the model in computer for later use
model.save('DA_SMEAR_to_PBC.h5')

from tensorflow import keras
model = keras.models.load_model('DA_SMEAR_to_PBC.h5',custom_objects={'GradReverse':GradReverse})

import pandas as pd
import json
hist_df = pd.DataFrame(history_da) 
hist_json_file = 'history_da_150e.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)


## Create the report and Confusion Matrices

from tqdm import tqdm

smear_pred_list = []
smear_actual_list = []

for image, y_act in tqdm(zip(x_source_test, y_source_test)):
    smear_pred_list.append(np.argmax(model.predict(image[np.newaxis,...])[0]))
    smear_actual_list.append(y_act)


from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(smear_actual_list, smear_pred_list)
report = classification_report(smear_actual_list, smear_pred_list)

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
    plt.savefig('confusion_matrix_smear_da_100e.png',dpi=300, bbox_inches='tight')
    plt.savefig('confusion_matrix_smear_da_100e.eps',dpi=300, bbox_inches='tight')
    # plt.show()

cm_analysis(smear_actual_list, smear_pred_list, [i for i in range(8)] , ymap=rev_index, figsize=(10,10))

with open("report_smear_da_100e.txt", "w") as text_file:
    text_file.write(report)

# with open('report_smear_da_100e.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(report)
#     #sys.stdout = original_stdout # Reset the standard output to its original value


## Create the report and Confusion Matrices

from tqdm import tqdm

pbc_pred_list = []
pbc_actual_list = []
print(len(x_target_test))
for image, y_act in tqdm(zip(x_target_test, y_target_test)):
    pbc_pred_list.append(np.argmax(model.predict(image[np.newaxis,...])[0]))
    pbc_actual_list.append(y_act)


from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(pbc_actual_list, pbc_pred_list)
report = classification_report(pbc_actual_list, pbc_pred_list)

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
    plt.savefig('confusion_matrix_pbc_da_100e.png',dpi=300, bbox_inches='tight')
    plt.savefig('confusion_matrix_pbc_da_100e.eps',dpi=300, bbox_inches='tight')
    # plt.show()

cm_analysis(pbc_actual_list, pbc_pred_list, [i for i in range(8)] , ymap=rev_index, figsize=(10,10))


with open("report_mnist_da_100e.txt", "w") as text_file:
    text_file.write(report)

# with open('report_mnist_da_100e.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(report)
#     #sys.stdout = original_stdout # Reset the standard output to its original value




