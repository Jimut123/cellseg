"""
Jimut Bahan Pal
05-06-21 
@ 29/09/2023 - Modified for R2 in ESWA
"""

##################################################
import os
# set the visible devices to 7 here
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
##################################################

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



dir = glob.glob('classification_data/*')
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
    

DATA_DIR = 'classification_data'  # 302410 images. validate accuracy: 98.8%
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


DATA_DIR = 'PBC_8_DA' 
H, W, C = 200, 200, 3
N_LABELS = len(index)
D = 1

files = glob.glob("{}/*/*.jpg".format(DATA_DIR))
print("Total files = ",len(files))



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

from tensorflow.keras.applications import Xception
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
    frozen = Xception(weights="imagenet", input_shape=input_shape, include_top=False)
    frozen.summary()

    trainable = frozen.output
    trainable = Dense(512, activation="relu")(trainable)
    features = Flatten(name='flatten_1')(trainable)
    x = Dense(128, activation='relu', name='dense_digits_1')(features)
    x = Dense(32, activation='relu', name='dense_digits_2')(x)
    digits_classifier = Dense(10, activation="softmax", name="digits_classifier")(x)

    domain_branch = Dense(128, activation="relu", name="dense_domain")(GradReverse()(features))
    domain_classifier = Dense(1, activation="sigmoid", name="domain_classifier")(domain_branch)
    
    return Model(inputs=frozen.input, outputs=[digits_classifier, domain_classifier])

model = get_adaptable_network()
model.summary()

######################################
batch_size = 16
epochs = 10
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





############################################################################
from keras.utils.layer_utils import count_params

# Initialize a counter for convolutional layers
conv_layer_count = 0

# Iterate through the layers of the model
for layer in model.layers:
    # Check if the layer is a convolutional layer
    if 'Conv2D' in str(layer.__class__):
        conv_layer_count += 1

# Print the total number of convolutional layers in the model
print("Total Convolutional Layers:", conv_layer_count)



# Initialize a counter for linear layers
linear_layer_count = 0

# Iterate through the layers of the model
for layer in model.layers:
    # Check if the layer is a dense layer
    if 'Dense' in str(layer.__class__):
        linear_layer_count += 1

# Print the total number of linear layers in the model
print("Total Linear (Dense) Layers:", linear_layer_count)


total_params = model.count_params()
print("Total Parameters:", total_params)



trainable_count = count_params(model.trainable_weights)
non_trainable_count = count_params(model.non_trainable_weights)




def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

get_memory_usage =  get_model_memory_usage(batch_size, model)


with open("COMPLEXITY_DUMP.txt", 'a') as f:
    f.write("Total Convolutional Layers: "+str(conv_layer_count)+'\n')
    f.write("Total Linear (Dense) Layers: "+str(linear_layer_count)+'\n')
    f.write("Total Parameters: "+str(total_params)+'\n')
    f.write("Trainable params: "+str(trainable_count)+'\n')
    f.write("Non-trainable params: "+str(non_trainable_count)+'\n')
    # f.write("FLOPs (total float operations): "+str(flops.total_float_ops)+"\n")
    f.write("Total memory usage: "+str(get_memory_usage)+"\n")
    f.close()
    
# class TimeHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.times = []

#     def on_epoch_begin(self, batch, logs={}):
#         self.epoch_time_start = time.time()

#     def on_epoch_end(self, batch, logs={}):
#         with open("TRAIN_EPOCH_TIME.txt", 'a') as f:
#             f.write(str(time.time() - self.epoch_time_start)+'\n')
#             f.close()
#         # self.times.append(time.time() - self.epoch_time_start)
        
        
# # Train the model and record training times for each epoch
# epochs = 1
# epoch_times = []

# for epoch in range(epochs):
#     start_time = time.time()  # Record the start time
#     end_time = time.time()  # Record the end time
    
#     epoch_time = end_time - start_time  # Calculate the epoch training time
#     epoch_times.append(epoch_time)  # Store the epoch training time

# time_callback = TimeHistory()

# history = history = model.fit(train_gen,
#                 steps_per_epoch=len(train_idx)//batch_size,
#                 epochs=10,
#                 callbacks=[tensorboard_callback,callbacks,time_callback],
#                 validation_data=valid_gen,
#                 validation_steps=len(valid_idx)//valid_batch_size)


# print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")


############################################################################





for epoch in range(epochs):
    start_time = time.time()  # Record the start time
    print("Epoch: {}/{}".format(epoch,epochs), end=" ")
    loss_record = train_epoch(source_train_generator, target_train_generator)
    end_time = time.time()  # Record the end time
    
    epoch_time = end_time - start_time  # Calculate the epoch training time
    with open("TRAIN_EPOCH_TIME.txt", 'a') as f:
        f.write(str(epoch_time)+'\n')
        f.close()
    
    ########################################################


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
    ########################################################
    start_time = time.time()  # Record the start time
    
    smear_pred_list.append(np.argmax(model.predict(image[np.newaxis,...])[0]))
    smear_actual_list.append(y_act)
    
    end_time = time.time()  # Record the end time
    epoch_time = end_time - start_time  # Calculate the epoch training time
    with open("INFERENCE_TIME.txt", 'a') as f:
        f.write(str(epoch_time)+'\n')
        f.close()
    
    ########################################################


########################################################

# calculate the size of .h5 and store it in the statistics

all_h5_files = glob.glob("*.h5")

file_size = os.stat(all_h5_files[0])
file_size_mb = file_size.st_size/(1024*1024)

with open("COMPLEXITY_DUMP.txt", 'a') as f:
    f.write("File size in MB: "+str(file_size_mb)+"\n")
    f.close()
    

session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

with graph.as_default():
    with session.as_default():
        model = keras.applications.mobilenet.MobileNet(
                alpha=1, weights=None, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)))

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        # Optional: save printed results to file
        # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
        # opts['output'] = 'file:outfile={}'.format(flops_log_path)

        # We use the Keras session graph in the call to the profiler.
        flops = tf.compat.v1.profiler.profile(graph=graph,
                                                run_meta=run_meta, cmd='op', options=opts)

tf.compat.v1.reset_default_graph()


with open("COMPLEXITY_DUMP.txt", 'a') as f:
    f.write("FLOPs (total float operations): "+str(flops.total_float_ops)+"\n")
    f.close()
    
########################################################