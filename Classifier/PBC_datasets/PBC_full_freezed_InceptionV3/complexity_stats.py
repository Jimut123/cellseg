# PBC full - InceptionV3 freezed


##################################################
import os
# set the visible devices to 7 here
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
##################################################

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




from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, GlobalAveragePooling2D



frozen = InceptionV3(weights="imagenet", input_shape=(360,360,3), include_top=False)
frozen.summary()

trainable = frozen.output
trainable = GlobalAveragePooling2D()(trainable)
#print(trainable.shape)
trainable = Dense(128, activation="relu")(trainable)
trainable = Dense(32, activation="relu")(trainable)
trainable = Dense(N_LABELS, activation="softmax")(trainable)
model = Model(inputs=frozen.input, outputs=trainable)
model.summary()

model.layers
for layer in model.layers[:-4]:
    layer.trainable = False

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
    
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        with open("TRAIN_EPOCH_TIME.txt", 'a') as f:
            f.write(str(time.time() - self.epoch_time_start)+'\n')
            f.close()
        # self.times.append(time.time() - self.epoch_time_start)
        
        
# # Train the model and record training times for each epoch
# epochs = 1
# epoch_times = []

# for epoch in range(epochs):
#     start_time = time.time()  # Record the start time
#     end_time = time.time()  # Record the end time
    
#     epoch_time = end_time - start_time  # Calculate the epoch training time
#     epoch_times.append(epoch_time)  # Store the epoch training time

time_callback = TimeHistory()

history = model.fit(train_gen,
                steps_per_epoch=len(train_idx)//batch_size,
                epochs=10,
                callbacks=[tensorboard_callback,callbacks,time_callback],
                validation_data=valid_gen,
                validation_steps=len(valid_idx)//valid_batch_size)


# print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")


############################################################################




import pandas as pd
hist_df = pd.DataFrame(history.history) 
hist_json_file = 'history_pbc_8_full_inception_v3_100e.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# download the model in computer for later use
model.save('classification_pbc_8_full_inception_v3_100e.h5')

from tensorflow import keras
model = keras.models.load_model('classification_pbc_8_full_inception_v3_100e.h5')



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
    
    ########################################################
    start_time = time.time()  # Record the start time
    
    y_pred = model.predict(im[np.newaxis, ...])
    
    end_time = time.time()  # Record the end time
    epoch_time = end_time - start_time  # Calculate the epoch training time
    with open("INFERENCE_TIME.txt", 'a') as f:
        f.write(str(epoch_time)+'\n')
        f.close()
    
    ########################################################
    y_pred_list.append(int(tf.math.argmax(y_pred, axis=-1)))
    y_test_list.append(index[label])
    


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