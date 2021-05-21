import numpy as np
# creating MNIST-M dataset
import tarfile
import os
import cv2
import numpy as np
import skimage
import skimage.io
import urllib.request
import tensorflow as tf
import h5py

# to use the full dataset
USE_SUBSET = False

def get_subset(x, y):
    if not USE_SUBSET:
        return x, y

    subset_index = 10000
    np.random.seed(1)
    indexes = np.random.permutation(len(x))[:subset_index]
    x, y = x[indexes], y[indexes]

    return x, y


from tensorflow.keras.datasets import mnist
from skimage.color import gray2rgb
from skimage.transform import resize
from sklearn.model_selection import train_test_split

(x_source_train, y_source_train), (x_source_test, y_source_test) = mnist.load_data()
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = tf.keras.datasets.mnist.load_data()


def process_mnist(x):
    x = np.moveaxis(x, 0, -1)
    x = resize(x, (32, 32), anti_aliasing=True, mode='constant')
    x = np.moveaxis(x, -1, 0)
    return gray2rgb(x).astype("float32")

x_source_train = process_mnist(x_source_train)
x_source_test = process_mnist(x_source_test)

x_source_train, y_source_train = get_subset(x_source_train, y_source_train)
#x_source_test, y_source_test = get_subset(x_source_test, y_source_test)

x_source_train, x_source_val, y_source_train, y_source_val = train_test_split(
    x_source_train, y_source_train,
    test_size=int(0.1 * len(x_source_train))
)

x_source_train.shape, x_source_val.shape, x_source_test.shape


#Load MNIST-M [Target]
MNIST_M_PATH = './Datasets/MNIST_M/mnistm.h5'

with h5py.File(MNIST_M_PATH, 'r') as mnist_m:
    mnist_m_train_x, mnist_m_test_x = mnist_m['train']['X'][()], mnist_m['test']['X'][()]


mnist_m_train_x, mnist_m_test_x = mnist_m_train_x / 255.0, mnist_m_test_x / 255.0
mnist_m_train_x, mnist_m_test_x = mnist_m_train_x.astype('float32'), mnist_m_test_x.astype('float32')

mnist_m_train_y, mnist_m_test_y = mnist_train_y, mnist_test_y

# assert(mnist_m_train_x.shape == (60000,32,32,3))
# assert(mnist_m_test_x.shape == (10000,32,32,3))
# assert(mnist_m_train_y.shape == (60000,10))
# assert(mnist_m_test_y.shape == (10000,10))

# import pickle as pkl

# with open("keras_mnistm.pkl", "rb") as f:
#     mnist_m = pkl.load(f)

(x_target_train, y_target_train), (x_target_test, y_target_test) = (mnist_m_train_x, mnist_m_train_y), (mnist_m_test_x, mnist_m_test_y) 


# x_target_train, y_target_train = get_subset(mnist_m["x_train"], mnist_m["y_train"])
# x_target_test, y_target_test = mnist_m["x_test"], mnist_m["y_test"]

# x_target_train = resize(x_target_train, (x_target_train.shape[0], 32, 32, 3), anti_aliasing=True, mode='edge').astype("float32")
# x_target_test = resize(x_target_test, (x_target_test.shape[0], 32, 32, 3), anti_aliasing=True, mode='edge').astype("float32")

x_target_train.shape, x_target_test.shape

from tensorflow.keras.layers import MaxPool2D, Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

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
    
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 5, padding='same', activation='relu', name='conv2d_1')(inputs)
    x = MaxPool2D(pool_size=2, strides=2, name='max_pooling2d_1')(x)
    x = Conv2D(48, 5, padding='same', activation='relu', name='conv2d_2')(x)
    x = MaxPool2D(pool_size=2, strides=2, name='max_pooling2d_2')(x)
    features = Flatten(name='flatten_1')(x)
    x = Dense(100, activation='relu', name='dense_digits_1')(features)
    x = Dense(100, activation='relu', name='dense_digits_2')(x)
    digits_classifier = Dense(10, activation="softmax", name="digits_classifier")(x)

    domain_branch = Dense(100, activation="relu", name="dense_domain")(GradReverse()(features))
    domain_classifier = Dense(1, activation="sigmoid", name="domain_classifier")(domain_branch)

    return Model(inputs=inputs, outputs=[digits_classifier, domain_classifier])

model = get_adaptable_network()
model.summary()


batch_size = 128
epochs = 10

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

optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

cce = SparseCategoricalCrossentropy()
bce = BinaryCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=[cce, bce],
    metrics=["accuracy", "accuracy"]
)

from tensorflow import keras
model = keras.models.load_model('DA_MNIST_to_MNIST_m.h5',custom_objects={'GradReverse':GradReverse})

with open('history_da_150e.json', 'r') as f:
    history_da = json.load(f)
print (len(history_da))

for item in history_da:
    print(item)


source_digits_loss_list = []
source_accuracy_list = []
source_domain_loss_list = []
target_domain_loss_list = []
dummy_count = 0
for sd_l, al, sdom_l, td_list in zip(history_da['source_digits_loss'],  history_da['source_accuracy'], history_da['source_domain_loss'], history_da['target_domain_loss']):
    source_digits_loss_list.append(history_da['source_digits_loss'][str(dummy_count)])
    source_accuracy_list.append(history_da['source_accuracy'][str(dummy_count)])
    source_domain_loss_list.append(history_da['source_domain_loss'][str(dummy_count)])
    target_domain_loss_list.append(history_da['target_domain_loss'][str(dummy_count)])
    dummy_count += 1


plt.figure(figsize=(12,6))
plt.title('Domain Adaptation Losses', fontsize=35, fontname = 'DejaVu Serif', fontweight = 500)
plt.plot(source_digits_loss_list,color='green', linestyle='--', dashes=(5, 1),  linewidth=3.0)
plt.plot(source_domain_loss_list,color='blue', linestyle='-', dashes=(5, 1),  linewidth=3.0)
plt.plot(target_domain_loss_list,color='red', linestyle='-.', dashes=(5, 1),  linewidth=3.0)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

lgd = plt.legend(['Source Digits Loss', 'Source Domain Loss', 'Target Domain Loss'],loc="lower right",
          prop={'family':'DejaVu Serif', 'size':20}, bbox_to_anchor=(1.45, 0.68))
plt.savefig('da_plot_losses_history.eps',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('da_plot_losses_history.png',  bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(figsize=(12,6))
plt.title('Domain Adaptation Accuracy', fontsize=35, fontname = 'DejaVu Serif', fontweight = 500)
plt.plot(source_accuracy_list,color='green', linestyle='--', dashes=(5, 1),  linewidth=3.0)
# plt.plot(source_domain_loss_list,color='blue', linestyle='-', dashes=(5, 1),  linewidth=3.0)
# plt.plot(target_domain_loss_list,color='red', linestyle='-.', dashes=(5, 1),  linewidth=3.0)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

lgd = plt.legend(['Source Digits Loss', 'Source Domain Loss', 'Target Domain Loss'],loc="lower right",
          prop={'family':'DejaVu Serif', 'size':20}, bbox_to_anchor=(1.42, 0.86))
plt.savefig('da_plot_acc_history.eps',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('da_plot_acc_history.png',  bbox_extra_artists=(lgd,), bbox_inches='tight')

from tqdm import tqdm

mnist_pred_list = []
mnist_actual_list = []

for image, y_act in tqdm(zip(x_source_test, y_source_test)):
    mnist_pred_list.append(np.argmax(model.predict(image[np.newaxis,...])[0]))
    mnist_actual_list.append(y_act)


from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(mnist_actual_list, mnist_pred_list)
report = classification_report(mnist_actual_list, mnist_pred_list)

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
    plt.savefig('confusion_matrix_mnist_da_100e.png')
    plt.savefig('confusion_matrix_mnist_da_100e.eps')
    plt.show()

cm_analysis(mnist_actual_list, mnist_pred_list, [i for i in range(10)] , ymap=None, figsize=(10,10))


with open('report_mnist_da_100e.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(report)
    #sys.stdout = original_stdout # Reset the standard output to its original value

from tqdm import tqdm

mnist_m_pred_list = []
mnist_m_actual_list = []
print(len(x_target_test))
for image, y_act in tqdm(zip(x_target_test, y_target_test)):
    mnist_m_pred_list.append(np.argmax(model.predict(image[np.newaxis,...])[0]))
    mnist_m_actual_list.append(y_act)


from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(mnist_m_actual_list, mnist_m_pred_list)
report = classification_report(mnist_m_actual_list, mnist_m_pred_list)

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
    plt.savefig('confusion_matrix_mnist_m_da_100e.png')
    plt.savefig('confusion_matrix_mnist_m_da_100e.eps')
    plt.show()

cm_analysis(mnist_m_actual_list, mnist_m_pred_list, [i for i in range(10)] , ymap=None, figsize=(10,10))


with open('report_mnist_da_100e.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(report)
    #sys.stdout = original_stdout # Reset the standard output to its original value



import math
n = 36
random_indices = np.random.permutation(n)
n_cols = 6
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    img_ = x_target_test[img_idx].copy()
    ax.imshow(img_)
    actual_pred = ""
    true_val = ""
    actual_pred = np.argmax(model.predict(img_[np.newaxis,...])[0])
    true_val = y_target_test[img_idx]
    if actual_pred == true_val:
        ax.set_title("Prediction = {}".format(actual_pred),fontsize=12).set_color('green')
    else:
        ax.set_title("Prediction = {}".format(actual_pred),fontsize=12).set_color('red')
    ax.set_xlabel('true: {}'.format(true_val),fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig('prediction_DA_MNIST_M.png')
plt.savefig('prediction_DA_MNIST_M.eps')



%matplotlib inline
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import glob
import numpy as np
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

images = []
labels = []

for image, y_act in tqdm(zip(x_source_test, y_source_test)):
    image = image.flatten()
    images.append(image)
    labels.append(y_act)


label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

label_ids = np.array([label_to_id_dict[x] for x in labels])

images_scaled = StandardScaler().fit_transform(images)

pca = PCA(n_components=180)
pca_result = pca.fit_transform(images_scaled)

tsne = TSNE(n_components=2, perplexity=40.0)

tsne_result = tsne.fit_transform(pca_result)

tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

def visualize_scatter_with_images(X_2d_data, images, figsize=(30,30), image_zoom=1, name="plot.png"):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.savefig(name)
    # plt.show()

def visualize_scatter(data_2d, label_ids, figsize=(20,20), name="plot.png"):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.savefig(name)
    


visualize_scatter(tsne_result_scaled, label_ids, name="MNIST_PCA_180_perplexity_40.eps")

visualize_scatter_with_images(tsne_result_scaled, images = [np.reshape(i, (32,32,3)) for i in images], image_zoom=0.7, name="TSNE_MNIST_images_PCA_180_perplexity_40.eps")


tsne = TSNE(n_components=3)
tsne_result = tsne.fit_transform(pca_result)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111,projection='3d')

plt.grid()
    
nb_classes = len(np.unique(label_ids))
    
for label_id in np.unique(label_ids):
    ax.scatter(tsne_result_scaled[np.where(label_ids == label_id), 0],
                tsne_result_scaled[np.where(label_ids == label_id), 1],
                tsne_result_scaled[np.where(label_ids == label_id), 2],
                alpha=0.8,
                color= plt.cm.Set1(label_id / float(nb_classes)),
                marker='o',
                label=id_to_label_dict[label_id])
ax.legend(loc='best')
ax.view_init(25, 45)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
plt.savefig("MNIST_PCA_180_perplexity_40_3D.eps")



anim = animation.FuncAnimation(fig, lambda frame_number: ax.view_init(30, 4 * frame_number), interval=75, frames=224)

anim.save('t-SNE_MNIST_3D.mp4', fps=10)






# =====================================================================

model.layers
layer_outputs = [layer.output for layer in model.layers]
layer_outputs[len(layer_outputs)-3]


feature_map_model = Model(inputs=model.input, outputs=layer_outputs[len(layer_outputs)-3])


images = []
labels = []

for image, y_act in tqdm(zip(x_source_test, y_source_test)):
    image = image.flatten()
    images.append(image)
    labels.append(y_act)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])



all_mnist_vect = []
for x_test_images in tqdm(x_source_test):
    vect_ = feature_map_model.predict(x_test_images[np.newaxis,...])[0]
    #print(vect_)
    # print(feature_map_model.predict(x_test_images[np.newaxis,...])[0])
    # print(feature_map_model.predict(x_test_images[np.newaxis,...])[0].shape)
    all_mnist_vect.append(np.array(vect_))
all_mnist_vect = np.array(all_mnist_vect)

print(len(all_mnist_vect))
print(all_mnist_vect.shape)

def visualize_scatter_with_images(X_2d_data, images, figsize=(30,30), image_zoom=1, name="plot.png"):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.savefig(name)
    plt.show()

def visualize_scatter(data_2d, label_ids, figsize=(20,20), name="plot.png"):
    plt.figure(figsize=figsize)
    plt.grid()
    # print(data_2d.shape)
    nb_classes = len(np.unique(label_ids))
    # print(nb_classes)
    # print(np.unique(label_ids))
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.savefig(name)


tsne = TSNE(n_components=2, perplexity=40.0)

tsne_result = tsne.fit_transform(all_mnist_vect)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

visualize_scatter(tsne_result_scaled, label_ids, name="MNIST_DA_TSNE_MODEL_100_perplexity_40.eps")




visualize_scatter_with_images(tsne_result_scaled, images = [np.reshape(i, (32,32,3)) for i in images], image_zoom=0.7, name="MNIST_images_DA_TSNE_MODEL_100_perplexity_40.eps")




from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111,projection='3d')

plt.grid()
    
nb_classes = len(np.unique(label_ids))
    
for label_id in np.unique(label_ids):
    ax.scatter(tsne_result_scaled[np.where(label_ids == label_id), 0],
                tsne_result_scaled[np.where(label_ids == label_id), 1],
                tsne_result_scaled[np.where(label_ids == label_id), 2],
                alpha=0.8,
                color= plt.cm.Set1(label_id / float(nb_classes)),
                marker='o',
                label=id_to_label_dict[label_id])
ax.legend(loc='best')
ax.view_init(25, 45)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
plt.savefig("MNIST_DA_TSNE_MODEL_100_perplexity_40_3D.eps")



anim = animation.FuncAnimation(fig, lambda frame_number: ax.view_init(30, 4 * frame_number), interval=75, frames=224)

anim.save('t-SNE_DA_TSNE_MNIST_MODEL_100_perplexity_40_3D.mp4', fps=10)











images = []
labels = []

for image, y_act in tqdm(zip(x_target_test, y_target_test)):
    image = image.flatten()
    images.append(image)
    labels.append(y_act)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])



all_mnist_m_vect = []
for x_target_images in tqdm(x_target_test):
    vect_ = feature_map_model.predict(x_target_images[np.newaxis,...])[0]
    #print(vect_)
    # print(feature_map_model.predict(x_test_images[np.newaxis,...])[0])
    # print(feature_map_model.predict(x_test_images[np.newaxis,...])[0].shape)
    all_mnist_m_vect.append(np.array(vect_))
all_mnist_m_vect = np.array(all_mnist_m_vect)

print(len(all_mnist_m_vect))
print(all_mnist_m_vect.shape)


tsne = TSNE(n_components=2, perplexity=40.0)

tsne_result = tsne.fit_transform(all_mnist_vect)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


visualize_scatter(tsne_result_scaled, label_ids, name="MNIST_M_DA_TSNE_MODEL_100_perplexity_40.eps")


visualize_scatter_with_images(tsne_result_scaled, images = [np.reshape(i, (32,32,3)) for i in images], image_zoom=0.7, name="MNIST_M_images_DA_TSNE_MODEL_100_perplexity_40.eps")




from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111,projection='3d')

plt.grid()
    
nb_classes = len(np.unique(label_ids))
    
for label_id in np.unique(label_ids):
    ax.scatter(tsne_result_scaled[np.where(label_ids == label_id), 0],
                tsne_result_scaled[np.where(label_ids == label_id), 1],
                tsne_result_scaled[np.where(label_ids == label_id), 2],
                alpha=0.8,
                color= plt.cm.Set1(label_id / float(nb_classes)),
                marker='o',
                label=id_to_label_dict[label_id])
ax.legend(loc='best')
ax.view_init(25, 45)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
plt.savefig("MNIST_M_DA_TSNE_MODEL_100_perplexity_40_3D.eps")



anim = animation.FuncAnimation(fig, lambda frame_number: ax.view_init(30, 4 * frame_number), interval=75, frames=224)

anim.save('t-SNE_MNIST_M_DA_TSNE_MODEL_100_perplexity_40_3D.mp4', fps=10)



