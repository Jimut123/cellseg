import numpy as np

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

count_dummy = 0

history_da = {}
history_da['source_digits_loss'] = {}
history_da['source_accuracy'] = {}
history_da['source_domain_loss'] = {}
history_da['target_domain_loss'] = {}




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

    for i, data in enumerate(zip(source_train_generator, target_train_generator)):
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

    print("Source digits loss = {}, Source Accuracy = {}, Source domain loss = {}, Target domain loss = {}".format(
        epoch_source_digits.result(), epoch_accuracy.result(),
        epoch_source_domains.result(), epoch_target_domains.result()))
    history_da['source_digits_loss'][count_dummy] = float(epoch_source_digits.result())
    history_da['source_accuracy'][count_dummy] = float(epoch_accuracy.result())
    history_da['source_domain_loss'][count_dummy] = float(epoch_source_domains.result())
    history_da['target_domain_loss'][count_dummy] = float(epoch_target_domains.result())
    count_dummy += 1


for epoch in range(epochs):
    print("Epoch: {}".format(epoch), end=" ")
    loss_record = train_epoch(source_train_generator, target_train_generator)


# download the model in computer for later use
model.save('DA_MNIST_to_MNIST_m.h5')

import pandas as pd
import json
hist_df = pd.DataFrame(history_da) 
hist_json_file = 'history_da_150e.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)










