from __future__ import print_function
from keras import *
from keras.datasets import cifar10
from tf_utils.keras_utils import *
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np

cifar_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#normalize means turn into [-1,1]
def get_data_cifar(normalize=False):
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    if normalize:
        x_train = 2*x_train-1
        x_test = 2*x_test-1
    print(x_train[0])
    y_train = np.ndarray.flatten(y_train)
    y_test = np.ndarray.flatten(y_test)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train,y_train,x_test,y_test

def get_bf_cifar(augmentation=True, batch_size=128, normalize=False):
    X_train, Y_train, X_test, Y_test = get_data_cifar(normalize=normalize)
    return bfs_from_datagen_f(X_train, Y_train, X_test, Y_test, cifar_datagen, batch_size, gen_for_test = False) if augmentation else (make_batch_feeder({'x': X_train, 'y':Y_train}), make_batch_feeder({'x': X_test, 'y':Y_test}))

def make_cifar_phs():
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.int64, shape=[None]) #,10 #int64 because argmax returns int64 by default
    return x, y

def make_with_cifar_inputs(m):
    x, y = make_cifar_phs()
    d = m(x,y)
    ph_dict = {'x':x, 'y':y}
    return d, ph_dict

def start_keras_session(learning_phase=0):
    K.set_learning_phase(learning_phase)
    sess = tf.Session()
    K.set_session(sess)
    return sess

