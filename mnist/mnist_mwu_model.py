import tensorflow as tf
from functools import *
import itertools
import inspect
from datetime import datetime
import re
import os
import os.path
import time
import numpy as np
#from six.moves import xrange
from utils.utils import *
from tf_utils.tf_vars import *
from tf_utils.tf_utils import *
from mwu.mwu import *
from mnist.mnist_utils import *

import operator

def mnist_linear_test(u=50, lr=0.1,max_steps=1200, smoothing=0.01):
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    model, W, b = mnist_linear(x,y,u=50)
    #model, Ws, bs = mnist_conv(x,y,u=50)
    Ws=[W]
    bs = [b]
    ph_dict = {'x': x, 'y': y}
    return make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

def mnist_mwu_model(u=50,u2=50):
    def f(x):
        #print('0:'+str(tf.shape(x)))
        #print('0:'+str(x.get_shape()))
        y1, W1, b1 = mwu_conv_layer(x,[8, 8, 1, 64], u=u, name='1', strides=(2,2), padding = 'SAME')
        #print('1:'+str(tf.shape(y1)))
        #print('1:'+str(y1.get_shape()))
        y2, W2, b2 = mwu_conv_layer(y1,[6, 6, 64, 128], u=u, name='2', strides=(2,2), padding = 'VALID')
        #print('2:'+str(tf.shape(y2)))
        #print('2:'+str(y2.get_shape()))
        y3, W3, b3 = mwu_conv_layer(y2,[5, 5, 128, 128], u=u, name='3', padding = 'VALID')
        #print('3:'+str(tf.shape(y3)))
        #print('3:'+str(y3.get_shape()))
        #dim = tf.reduce_prod(tf.shape(y3)[1:])
        yf = tf.reshape(y3, [-1, 128])
        #http://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
        #yf = tf.contrib.layers.flatten(y3)
        y4, W4, b4 = mwu_linear_layer(yf, 10, name='4', u=u2)
        y = tf.nn.softmax(y4)
        return y
    return f


def make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=0.1,max_steps=3000, smoothing=0.01,batch_size = 100, print_steps = 100, eval_steps = 600, verbosity=1, train_dir = "mwu/"):
    #opter = lambda: tf.train.GradientDescentOptimizer(learning_rate=0.1)
    opter= lambda: MWOptimizer(Ws, bs, learning_rate=lr, smoothing=smoothing)
    evaler = Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test")
    addons = [Train(opter, 
                    batch_size, 
                    train_feed={}, 
                    loss = 'loss', 
                    print_steps=print_steps),
              evaler]
    trainer = Trainer(model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity)
    trainer.init()
    #print("Weights:",trainer.sess.run([W,b]))
    trainer.train()
    #[weights, biases] = trainer.sess.run([W,b])
    #print("Weights:",weights, biases)
    #print(np.sum(weights, axis=0))
    #print(np.max(weights, axis=0))
    trainer.finish()
    try:
        ans = evaler.record[-1][0]
    except IndexError:
        ans = 0
    return ans

def make_mwu_trainer_gs(model, ph_dict, Ws, bs, train_data, test_data, lr=0.1,max_steps=3000, smoothing=0.01,batch_size = 100, print_steps = 100, eval_steps = 600, verbosity=1, train_dir = "mwu/"):
    #opter = lambda: tf.train.GradientDescentOptimizer(learning_rate=0.1)
    opter= lambda gs: MWOptimizer(Ws, bs, learning_rate=lr/tf.ceil(tf.cast(gs+1, tf.float32)/500), smoothing=smoothing)
    # /tf.ceil(tf.cast(gs+1, tf.float32)/1000)
    evaler = Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test")
    addons = [GlobalStep(),
              Train(opter, 
                    batch_size, 
                    train_feed={}, 
                    loss = 'loss', 
                    print_steps=print_steps),
              evaler]
    trainer = Trainer(model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity)
    trainer.init()
    #print("Weights:",trainer.sess.run([W,b]))
    trainer.train()
    #[weights, biases] = trainer.sess.run([W,b])
    #print("Weights:",weights, biases)
    #print(np.sum(weights, axis=0))
    #print(np.max(weights, axis=0))
    trainer.finish()
    try:
        ans = evaler.record[-1][0]
    except IndexError:
        ans = 0
    return ans

def mnist_conv_test(u=50, u2=50, lr=0.1,max_steps=3000, smoothing=0.01):
    print(u,u2,lr)
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #model, W, b = mnist_linear(x,y,u=u)
    model, Ws, bs = mnist_conv(x,y,u=u,u2=u2)
    ph_dict = {'x': x, 'y': y}
    return make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

def mnist_conv_test_gs(u=50, u2=50, lr=0.1,max_steps=3000, smoothing=0.01):
    print(u,u2,lr)
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #model, W, b = mnist_linear(x,y,u=u)
    model, Ws, bs = mnist_conv(x,y,u=u,u2=u2)
    ph_dict = {'x': x, 'y': y}
    return make_mwu_trainer_gs(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

def mnist_conv_test2(u=50, u2=50, lr=0.1,max_steps=3000, smoothing=0.01):
    print(u,u2,lr)
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    u1 = tf.placeholder(tf.float32, shape=1)
    u2 = tf.placeholder(tf.float32, shape=1)
    u3 = tf.placeholder(tf.float32, shape=1)
    u4 = tf.placeholder(tf.float32, shape=1)
    #model, W, b = mnist_linear(x,y,u=u)
    model, Ws, bs = mnist_conv(x,y,u=u,u2=u2)
    ph_dict = {'x': x, 'y': y}
    make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

