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
#from cleverhans.utils_mnist import data_mnist

import operator

def prod(li):
    return reduce(operator.mul, li, 1)

def make_mw_ops(y, ws, eta, grads=None): #include clipping
    # https://www.tensorflow.org/api_docs/python/tf/gradients
    if grads == None:
        grads = tf.gradients(y, ws)
    assign_ops = []
    for (w,g) in zip(ws, grads):
        # https://blog.metaflow.fr/tensorflow-mutating-variables-and-control-flow-2181dd238e62
        w1 = w*(1-eta*g)
        #normalize each column
        w2 = w1 / tf.reduce_sum(w1, range(tf.rank(w1)-1), keep_dims = True)
        w_op = tf.assign(w, w2)
        assign_ops.append(w_op)
    return assign_ops

class MWOptimizer:
    def __init__(self, ws, bs, learning_rate=0.01,smoothing=0):
        self.ws = ws #hand all parameters
        self.bs = bs
        self.grads = None
        self.b_grads = None
        self.eta  = learning_rate
        self.smoothing= smoothing
        #self.global_step= global_step
    def compute_gradients(self, y):
        self.grads = tf.gradients(y,self.ws)
        self.b_grads = tf.gradients(y,self.bs)
        #print("computing gradients:"+str((self.grads, self.b_grads)))
        return (self.grads, self.b_grads)
    def apply_gradients(self, gradients, global_step=None):
        #if self.grads == None:
        #    gradients = self.compute_gradients(y)
        return make_mw_ops_with_biases(self.ws, self.bs, self.eta, gradients=gradients, smoothing=self.smoothing, global_step = global_step)

def make_mw_ops_with_biases(ws, bs, eta, y=None, gradients=None, smoothing=0, global_step = None): #include clipping
    # https://www.tensorflow.org/api_docs/python/tf/gradients
    if gradients==None:
        grads = tf.gradients(y, ws)
        b_grads = tf.gradients(y, bs)
    else:
        (grads, b_grads) = gradients
    #print('making grads: ', gradients)
    w_assign_ops = []
    b_assign_ops = []
    if global_step !=None:
        g_assign = [tf.assign(global_step, global_step+1)]
    else:
        g_assign = []
    for (w,g,b,gb) in zip(ws, grads, bs, b_grads):
        # https://blog.metaflow.fr/tensorflow-mutating-variables-and-control-flow-2181dd238e62
        #g = tf.Print(g,[g],'grad W:',summarize=128)
        w1 = w*tf.exp(-eta*g)
        #print('making grads: ', b, eta, gb)
        b1 = b*tf.exp(-eta*gb)
        ld = int(w1.get_shape()[-1])
        d = prod([int(i) for i in w1.get_shape()[:-1]])
        #normalize each column
        z = tf.reduce_sum(w1, axis = list(range(len(w1.get_shape())-1))) + b1[0:ld] + b[ld:2*ld]
                          #range(tf.rank(w1)-1)) + b1
        w2 = (w1 / z) * (1-smoothing) + (1/(d+2*ld))*smoothing 
        b2 = (b1 / tf.tile(z,[2])) * (1-smoothing) + (1/(d+2*ld))*smoothing
        with tf.control_dependencies(g_assign):
            w_op = tf.assign(w, w2)
            b_op = tf.assign(b,b2)
        w_assign_ops.append(w_op)
        b_assign_ops.append(b_op)
    with tf.control_dependencies(w_assign_ops+b_assign_ops):
        op = tf.no_op()
    return op

def pm(x):
    return tf.concat([x, -x], axis=-1)

#http://stackoverflow.com/questions/29831489/numpy-1-hot-array

def np_onehot(i, n):
    b = np.zeros([n])
    b[i] = 1
    return b
    #l = np.size(a)
    #b = np.zeros([l,n])
    #b[np.arange(l), a] = 1
    #return b


def synthesize_data(bf, batch_size):
    [dim, i, sgn] = bf.args
    xs = 2*np.random.randint(0,2, size = (batch_size, dim)) - 1
    ys = np.asarray([np_onehot(int((sgn * x[i] + 1)/2),2) for x in xs])
    #print("ys shape", ys.shape)
    return {'x': xs, 'y': ys}

def mnist_linear(x,y,dim=784, n=10, labels=False,u=1):
    x1 = tf.reshape(x, (-1, dim))
    pm_x = pm(x1)
    W = get_scope_variable('W', shape=[2*dim, n], initializer = tf.constant_initializer(1/(2*dim+1)))
    b = get_scope_variable('b', shape=[2*n], initializer = tf.constant_initializer(1/(2*dim+1)))
    #tf.ones([2*dim,10])/(2*dim))
    yhat = u*(tf.matmul(pm_x, W) + b[0:n] - b[n:2*n])
    #yhat = tf.nn.softmax(u*(tf.matmul(pm_x, W) + b))
    #print(y.get_shape())
    #print(yhat.get_shape())
    loss = tf.losses.softmax_cross_entropy(y, yhat)
    acc = accuracy(y,yhat) if labels else accuracy2(y, yhat)
    return {'loss': loss, 'inference' : yhat, 'accuracy': acc}, W, b

def dirichlet(dims,n):
    s = prod(dims)
    return np.reshape(np.random.dirichlet(np.ones([s]), n), [n]+dims)

def dirichlet_initializer(dims,n):
    s = prod(dims)
    r = np.random.dirichlet(np.ones([s+2]), n).transpose().astype(np.float32)
    #print(r[0,0:10])
    return np.reshape(r[0:s,:], dims+[n]), np.ndarray.flatten(r[s:s+2,:])

def mwu_linear_layer(x, n, u=50, name='fc'):
    pm_x = pm(x)
    dim = int(pm_x.get_shape()[-1])
    #print('dim', dim)
    W0,b0 = dirichlet_initializer([dim],n)
    W = get_scope_variable('W'+name, initializer = W0, dtype = 'float32')
                           #shape=[dim, n], tf.constant_initializer(1/(dim+2)))
    b = get_scope_variable('b'+name, initializer = b0, dtype = 'float32')
    #shape=[2*n], 
    #print(b[0:n].get_shape())
    #print(tf.matmul(pm_x, W).get_shape())
    y = (u*(tf.matmul(pm_x, W) + b[0:n] - b[n:2*n]))
    #tf.nn.softmax
    return y, W, b

def make_softmax_model(y,yhat, labels=False):
    loss = tf.losses.softmax_cross_entropy(y, yhat)
    acc = accuracy(y,yhat) if labels else accuracy2(y, yhat)
    return {'loss': loss, 'inference' : yhat, 'accuracy': acc}

def mwu_conv_layer(x, dims, u=50, padding = 'valid', strides = (1,1), name='', f = tf.nn.relu):
                   #tf.nn.relu):
    x1 = pm(x)
    [a,b,c,d] = dims
    dims2 = [a, b, 2*c, d]
    W0,b0 = dirichlet_initializer([a,b,2*c],d)
    W = get_scope_variable('W'+name, initializer = W0)
                           #shape = dims2, initializer = tf.constant_initializer(1/(2*a*b*c+2)))
    bias = get_scope_variable('b'+name, initializer = b0)
                              #shape = [2*d], initializer = tf.constant_initializer(1/(2*a*b*c+2)))
    #print(bias[0:d].get_shape())
    #print(tf.nn.convolution(x1, W, padding, strides=strides).get_shape())
    y = f(u*(tf.nn.convolution(x1, W, padding, strides=strides)  + bias[0:d] - bias[d:2*d])) #, dilation_rate=None, name=None, data_format=None)
    tf.add_to_collection('Ws', W)
    tf.add_to_collection('bs', bias)
    return y, W, bias

def mnist_conv(x,y, u=50,u2=50):
    return mnist_conv_u(x,y,u,u,u,u2)

def mnist_conv_u(x,y, u1=50,u2=50, u3=50,u4=50):
    y1, W1, b1 = mwu_conv_layer(x,[8, 8, 1, 64], u=u1, name='1', strides=(2,2), padding = 'SAME')
    y2, W2, b2 = mwu_conv_layer(y1,[6, 6, 64, 128], u=u2, name='2', strides=(2,2), padding = 'VALID')
    y3, W3, b3 = mwu_conv_layer(y2,[5, 5, 128, 128], u=u3, name='3', padding = 'VALID')
    yf = tf.contrib.layers.flatten(y3)
    y4, W4, b4 = mwu_linear_layer(yf, 10, name='4', u=u4)
    #y4 = tf.Print(y4, [tf.nn.softmax(y4)], "output", summarize=10)
    model = make_softmax_model(y,y4)
    return model, [W1, W2, W4], [b1, b2, b4]

def mnist_conv1(x,y, u=50,u2=50):
    y1, W1, b1 = mwu_conv_layer(x,[8, 8, 1, 64], u=u, name='1', strides=(2,2), padding = 'SAME')
    yf = tf.contrib.layers.flatten(y1)
    y4, W4, b4 = mwu_linear_layer(yf, 10, name='4',u=u2)
    #y4 = tf.Print(y4, [tf.nn.softmax(y4)], "output", summarize=10)
    model = make_softmax_model(y,y4)
    return model, [W1, W4], [b1, b4]
