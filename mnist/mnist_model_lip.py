from lipschitz.lipschitz import *
import tensorflow as tf

def mnist_model_lip(x):
    x = tf.transpose(x, [0,2,3,1])
    #NOTE: x should be (batch_size * filters * w * h) rather than the usual (batch_size * w * h * filters)
    y1 = lip_conv_layer(x, 64, dims, la=1, strides=(2,2), name='1') # same
    y2 = lip_conv_layer(y1, 128, dims, la=1, strides=(2,2), name='2') #valid 
    y3 = lip_conv_layer(y2, 128, dims, la=1, name='3') 
    #yf = tf.reshape(y3, [-1, 7*7*128])
    #http://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
    yf = tf.contrib.layers.flatten(y3)
    y4, W4, b4 = lip_linear_layer(yf, 10, name='4')
    y = tf.nn.softmax(y4)
    return y