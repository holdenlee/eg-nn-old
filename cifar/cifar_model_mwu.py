from mwu.mwu import *
from utils.utils import *
from tf_utils.tf_vars import *
from tf_utils.tf_utils import *

def cifar_model_mwu(x, us=[20,20,20,20,20,20], weights=False):
    y1, W1, b1 = mwu_conv_layer(x,[3, 3, 3, 32], u=us[0], name='1', strides=(1,1), padding = 'SAME')
    y2, W2, b2 = mwu_conv_layer(y1,[3, 3, 32, 32], u=us[1], name='2', strides=(1,1), padding = 'VALID')
    #max pool
    #ksize, strides
    y2 = tf.nn.max_pool(y2, [1,2,2,1], [1,2,2,1], 'VALID')
    y3, W3, b3 = mwu_conv_layer(y2,[3, 3, 32, 64], u=us[2], name='3', padding = 'SAME')
    y4, W4, b4 = mwu_conv_layer(y3,[3, 3, 64, 64], u=us[3], name='4', padding = 'VALID')
    y4 = tf.nn.max_pool(y4, [1,2,2,1], [1,2,2,1], 'VALID')
    
    yf = tf.contrib.layers.flatten(y4)
    yl, Wl, bl = mwu_linear_layer(yf, 512, name='l', u=us[4])
    yl = tf.nn.relu(yl)
    yo, Wo, bo = mwu_linear_layer(yl, 10, name='o', u=us[5])
    #model = make_softmax_model(y,y4)
    if weights:
        return yo, [W1, W2, W3, W4, Wl, Wo], [b1, b2, b3, b4, bl, bo]
    else:
        return yo

