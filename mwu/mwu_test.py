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

from .mwu import *

def basic_test(dim=100, print_steps=1, max_steps=10, batch_size=128):
    i = np.random.randint(0, dim)
    sgn = 2*np.random.randint(0,2)-1
    bf = BatchFeeder([dim, i, sgn], 100, synthesize_data)
    x = tf.placeholder(tf.float32, shape=(None, dim))
    y = tf.placeholder(tf.float32, shape=(None, 2))
    model, W, b = mnist_linear(x,y, dim,n=2, labels=False)
    ph_dict = {'x': x, 'y': y}
    evaler = Eval(bf, 100, ['accuracy'], eval_steps = print_steps, name='test')
    opter= lambda: MWOptimizer([W], [b], learning_rate=0.1)
    addons = [Train(opter, 
                    batch_size, 
                    train_feed={}, 
                    loss = 'loss', 
                    print_steps=print_steps),
              evaler]
    trainer = Trainer(model, max_steps, bf, addons, ph_dict, train_dir = 'pm1/', verbosity=1)
    trainer.init()
    print("Weights:",trainer.sess.run([W,b]))
    trainer.train()
    [weights, biases] = trainer.sess.run([W,b])
    print("Weights:",weights, biases)
    print(np.sum(weights, axis=0))
    print(np.max(weights, axis=0))
    trainer.finish()
    return evaler.record[-1][0]
