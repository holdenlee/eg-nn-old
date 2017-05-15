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
from .tf_utils import *


class GenBatchFeeder(BatchFeeder):
  def __init__(self, xs, ys, datagen, batch_size): #, batch_size):
      self.datagen = datagen
      self.datagen.fit(xs)
      self.flow = self.datagen.flow(xs, ys, batch_size=batch_size)
      self.batch_size = batch_size
      self.xs = xs
      self.ys = ys
      self.index = 0
      self.num_examples = len(xs) #num_examples
      self.epochs_completed = 0

  def next_batch(self, batch_size=None, args=None):
      #batch_size is ignored. args is ignored.
      #, refresh_f = shuffle_refresh):
      #does keras shuffle? If so, shuffle_refresh is not necessary
      start = self.index
      if start >= self.num_examples:
          self.epochs_completed +=1
          #restart the datagen (is this necessary?)
          self.flow = self.datagen.flow(self.xs, self.ys, batch_size=self.batch_size)
      x_batch, y_batch = self.flow.next()
      return {'x': x_batch, 'y': y_batch}

#batch size must be pre-specified

"""
def datagen_batch_feeder_f(bf, batch_size)
    #, refresh_f = shuffle_refresh):
    #does keras shuffle? If so, shuffle_refresh is not necessary
    start = bf.index
    if start >= bf.num_examples:
        refresh_f(bf)

    # for simplicity, discard those at end.
    x_batch, y_batch = bf.args.next()
    bf.index += batch_size

    if bf.index > bf.num_examples:
      # Finished epoch
      bf.epochs_completed += 1
      # Shuffle the data
      refresh_f(bf)
      # Start next epoch
      start = 0
      bf.index = batch_size
      assert batch_size <= bf.num_examples
    return { k : v[start:end] for (k,v) in bf.args.items()}
    end = bf.index
"""

def bfs_from_datagen_f(X_train, Y_train, X_test, Y_test, datagen, batch_size, gen_for_test = False):
    train_data = GenBatchFeeder(X_train, Y_train, datagen, batch_size)
    test_data = GenBatchFeeder(X_test, Y_test, datagen, batch_size) if gen_for_test else make_batch_feeder({'x': X_test, 'y':Y_test})
    return train_data, test_data


