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

import operator

if __name__=='__main__':
    np.set_printoptions(threshold=np.inf, precision=3)
    ans = mnist_conv_test(20,20,0.3, 10000,smoothing=0.00)
    print(ans)
