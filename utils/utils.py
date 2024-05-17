import os
import random

import numpy as np
import tensorflow as tf

def seedBasic(seed=1312):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)