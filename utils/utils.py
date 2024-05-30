import os
import random

import numpy as np
import tensorflow as tf

def seedBasic(seed=1312):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        tf.config.set_soft_device_placement(True)
    except ValueError:
        gpus = tf.config.experimental.list_logical_devices("GPU")
        if len(gpus) > 0:
            strategy = tf.distribute.MirroredStrategy(gpus)
        else:
            strategy = tf.distribute.get_strategy()
    AUTO = tf.data.experimental.AUTOTUNE
    return strategy, AUTO