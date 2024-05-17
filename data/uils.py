
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

import ImageLoader

def split_dirichlet(l, n):
    a = (np.random.dirichlet(np.ones(n))*len(l)).astype(np.int32)
    a[-1] = len(l)-sum(a[:-1])
    res = [l[sum(a[:i]):sum(a[:i+1])] for i in range(len(a))]
    return res

def load_dataset(labels, imagePaths, image_transform, AUTO, batch_size=32, shuffle = False, seed=1312):
    imageLoader = ImageLoader(labels, imagePaths, image_transform)
    dataset = tf.data.Dataset.from_generator(imageLoader.iter,
                                             output_signature = (tf.TensorSpec(shape=(None, None, 3), dtype=tf.int32),
                                                                 tf.TensorSpec(shape=(None,), dtype=tf.int32)))
    dataset = dataset.map(imageLoader.margin_format, num_parallel_calls = AUTO)
    if (shuffle):
        dataset = dataset.shuffle(batch_size, seed = seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

def make_data(n, labels, imagePaths, image_transform, shuffle, iid, seed=1312):
    dataset = []
    if iid==1:
        skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(skf.split(imagePaths, np.argmax(labels, axis = 1))):
            X = imagePaths[test_index]
            y = labels[test_index]
            dataset.append(load_dataset(y, X, image_transform, shuffle))
    else:
        imagePaths = np.array(imagePaths)
        _labels = np.argmax(labels, axis = 1)
        xs = [[] for i in range(n)]
        ys = [[] for i in range(n)]
        for i in range(len(set(_labels))):
            splitted_idx = split_dirichlet(np.where(_labels == i)[0], n)
            for j in range(n):
                xs[j] += list(imagePaths[splitted_idx[j]])
                ys[j] += list(labels[splitted_idx[j]])

        for X, y in zip(xs, ys):
            dataset.append(load_dataset(y, X, image_transform, shuffle))
    return dataset