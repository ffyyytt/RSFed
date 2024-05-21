from data.uils import *
from model.utils import *
from utils.utils import *
from callbacks.MIA import *

import os
import glob
import pickle
import tensorflow as tf
import albumentations as A

strategy, AUTO = strategy()

labels = []
labelset = {}
imagePaths = glob.glob(os.path.expanduser("~")+"/datasets/AID/*/*")
for file in imagePaths:
    if file.split("/")[-2] not in labelset:
        labelset[file.split("/")[-2]] = len(labelset)
    labels.append(file.split("/")[-2])

for k, v in labelset.items():
    a = [0]*len(labelset)
    a[labelset[k]] = 1
    labelset[k] = a
imagePaths = np.array(imagePaths)
labels = np.array([labelset[label] for label in labels])
trainImagePaths, validImagePaths, trainLabels, validLabels = train_test_split(imagePaths, labels, test_size = 0.3, shuffle=True, random_state=1312)

valid_transform  = A.Compose([A.Resize(width=256, height=256)])
train_dataset = load_dataset(trainLabels, trainImagePaths, valid_transform, AUTO)
valid_dataset = load_dataset(validLabels, validImagePaths, valid_transform, AUTO)

train_federated_dataset = make_data(10, trainLabels, trainImagePaths, valid_transform, True, True)

flModels = []
with strategy.scope():
    model = simple_model_factory(backbone = "resnet18",
                                 n_classes = len(labelset))
    
    for i in range(10):
        _model = tf.keras.models.clone_model(model)
        _model.set_weights(model.get_weights())
        _model.compile(optimizer = "SGD",
                       loss = [tf.keras.losses.CategoricalCrossentropy()],
                       metrics = [tf.keras.metrics.CategoricalAccuracy()])
        flModels.append(_model)
    
    mia = MIA(train_dataset, valid_dataset)
    trainingHandler = TrainingHandler(flModels, train_federated_dataset, [mia])

trainingHandler.fit(rounds = 10, local_epochs = 5)