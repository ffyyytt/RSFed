import numpy as np
import tensorflow as tf

class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self, data, label):
        self.data = data
        self.label = np.argmax(label, axis = 1)

    def on_epoch_end(self, epoch, logs={}):
        predProb = self.model.predict(self.data, verbose = 0)
        pred = np.argmax(predProb, axis = 1)
        print(f"Accuracy: {np.mean(pred == self.label)}")