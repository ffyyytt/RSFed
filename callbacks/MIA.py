import numpy as np
import tensorflow as tf

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class MIA(tf.keras.callbacks.Callback):
    def __init__(self, member, nonmem):
        self.member = member
        self.nonmem = nonmem

    def on_epoch_end(self, epoch, logs={}):
        memberProb = self.model.predict(self.member, verbose = 0)
        nonmemProb = self.model.predict(self.nonmem, verbose = 0)

        X = np.vstack([memberProb, nonmemProb])
        Y = np.array([1]*len(memberProb) + [0]*len(nonmemProb))
        X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.33, shuffle=True, random_state=1312)
        clf = DecisionTreeClassifier(random_state=1312)
        clf.fit(X_train, y_train)
        print(f"Train score: {clf.score(X_train, y_train)}; Valid score: {clf.score(X_valid, y_valid)}")