import numpy as np
import tensorflow as tf
from tqdm import trange

class TrainingHandler():
    def __init__(self, method, server, models, datasets, callbacks):
        self.method = method
        self.server = server
        self.models = models
        self.datasets = datasets
        self.callbacks = callbacks

    def do_one_round(self, round, local_epochs):
        for i in trange(len(self.models)):
            self.models[i].fit(self.datasets[i], verbose = 0, epochs = local_epochs)

    def do_callbacks(self, round):
        for callback in self.callbacks:
            callback.model = self.models[0]
            callback.on_epoch_end(round)

    def broadcast(self, round):
        for i in range(len(self.models)):
            for j in range(len(self.models[i].trainable_variables)):
                self.models[i].trainable_variables[j].assign(self.server.trainable_variables[j])
    
    def on_round_end(self, round):
        weights = []
        for model in self.models:
            weights.append(model.trainable_variables)
        if self.method == "FedAVG":
            for i in range(len(self.server.trainable_variables)):
                self.server.trainable_variables[i].assign(np.average([weights[k][i] for k in range(len(self.models))], axis = 0))
        elif self.method == "FedNova":
            for i in range(len(self.server.trainable_variables)):
                w = tf.math.reduce_mean([tf.math.l2_normalize(weights[k][i] - self.server.trainable_variables[i]) for k in range(len(self.models))], axis = 0)
                print(w)
                print(np.average([weights[k][i] for k in range(len(self.models))], axis = 0)-self.server.trainable_variables[i])

    def fit(self, rounds, local_epochs):
        for round in trange(rounds):
            print(f"Round: {round}/{rounds}")
            self.do_one_round(round, local_epochs)
            self.on_round_end(round)
            self.do_callbacks(round)