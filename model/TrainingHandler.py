import numpy as np
from tqdm import trange

class TrainingHandler():
    def __init__(self, models, datasets, callbacks):
        self.models = models
        self.datasets = datasets
        self.callbacks = callbacks

    def do_one_round(self, round, local_epochs):
        for i in trange(len(self.models)):
            self.models[i].fit(self.datasets[i], verbose = 0, epochs = local_epochs)

    def callbacks(self, round):
        for callback in self.callbacks:
            callback.model = self.model[0]
            callback.on_epoch_end(round)
    
    def on_round_end(self, round):
        weights = []
        for model in self.models:
            weights.append(model.trainable_variables)
        for i in range(len(self.models)):
            for j in range(len(self.models[i].trainable_variables)):
                self.models[i].trainable_variables[j].assign(np.average([weights[k][j] for k in range(len(self.models))]))

    def fit(self, rounds, local_epochs):
        for round in trange(rounds):
            print(f"Round: {round}/{rounds}")
            self.do_one_round(round, local_epochs)
            self.on_round_end(round)
            self.callbacks(round)