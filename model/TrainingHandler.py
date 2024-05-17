import numpy as np
from tqdm import trange

class TrainingHandler():
    def __init__(self, models, datasets, valid_labels, train_dataset, valid_dataset):
        self.models = models
        self.datasets = datasets

        self.valid_labels = valid_labels
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def do_one_round(self, round, local_epochs):
        valid_acc = []
        for i in trange(len(self.models)):
            self.models[i].fit(self.datasets[i], verbose = 0, epochs = local_epochs)

    def evaluate(self):
        y_pred = np.argmax(self.models[0].predict(self.valid_dataset, verbose = 0), axis = 1)
        valid_acc = np.mean(y_pred==np.argmax(self.valid_labels, axis = 1))
        return valid_acc
    
    def on_round_end(self, round):
        weights = []
        for model in self.models:
            weights.append(model.trainable_variables)
        weights_mean = np.average(weights, axis = 0)
        for i in range(len(self.models)):
            for j in range(len(self.models[i].trainable_variables)):
                self.models[i].trainable_variables[j].assign(weights_mean[j])

    def fit(self, rounds, local_epochs):
        for round in trange(rounds):
            print(f"Round: {round}/{rounds}")
            self.do_one_round(round, local_epochs)
            self.on_round_end(round)
            print(f"Accuracy: {self.evaluate()}")