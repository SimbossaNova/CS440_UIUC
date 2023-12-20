# neuralnet_modified.py
# ---------------------
# The modified version of the neural network implementation for educational use.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.network = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_size)
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lrate)

    def forward(self, x):
        normalized_x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True))
        return self.network(normalized_x)

    def step(self, x, y):
        self.optimizer.zero_grad()
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    in_size = train_set.shape[1]
    out_size = np.unique(train_labels).size
    lrate = 0.01
    net = NeuralNet(lrate, nn.CrossEntropyLoss(), in_size, out_size)
    losses = []

    train_set_normalized = (train_set - train_set.mean(dim=0)) / train_set.std(dim=0)
    permutation = torch.randperm(train_set.size(0))
    train_set_shuffled = train_set_normalized[permutation]
    train_labels_shuffled = train_labels[permutation]

    for epoch in range(epochs):
        for i in range(0, train_set.size(0), batch_size):
            batch_end = min(i + batch_size, train_set.size(0))
            x_batch = train_set_shuffled[i:batch_end]
            y_batch = train_labels_shuffled[i:batch_end]

            loss = net.step(x_batch, y_batch)
            losses.append(loss)

    dev_set_normalized = (dev_set - dev_set.mean(dim=0)) / dev_set.std(dim=0)
    dev_predictions = net(dev_set_normalized)
    yhats = torch.argmax(dev_predictions, dim=1).numpy().astype(np.int)
    return losses, yhats, net