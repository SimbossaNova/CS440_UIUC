# neuralnet.py
# ---------------
# Licensing Information: You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful not to modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, learning_rate, loss_fn, input_size, output_size):

        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.input_size = input_size
        self.output_size = output_size
        self.function = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_size)
        )
        self.optimizer = optim.Adam(self.function.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.function(x.view(x.size(0), 3, 31, 31))

    def step(self, x, y):
        output = self.forward(x)
        self.optimizer.zero_grad()
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    processed_set = get_dataset_from_arrays(train_set, train_labels)
    loader = DataLoader(processed_set, batch_size=batch_size)
    net = NeuralNet(0.00042, torch.nn.CrossEntropyLoss(), 3072, 4)
    losses = []
    
    for i in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x = batch['features']
            y = batch['labels']
            epoch_loss += net.step(x, y)
        
        losses.append(epoch_loss)
    
    return losses, torch.argmax(net(dev_set), dim=1).detach().cpu().numpy().astype(int), net
