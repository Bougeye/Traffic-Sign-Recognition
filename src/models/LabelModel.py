import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
import os
import sys
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from torch.utils.data import DataLoader

class LabelModel(nn.Module):
    def __init__(self, layers=1, hidden_dim=0, hidden_dim2=0):
        super().__init__()
        
        self.countLayers = layers

        self.optimizer_map = {
            "adam": torch.optim.Adam(self.model.parameters(), self.config["lr"]),
            "sgd": torch.optim.SGD(self.model.parameters(), self.config["lr"]),
            "adagrad": torch.optim.Adagrad(self.model.parameters(), self.config["lr"]),
            "adadelta": torch.optim.Adadelta(self.model.parameters(), self.config["lr"])
        }
        
        self.loss_fn = None
        self.optimizer = None

        self.setLossFn()
        self.setOptimizer(self.config["optimizer"])
        
        if(layers == 1):
            self.fc1 = nn.Linear(43,43)
            
        elif(layers == 2):
            self.fc1 = nn.Linear(43,hidden_dim)
            self.fc2 = nn.Linear(hidden_dim,43)
            
        elif(layers == 3):
            self.fc1 = nn.Linear(43,hidden_dim)
            self.fc2 = nn.Linear(hidden_dim,hidden_dim2)
            self.fc3 = nn.Linear(hidden_dim2,43)

    def forward(self, x):
        if self.countLayers == 1:
            return self.fc1(x)

        elif self.countLayers == 2:
            x = F.relu(self.fc1(x))
            return self.fc2(x)

        elif self.countLayers == 3:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def getLossFn(self):
        return self.loss_fn

    def getOptimizer(self):
        return self.optimizer

    def setOptimizer(self, optimizier="adam"):
        self.optimizer = self.optimizer_map[optimizer]

    def setLossFn(self, loss_fn=nn.CrossEntropyLoss()):
        self.loss_fn = loss_fn
