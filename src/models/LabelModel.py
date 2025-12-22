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

class LabelModel():
    def __init__(self, lr=0.01, optimizer="adam", layers=1, hidden_dim=0, hidden_dim2=0):
        
        self.countLayers = layers

        self.optimizer_map = {
            "adam": torch.optim.Adam(self.model.parameters(), lr),
            "sgd": torch.optim.SGD(self.model.parameters(), lr),
            "adagrad": torch.optim.Adagrad(self.model.parameters(), lr),
            "adadelta": torch.optim.Adadelta(self.model.parameters(), lr)
        }
        
        self.loss_fn = None
        self.optimizer = None

        self.set_loss_fn()
        self.set_optimizer(optimizer)

        self.model = self._build_model(layers, hidden_dim, hidden_dim2)
    
    def _build_model(self, layers, hidden_dim, hidden_dim2):
        if layers == 1:
            return nn.Sequential(
                nn.Linear(43, 43)
            )
        elif layers == 2:
            return nn.Sequential(
                nn.Linear(43, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 43)
            )
        elif layers == 3:
            return nn.Sequential(
                nn.Linear(43, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, 43)
            )

    def get_loss_fn(self):
        return self.loss_fn

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizier="adam"):
        self.optimizer = self.optimizer_map[optimizer]

    def set_loss_fn(self, loss_fn=nn.CrossEntropyLoss()):
        self.loss_fn = loss_fn

    def get_instance(self):
        return self.model

    def get_model(self) -> str:
        return f"Model: {self.base_model} ({self.layers} layers)"
