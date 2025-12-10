import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
import os
import sys
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from torch.utils.data import DataLoader
from src.data.gtsrb_dataset import GTSRBDataset

class ENV2:
    def __init__(self,model_variant="M"):
        self.base_model = "EffcientNetV2"
        self.variant_map = {
            "S": lambda:models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT),
            "M": lambda:models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT),
            "L": lambda:models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)}
        
        self.model_variant = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        
        with open("config/training.yml", "r") as f:
            self.config = yaml.safe_load(f)
        self.switch_variant(model_variant)
        self.set_loss_fn()
        
        self.optimizer_map = {
            "adam": torch.optim.Adam(self.model.parameters(), self.config["lr"]),
            "sgd": torch.optim.SGD(self.model.parameters(), self.config["lr"]),
            "adagrad": torch.optim.Adagrad(self.model.parameters(), self.config["lr"]),
            "adadelta": torch.optim.Adadelta(self.model.parameters(), self.config["lr"])}
        self.set_optimizer(self.config["optimizer"])
        
    def switch_variant(self,model_variant="M"):
        self.model_variant = model_variant
        self.model = self.variant_map[model_variant]()
        self.modify_model()

    def get_model(self):
        return f"Model: {self.base_model} {self.model_variant}"

    def modify_model(self):
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]),
                                         nn.Flatten(1),
                                         torch.nn.Linear(1280,43))

    def set_loss_fn(self,loss_fn=nn.BCEWithLogitsLoss()):
        self.loss_fn = loss_fn

    def get_loss_fn(self):
        return self.loss_fn

    def set_optimizer(self, optimizer="adam"):
        self.optimizer = self.optimizer_map[optimizer]

    def get_optimizer(self):
        return self.optimizer

    def get_instance(self):
        return self.model

