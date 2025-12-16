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
        """
        Req: A model variant is given as a string being either 'S','M' or 'L'.
        Eff: A modified EfficientNetV2 is created, hyper-parameters are loaded
             through yml or set by the class.
        Res: -
        """
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
        """
        Req: A model variant is given as a string being either 'S','M' or 'L'.
        Eff: The model is switched to the given variant of EfficientNetV2.
        Res: -
        """
        self.model_variant = model_variant
        self.model = self.variant_map[model_variant]()
        self._modify_model()

    def get_model(self) -> str:
        """
        Req: -
        Eff: -
        Res: The current model is returned as a String.
        """
        return f"Model: {self.base_model} {self.model_variant}"

    def _modify_model(self):
        """
        Req: -
        Eff: The model has its last layer removed and replaced by a 43 neuron FC layer.
        Res: -
        """
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]),
                                         nn.Flatten(1),
                                         torch.nn.Linear(1280,43))

    def set_loss_fn(self,loss_fn=nn.BCEWithLogitsLoss()):
        """
        Req: A loss function -loss_fn- may be provided from the torch.nn module.
             If no loss function is given, nn.BCEWithLogitsLoss() serves as standard.
        Eff: The model's loss function is switched to -loss_fn-.
        Res: -
        """
        self.loss_fn = loss_fn

    def get_loss_fn(self) -> object:
        """
        Req: -
        Eff: -
        Res: The model's loss function is returned as a torch.nn function.
        """
        return self.loss_fn

    def set_optimizer(self, optimizer="adam"):
        """
        Req: An optimizer -optimizer- may be provided as a String, 
             it may be one of the following: 'adam', 'sgd', 'adagrad', 'adadelta'.
             If no optimizer is given, 'adam' serves as standard.
        Eff: The model's optimizer is set to -optimizer-
        Res: -
        """
        self.optimizer = self.optimizer_map[optimizer]

    def get_optimizer(self) -> object:
        """
        Req: -
        Eff: -
        Res: The model's optimizer is returned as a torch.optim function.
        """
        return self.optimizer

    def get_instance(self) -> object:
        """
        Req: -
        Eff: -
        Res: The model is returned as a torch model.
        """
        return self.model

