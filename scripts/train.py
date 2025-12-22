import os
import sys
import yaml
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import TensorDataset, DataLoader
from src.data.gtsrb_dataset import GTSRBDataset

from src.training.Training_Loop import Training_Loop
import src.models.ENV2 as stage_1_models
import src.models.LabelModel as stage_2_models

class train:
    def __init__(self, model_variant="M", early_stopping=True):
        with open("config/training.yml", "r") as f:
            self.tr_cfg = yaml.safe_load(f)
        with open("config/dataset.yml", "r") as f:
            self.ds_cfg = yaml.safe_load(f)
        self.model_stage_1 = stage_1_models.ENV2(model_variant,self.tr_cfg["lr"], self.tr_cfg["optimizer"])
        self.model_stage_2 = stage_2_models.LabelModel(3,128,64)
        self.early_stopping = early_stopping

    def train(self):
        self.train_1 = Training_Loop(epochs=self.tr_cfg["epochs"], bsize=self.tr_cfg["bsize"],
                                              bpdc=self.tr_cfg["bpdc"], patience=self.tr_cfg["patience"],
                                              min_delta=self.tr_cfg["min_delta"],early_stopping=self.early_stopping)
        self.train_2 = Training_Loop(epochs=self.tr_cfg["epochs"], bsize=self.tr_cfg["bsize"],
                                              bpdc=self.tr_cfg["bpdc"], patience=self.tr_cfg["patience"],
                                              min_delta=self.tr_cfg["min_delta"],early_stopping=self.early_stopping)
        
        self.train_1.set_model(self.model_stage_1)
        self.train_2.set_model(self.model_stage_2)
        
        dataset_1 = GTSRBDataset(dataset_config="config/dataset.yml",
                                 path_config="config/paths.yml")
        self.train_1.train(dataset_1, out_folder="Stage_1")
        
        dataset_2 = self.forward_stage_1()
        self.train_2.train(dataset_2, out_folder="Stage_2")

    def forward_stage_1(self):
        X_in, y_in = [],[]
        instance = self.model_stage_1.get_instance()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        loss_fn = self.model_stage_1.get_loss_fn()
        
        gtsrb_ds = GTSRBDataset(dataset_config="config/dataset.yml",
                                path_config="config/paths.yml")
        loader = DataLoader(gtsrb_ds, shuffle=True, num_workers=8, persistent_workers=True,
                            pin_memory = (device.type=="cuda"), batch_size=self.tr_cfg["bsize"])
        
        instance.to(device, memory_format=torch.channels_last)
        
        for i, (xb,yb) in enumerate(loader):
            print(f"Batch {i}")
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb[0] = yb[0].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                logits = instance(xb)
                loss = loss_fn(logits,yb[0])

            pred = (logits.detach().cpu() >= 0).numpy().astype(float)
            target = yb[0].detach().cpu().numpy().astype(float)
            X_in.append(pred)
            y_in.append(yb[1])
            
        X_in = torch.Tensor(np.vstack(X_in))
        y_in = torch.Tensor(torch.cat(y_in, dim=0))
        dataset = TensorDataset(X_in, y_in)
        
        return dataset

    def read_dataset(self,dataset):
        loader = DataLoader(dataset)
        for i,(xb,yb) in enumerate(loader):
            print(xb,yb)

    def experiment_training(self, model_variant="M", early_stopping=True):
        ###Learning rate optimization
        gtsrb_ds = GTSRBDataset(dataset_config="config/dataset.yml",
                                path_config="config/paths.yml")
        for learning_rate in [0.01,0.001,0.0001,0.00001]:
            model_stage_1 = stage_1_models.ENV2(model_variant,learning_rate, self.tr_cfg["optimizer"])
            train_1 = Training_Loop(epochs=self.tr_cfg["epochs"], bsize=self.tr_cfg["bsize"],
                                                bpdc=self.tr_cfg["bpdc"], patience=self.tr_cfg["patience"],
                                                min_delta=self.tr_cfg["min_delta"],early_stopping=early_stopping)
            train_1.set_model(model_stage_1)
            train_1.train(dataset=gtsrb_ds, out_folder=f"lr-{learning_rate}")

if __name__ == "__main__":
    x = train(model_variant="S", early_stopping=True)
    x.train()
