import os
import sys
import yaml
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
from src.data.gtsrb_dataset import GTSRBDataset
from src.data.concepts_dataset import ConceptsDataset

from sklearn.model_selection import train_test_split

from src.training.Training_Loop import Training_Loop
import src.models.ENV2 as stage_1_models
import src.models.LabelModel as stage_2_models

class train:
    def __init__(self, model_variant="M", early_stopping=True):
        with open("config/training.yml", "r") as f:
            self.tr_cfg = yaml.safe_load(f)
        with open("config/dataset.yml", "r") as f:
            self.ds_cfg = yaml.safe_load(f)
        with open("config/paths.yml", "r") as f:
            self.pth_cfg = yaml.safe_load(f)
        self.model_stage_1 = stage_1_models.ENV2(model_variant=model_variant, lr=self.tr_cfg["stage_1"]["lr"], optimizer=self.tr_cfg["stage_1"]["optimizer"])
        self.model_stage_2 = stage_2_models.LabelModel(lr=self.tr_cfg["stage_2"]["lr"], optimizer=self.tr_cfg["stage_2"]["optimizer"],
                                                       layers=3,hidden_dim=128,hidden_dim2=64)
        self.early_stopping = early_stopping


    def train(self):
        self.train_1 = Training_Loop(epochs=self.tr_cfg["stage_1"]["epochs"], bsize=self.tr_cfg["stage_1"]["bsize"],
                                     bpdc=self.tr_cfg["stage_1"]["bpdc"], patience=self.tr_cfg["stage_1"]["patience"],
                                     min_delta=self.tr_cfg["stage_1"]["min_delta"],early_stopping=self.early_stopping,
                                     multi_label=True)
        self.train_2 = Training_Loop(epochs=self.tr_cfg["stage_2"]["epochs"], bsize=self.tr_cfg["stage_2"]["bsize"],
                                     bpdc=self.tr_cfg["stage_2"]["bpdc"], patience=self.tr_cfg["stage_2"]["patience"],
                                     min_delta=self.tr_cfg["stage_2"]["min_delta"],early_stopping=self.early_stopping,
                                     multi_label=False)
        
        self.train_1.set_model(self.model_stage_1)
        self.train_2.set_model(self.model_stage_2)
        
        dataset_1 = GTSRBDataset(self.ds_cfg,
                                 self.pth_cfg)
        
        total_samples = len(dataset_1)
        labels = [dataset_1[i][1][1] for i in range(len(dataset_1))]
        idx = list(range(len(dataset_1)))

        train_idx, val_idx = train_test_split(list(range(len(dataset_1))), test_size=0.2, random_state=69, stratify=labels)
        train_idx = [idx[i] for i in train_idx]
        val_idx = [idx[i] for i in val_idx]

        ds_train = Subset(dataset_1, train_idx)
        ds_val = Subset(dataset_1, val_idx)
        
        self.train_1.train(ds_train, ds_val, out_folder="stage_1")
        ds_train = self.forward_stage_1(ds_train)
        ds_val = self.forward_stage_1(ds_val)
        self.train_2.train(ds_train, ds_val, out_folder="stage_2")
        

    def forward_stage_1(self,dataset):
        X_in, y_in = [],[]
        instance = self.model_stage_1.get_instance()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        loader = DataLoader(dataset, shuffle=True, num_workers=8, persistent_workers=True,
                            pin_memory = (device.type=="cuda"), batch_size=self.tr_cfg["stage_1"]["bsize"])
        
        instance.to(device, memory_format=torch.channels_last)
        instance.eval()
        
        with torch.no_grad():
            for i, (xb,yb) in enumerate(loader):
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                yb[0] = yb[0].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                    logits = instance(xb)

                pred = (logits.detach().cpu() >= 0).numpy().astype(float)
                target = yb[0].detach().cpu().numpy().astype(float)
                for j in range(len(pred)):
                    X_in.append(pred[j])
                for j in range(len(yb[1])):
                    y_in.append(int(yb[1][j]))
        dataset = ConceptsDataset(X_in, y_in)
        
        return dataset

    def read_dataset(self,dataset):
        loader = DataLoader(dataset)
        for i,(xb,yb) in enumerate(loader):
            print(xb,yb)

    def experiment_training(self, model_variant="M", early_stopping=True):
        ###Learning rate optimization
        gtsrb_ds = GTSRBDataset(self.ds_cfg,
                                self.pth_cfg)
        for learning_rate in [0.01,0.001,0.0001,0.00001]:
            model_stage_1 = stage_1_models.ENV2(model_variant,learning_rate, self.tr_cfg["stage_1"]["optimizer"])
            train_1 = Training_Loop(epochs=self.tr_cfg["stage_1"]["epochs"], bsize=self.tr_cfg["stage_1"]["bsize"],
                                                bpdc=self.tr_cfg["stage_1"]["bpdc"], patience=self.tr_cfg["stage_1"]["patience"],
                                                min_delta=self.tr_cfg["stage_1"]["min_delta"],early_stopping=early_stopping)
            train_1.set_model(model_stage_1)
            train_1.train(dataset=gtsrb_ds, out_folder=f"lr-{learning_rate}")

if __name__ == "__main__":
    x = train(model_variant="S", early_stopping=True)
    x.train()
    #ds = x.forward_stage_1()
    #x.read_dataset(ds)
    
