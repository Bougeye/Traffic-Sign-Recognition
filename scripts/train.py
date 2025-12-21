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

import src.training.Stage_1 as Stage_1

class train:
    def __init__(self, model_variant="M", early_stopping=True):
        with open("config/training.yml", "r") as f:
            self.tr_cfg = yaml.safe_load(f)
        with open("config/dataset.yml", "r") as f:
            self.ds_cfg = yaml.safe_load(f)
        self.Stage1 = Stage_1.Training_loop(model_variant=model_variant, epochs=self.tr_cfg["epochs"],
                                       lr=self.tr_cfg["lr"], bsize=self.tr_cfg["bsize"], optimizer=self.tr_cfg["optimizer"],
                                       folds=self.tr_cfg["folds"], bpdc=self.tr_cfg["bpdc"], early_stopping=early_stopping)
        
    def Train_Stage_1(self):
        self.Stage1.train()

    #def Train_Stage_2(self):
        #self.Stage2.train(ds=Forward_Stage_1())

    def Forward_Stage_1(self):
        X_in, y_in = [],[]
        instance = self.Stage1.get_model().get_instance()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        loss_fn = self.Stage1.get_model().get_loss_fn()
        
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

    def Experiment_Training(self, model_variant="M", early_stopping=True):
        ###Learning rate optimization
        for learning_rate in [0.01,0.001,0.0001,0.00001]:
            Stage1 = Stage_1.Training_loop(model_variant=model_variant, epochs=self.tr_cfg["epochs"],
                                           lr=learning_rate, bsize=self.tr_cfg["bsize"], optimizer=self.tr_cfg["optimizer"],
                                           folds=self.tr_cfg["folds"], bpdc=self.tr_cfg["bpdc"], early_stopping=early_stopping)
            Stage1.train(out_folder=f"lr-{learning_rate}")

if __name__ == "__main__":
    x = train(model_variant="S", early_stopping=True)
    x.Train_Stage_1()
    ds = x.Forward_Stage_1()
    x.Train_Stage_2(ds)
