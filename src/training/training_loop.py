import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, KFold
import os
import pandas as pd
import sys
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import DataLoader, random_split, Subset
from src.data.gtsrb_dataset import GTSRBDataset
import src.models.model as models

class Training_loop:
    def __init__(self,model_variant="M"):
        with open("config/training.yml","r") as f:
            self.config = yaml.safe_load(f)
        self.model_variant = model_variant
        self.ds = GTSRBDataset(dataset_config="config/dataset.yml",
                               path_config="config/paths.yml")
        self.output = pd.DataFrame({})

    def train(self):
        labels = [label for _,label in self.ds.samples]
        train_idx, test_idx = train_test_split(list(range(len(self.ds))), test_size = 0.2, random_state=69,stratify=labels)
        test_df = Subset(self.ds, test_idx)

        kf = KFold(n_splits=self.config["folds"], shuffle=True, random_state=69)
        fold_id = 0
        for fold, (subtrain_idx, val_idx) in enumerate(kf.split(train_idx)):
            fold_id+=1
            subtrain_idx = [train_idx[i] for i in subtrain_idx]
            val_idx = [train_idx[i] for i in val_idx]

            train_ds = Subset(self.ds, subtrain_idx)
            val_ds = Subset(self.ds, val_idx)

            self._train_one(train_ds, val_ds, fold_id)
            if fold_id == 2:
                break
        self.output.to_csv("Output.csv",index=False)
        
    def _train_one(self, train_ds, val_ds, fold_id):
        Fold,Epoch,Train_Loss,Val_Loss = [],[],[],[]
        
        model = models.ENV2(self.model_variant)
        instance = model.get_instance()
        optimizer = model.get_optimizer()
        loss_fn = model.get_loss_fn()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        instance.to(device, memory_format=torch.channels_last)

        train_loader = DataLoader(train_ds, shuffle=False, batch_size=self.config["bsize"])

        for epoch in range(self.config["epochs"]):
            instance.train()
            print(f"Epoch: {epoch} of {self.config['epochs']}")
            
            total_samples = 0
            running_loss = 0
            
            for i, (xb,yb) in enumerate(train_loader):
                instance.train()
                concept_vector = yb[0]
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                yb[0] = yb[0].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                    logits = instance(xb)
                    loss = loss_fn(logits,yb[0])
                    
                scaler.scale(loss).backward()
                #torch.nn.utils.clip_grad_norm_(instance.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_samples+=self.config["bsize"]
                running_loss += loss.item()*self.config["bsize"]
                
                if (i+1)%20 == 0:
                    avg_loss = running_loss/total_samples
                    print(f"[batch {i+1}] samples: {total_samples}, Loss: {avg_loss:.4f}")
            
            epoch_loss = running_loss/total_samples
            print(f"--m-Epoch {epoch+1} done.")
            print(f"   Training Loss: {epoch_loss:.4f}")
            val_loss = self.validate(instance, device, loss_fn, val_ds=val_ds)
            print(f"   Validation Loss: {val_loss:.4f}")
            Fold.append(fold_id)
            Epoch.append(epoch)
            Train_Loss.append(epoch_loss)
            Val_Loss.append(val_loss)
        self.output = pd.concat([self.output,pd.DataFrame({"Fold":Fold,"Epoch":Epoch,"Training Loss":Train_Loss,"Validation Loss":Val_Loss})])
            

    def validate(self, instance, device, loss_fn, val_ds="Full"):
        if val_ds == "Full":
            val_ds = self.ds
        instance.eval()
        val_loader = DataLoader(val_ds, shuffle=False, batch_size=self.config["bsize"])
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                concept_vector = yb[0]
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                yb[0] = yb[0].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                    logits = instance(xb)
                    loss = loss_fn(logits,yb[0])
                total_samples+=self.config["bsize"]
                total_loss+=loss.item()*self.config["bsize"]
        return total_loss/total_samples

    def get_model(self):
        return self.model

if __name__ == "__main__":
    X = Training_loop(model_variant="M")
    X.train()
