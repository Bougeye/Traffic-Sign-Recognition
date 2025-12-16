import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import pandas as pd
import sys
import yaml
import time, datetime
import random
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import DataLoader, random_split, Subset
from src.data.gtsrb_dataset import GTSRBDataset
import src.models.ENV2 as models

class Training_loop:
    def __init__(self,model_variant="M", early_stopping=True):
        with open("config/training.yml","r") as f:
            self.tr_config = yaml.safe_load(f)
        with open("config/dataset.yml", "r") as f:
            self.ds_config = yaml.safe_load(f)
        self.model_variant = model_variant
        self.ds = GTSRBDataset(dataset_config="config/dataset.yml",
                               path_config="config/paths.yml")
        self.output = pd.DataFrame({})
        self.BPDC = self.tr_config["bpdc"]
        self.total_batches = self.ds_config["dataset"]["training_samples"]//self.tr_config["bsize"]
        self.progress = 0.0
        self.early_stopping = early_stopping

    def train(self):
        labels = [label for _,label in self.ds.samples]
        idx = list(range(len(self.ds)))

        kf = StratifiedKFold(n_splits=self.tr_config["folds"], shuffle=True, random_state=69)
        fold_id = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(idx,labels)):
            fold_id+=1
            train_idx = [idx[i] for i in train_idx]
            val_idx = [idx[i] for i in val_idx]

            train_ds = Subset(self.ds, train_idx)
            val_ds = Subset(self.ds, val_idx)

            self._train_one(train_ds, val_ds, fold_id)
            if fold_id == 1:
                break
        os.makedirs("Results", exist_ok=True)
        self.output.to_csv("Results/Output.csv",index=False)
        
    def _train_one(self, train_ds, val_ds, fold_id):
        Fold,Epoch,Batch,Train_Loss,Val_Loss = [],[],[],[],[]
        
        model = models.ENV2(self.model_variant)
        instance = model.get_instance()
        optimizer = model.get_optimizer()
        loss_fn = model.get_loss_fn()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        instance.to(device, memory_format=torch.channels_last)

        train_loader = DataLoader(train_ds, shuffle=False, batch_size=self.tr_config["bsize"])

        #Early Stopping Parameters
        self.patience = 5
        self.alpha = 0.3
        self.smooth_log = None
        self.best_smooth_log = float("inf")
        self.best_val_loss = 10**10
        self.min_improvement_factor = 1.05
        self.breaker = False
        
        start = time.time()
        for epoch in range(self.tr_config["epochs"]):
                if not self.breaker:
                    instance.train()
                    print(f"Epoch: {epoch} of {self.tr_config['epochs']}")
                    
                    total_samples = 0
                    running_loss = 0
                    for i, (xb,yb) in enumerate(train_loader):
                        instance.train()
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

                        total_samples+=self.tr_config["bsize"]
                        running_loss += loss.item()*self.tr_config["bsize"]
                        
                        if (i+1)%self.BPDC == 0:
                            end = time.time()
                            avg_loss = running_loss/total_samples
                            print(f"[batch {i+1}] samples: {total_samples}, Training Loss: {avg_loss:.4f}")
                            val_loss = self.validate(instance, device, loss_fn, val_ds, mode="train", target="full")
                            print(f"   Validation Loss: {val_loss:.4f}")
                            print(f"   Time since start: {str(datetime.timedelta(seconds=(end-start)))}")
                            Fold.append(fold_id)
                            Epoch.append(epoch)
                            Batch.append(i+1)
                            Train_Loss.append(avg_loss)
                            Val_Loss.append(val_loss)
                            if self.progress >= self.tr_config["earliest_stop"] and self.early_stopping == True:
                                self._check_early_stopping(val_loss)
                            if self.breaker:
                                break
                        self.progress += 1/(self.total_batches*self.tr_config["epochs"])
                            
                    epoch_loss = running_loss/total_samples
                    print(f"--m-Epoch {epoch+1} done.")
                    print(f"   Training Loss: {epoch_loss:.4f}")
                    val_loss = self.validate(instance, device, loss_fn, val_ds, mode="eval", target="full")
                    print(f"   Validation Loss: {val_loss:.4f}")
        self.output = pd.concat([self.output,pd.DataFrame({"Fold":Fold,"Epoch":Epoch,"Batch":Batch,"Training Loss":Train_Loss,"Validation Loss":Val_Loss})])
            

    def validate(self, instance, device, loss_fn, val_ds, mode="eval", target="batch"):
            
        instance.eval()
        if mode == "train":
            instance.train()

        val_loader = DataLoader(val_ds, shuffle=False, batch_size=self.tr_config["bsize"])
        total_batches = len(val_loader)
        
        batch_ids = None
        if target == "batch":
            batch_ids = random.sample(range(total_batches), k=min(self.BPDC, total_batches))
        if target == "full":
            batch_ids = range(total_batches)
        
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                if i not in batch_ids:
                    continue
                
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                yb[0] = yb[0].to(device, non_blocking=True)
                
                with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                    logits = instance(xb)
                    loss = loss_fn(logits,yb[0])
                    
                total_samples+=self.tr_config["bsize"]
                total_loss+=loss.item()*self.tr_config["bsize"]
                
        return total_loss/total_samples

    def _check_early_stopping(self, val_loss):
        log_loss = math.log(val_loss+1e-8)

        if self.smooth_log is None:
            self.smooth_log = log_loss
        else:
            self.smooth_log = log_loss*self.alpha + (1-self.alpha)*self.smooth_log

        improvement_factor = math.exp(self.best_smooth_log - self.smooth_log)
        if improvement_factor > self.min_improvement_factor:
            self.best_smooth_log = self.smooth_log
            self.patience = 5
        else:
            self.patience -= 1
            print(f"   Patience decreased - Patience is now {self.patience}")
        self.breaker = self.patience <= 0

    def debug_val_outliers(self, instance, device, loss_fn, val_ds, loss_threshold=1e3):
        instance.eval()
        loader = DataLoader(val_ds, shuffle=False, batch_size=self.tr_config["bsize"])

        print("--- Debugging validation outliers")
        with torch.no_grad():
            for i, (xb, yb) in enumerate(loader):
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = yb[0].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, enabled=False):  # disable AMP for debugging
                    logits = instance(xb)
                    loss = loss_fn(logits, labels)

                if not torch.isfinite(loss):
                    print(f"[batch {i}] INFINITE LOSS:", loss.item())
                elif loss.item() > loss_threshold:
                    print(f"[batch {i}] HIGH LOSS: {loss.item():.4f}")
                    print("    labels dtype:", labels.dtype)
                    print("    labels shape:", labels.shape)
                    print("    labels min/max:", labels.min().item(), labels.max().item())

        
    def get_model(self):
        return self.model

if __name__ == "__main__":
    X = Training_loop(model_variant="S", early_stopping=False)
    X.train()
