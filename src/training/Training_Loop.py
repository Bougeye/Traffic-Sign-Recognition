import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import os
import pandas as pd
import sys
import yaml
import time, datetime
import random
import math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import DataLoader, random_split, Subset
from src.data.gtsrb_dataset import GTSRBDataset
import src.utils.plots as plots
import src.models.ENV2 as models

class Training_Loop:
    
    def __init__(self, epochs=15, bsize=16, bpdc=20, patience=5, min_delta=0.05, early_stopping=True, multi_label=False):
        """
        Req.: The following list of parameters may be provided:
              #epochs: The amount of epochs the model goes through in training
              #bsize: The batch size used when fetching samples from the dataset
              #bpdc: The amount of batches after which mid-epoch data is gathered
              #patience: The amount of times validation loss can exceed the threshold before stopping early
              #min_delta: The minimal percentage improvement validation loss needs to show after an epoch
              #early_stopping: Enables early stopping
              #multi_label: If the output data allows multi-label classification
        Eff.: A training loop with the specified parameters is instanciated. 
        Res.: -
        """
        self.model = None
        self.epochs = epochs
        self.bsize = bsize
        self.bpdc = bpdc
        self.multi_label = multi_label

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self.output_batches, self.output_epochs = pd.DataFrame({}),pd.DataFrame({})
        self.output_report, self.output_accuracy = pd.DataFrame({}),pd.DataFrame({})

    
    def train(self, dataset, out_folder=""):
        """
        Req.: A dataset is provided that the model can be trained on.
              The dataset shall provide a tuple (x,y) as sample where x is the model input and y is the target.
              Additionally, -out_folder- may be set for metrics and plots storage, otherwise base folder is standard.
        Eff.: A previously set model is trained on the dataset with the hyperparameters set at initialization of the loop.
        Res.: A dictionary of training metrics is returned, containing accuracy, validation loss, training loss, training time etc..
        """
          
        
        if self.model == None:
            print("Error: Model not configured")
            return

        
        
        total_samples = len(dataset)
        labels = [dataset[i][1][1] for i in range(len(dataset))]
        idx = list(range(len(dataset)))

        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=69, stratify=labels)
        train_idx = [idx[i] for i in train_idx]
        val_idx = [idx[i] for i in val_idx]

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        Epoch,Batch,Train_Loss,Val_Loss = [],[],[],[]
        
        instance = self.model.get_instance()
        optimizer = self.model.get_optimizer()
        loss_fn = self.model.get_loss_fn()

        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        max_workers = 4+4*(device.type != "cuda")

        instance.to(device)

        train_loader = DataLoader(train_ds, shuffle=True, num_workers=max_workers, persistent_workers=True,
                                  pin_memory = (device.type=="cuda"), batch_size=self.bsize)
        val_loader = DataLoader(val_ds, shuffle=True, num_workers=max_workers, persistent_workers=False,
                                pin_memory=(device.type=="cuda"), batch_size=self.bsize)
        start = time.time()

        val_loss = np.inf
        best_val_loss = np.inf

        checkpoint = {"epoch":0,
                      "best_model":instance.state_dict(),
                      "best_optimizer":optimizer.state_dict(),
                      "val_loss":val_loss}

        print("device:",device)
        
        for epoch in range(1,self.epochs+1):
                instance.train()
                print(f"Epoch: {epoch} of {self.epochs}")
                    
                total_samples = 0
                running_loss = 0
                for i, (xb,yb) in enumerate(train_loader):
                    xb = xb.to(device, non_blocking=True)
                    yb[0] = yb[0].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                        
                    with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                        logits = instance(xb)
                        loss = loss_fn(logits,yb[0])
                            
                    scaler.scale(loss).backward()
                    #torch.nn.utils.clip_grad_norm_(instance.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    total_samples+=self.bsize
                    running_loss += loss.item()*self.bsize
                    if (i+1)%self.bpdc == 0:
                        end = time.time()
                        train_loss = loss.item()
                        print(f"[batch {i+1}] samples: {total_samples}, Training Loss: {train_loss:.4f}")
                        print(f"   Time since start: {str(datetime.timedelta(seconds=(end-start)))}")
                        Epoch.append(epoch)
                        Batch.append(i+1)
                        Train_Loss.append(train_loss)
                             
                epoch_loss = running_loss/total_samples
                print(f"--m-Epoch {epoch} done.")
                print(f"   Training Loss: {epoch_loss:.4f}")
                val_loss = self.validate(epoch, instance, device, loss_fn, val_loader, mode="eval", target="full")
                print(f"   Validation Loss: {val_loss:.4f}")
                self.output_epochs = pd.concat([self.output_epochs,pd.DataFrame({"Epoch":[epoch],"Training Loss":[epoch_loss],"Validation Loss":[val_loss]})], ignore_index=True)

                #check early stopping:
                if self.early_stopping:
                    stop = self.check_early_stopping(val_loss, best_val_loss)
                    if stop:
                        self.patience -= 1
                        print(f"patience decreased: patience is now  {self.patience}")
                    else:
                        checkpoint = {"epoch":epoch,
                                      "best_model":instance.state_dict(),
                                      "best_optimizer":optimizer.state_dict(),
                                      "val_loss":val_loss}
                        self.patience = min(self.patience+1, 5)
                        best_val_loss = val_loss
                    if self.patience == 0 or np.isnan(train_loss) or np.isnan(val_loss):
                        print("Stopping early")
                        break
                
        self.output_batches = pd.concat([self.output_batches,pd.DataFrame({"Epoch":Epoch,"Batch":Batch,"Training Loss":Train_Loss})], ignore_index=True)

        target_folder = os.path.join("Results",out_folder)
        os.makedirs(target_folder, exist_ok=True)
        
        self.output_batches.to_csv(f"{target_folder}/Output_batches.csv",index=False)
        self.output_epochs.to_csv(f"{target_folder}/Output_epochs.csv",index=False)
        self.output_report.to_csv(f"{target_folder}/Output_report.csv",index=False)
        self.output_accuracy.to_csv(f"{target_folder}/Output_accuracy.csv",index=False)

        plots.epoch_loss(pd.read_csv(f"{target_folder}/Output_epochs.csv"),f"{target_folder}/Plots")
        plots.batch_loss(pd.read_csv(f"{target_folder}/Output_batches.csv"),self.bpdc, total_samples,f"{target_folder}/Plots")
        plots.report(pd.read_csv(f"{target_folder}/Output_report.csv"),f"{target_folder}/Plots")
        plots.epoch_accuracy(pd.read_csv(f"{target_folder}/Output_accuracy.csv"),f"{target_folder}/Plots")

        checkpoint["last_model"] = instance.state_dict()
        checkpoint["last_optimizer"] = optimizer.state_dict()
        torch.save(checkpoint,f"{target_folder}/{target_folder}.pth")

        output_dict = {"output_batches":self.output_batches,"output_epochs":self.output_epochs,
                       "output_report":self.output_report,"output_accuracy":self.output_accuracy}
        return output_dict

    
    def validate(self, epoch, instance, device, loss_fn, val_loader, 
                 mode="eval", target="batch"):
        """
        Req.: An epoch ID, a torch model, a device, a loss function and dataloader of the validation dataset are provided.
              Additionally -mode- can be set to either 'eval' or 'train' to set instance mode.
              Furthermore -target- can be specified to either validate -bpdc- random batches or the entire validation split.
        Eff.: The model is validated using the loss function and validation split, epoch serves as an ID in metrics.
        Res.: The loss of the validation is returned.
        """
        instance.eval()
        if mode == "train":
            instance.train()

        total_batches = len(val_loader)
        
        batch_ids = None
        if target == "batch":
            batch_ids = random.sample(range(total_batches), k=min(self.bpdc, total_batches))
        if target == "full":
            batch_ids = range(total_batches)
        
        total_loss = 0
        total_samples = 0

        all_preds, all_targets = [],[]
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                if i not in batch_ids:
                    continue
                
                xb = xb.to(device, non_blocking=True)
                yb[0] = yb[0].to(device, non_blocking=True)
                
                with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                    logits = instance(xb)
                    loss = loss_fn(logits,yb[0])
                    
                total_samples+=self.bsize
                total_loss+=loss.item()*self.bsize

                if self.multi_label == True:
                    pred = (logits.detach().cpu() >= 0).numpy().astype(float)
                else:
                    pred = logits.argmax(dim=1).detach().cpu()

                target = yb[0].detach().cpu()
                all_preds.append(pred)
                all_targets.append(target)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metric_map = self.generate_metrics(all_preds, all_targets, epoch)
        self.output_report = pd.concat([self.output_report,metric_map["report"]], ignore_index=True)
        self.output_accuracy = pd.concat([self.output_accuracy,metric_map["accuracy"]], ignore_index=True)
        
        return total_loss/total_samples

    def check_early_stopping(self, val_loss, best_val_loss):
        """
        Req.: The current validation loss and the best validation loss of the training are provided
        Eff.: -
        Res.: Return true if the current validation loss is at most the best validation loss minus
              the minimum improvement percentage of the best validation loss, otherwise false.
        """
        return val_loss > (1-self.min_delta)*best_val_loss

    def generate_metrics(self,preds,targets,epoch):
        """
        Req.: A prediction array, corresponding target array and an epoch ID are provided
        Eff.: -
        Res.: A metrics report is generated by collecting accuracy, precision, recall and f1-score
              of each output class. The report is returned as a dictionary.
        """
        metric_map = {}
        report = pd.DataFrame(metrics.classification_report(targets,preds,output_dict=True)).transpose()
        report.loc[:,"epoch"] = epoch
        metric_map["report"] = report
        accuracy = metrics.accuracy_score(targets,preds,normalize=True)
        metric_map["accuracy"] = pd.DataFrame({"epoch":[epoch],"accuracy":[accuracy]})
        return metric_map

    def set_model(self,model):
        """
        Req.: A torch model is provided as -model-
        Eff.: The training loop's model is set to the provided model
        Res.: -
        """
        self.model = model

    def get_model(self):
        """
        Req.: -
        Eff.: -
        Res.: The current torch model of the training loop is returned.
        """
        return self.model
