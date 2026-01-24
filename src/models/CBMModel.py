import os
import sys
import yaml
import torch
import numpy

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

class CBMModel:
    def __init__(self, train_cfg, ds_cfg, pth_cfg, model_variant="S", lr_stage1=0.0001, lr_stage2=0.001):
        self.tr_cfg = train_cfg
        self.ds_cfg = ds_cfg
        self.pth_cfg = pth_cfg
        self.model_stage_1 = stage_1_models.ENV2(model_variant=model_variant, lr=self.tr_cfg["stage_1"]["lr"], optimizer=self.tr_cfg["stage_1"]["optimizer"])
        self.model_stage_2 = stage_2_models.LabelModel(lr=self.tr_cfg["stage_2"]["lr"], optimizer=self.tr_cfg["stage_2"]["optimizer"],
                                                       layers=3,hidden_dim=128,hidden_dim2=64)

    def train(self, random_seed=69, epochs_stage1=20, epochs_stage2=20, early_stopping=True):
        ds_train, ds_val = self._setup_splits(random_seed)
        self.train_stage_1(ds_train, ds_val, epochs_stage1, early_stopping, "stage_1")
        ds_train = self.forward_stage_1(ds_train)
        ds_val = self.forward_stage_1(ds_val)
        self.train_stage_2(ds_train, ds_val, epochs_stage2, early_stopping, "stage_2")

    def _setup_splits(self, random_seed):
        dataset_1 = GTSRBDataset(self.ds_cfg, self.pth_cfg)
        
        total_samples = len(dataset_1)
        labels = [dataset_1[i][1][1] for i in range(len(dataset_1))]
        idx = list(range(len(dataset_1)))

        train_idx, val_idx = train_test_split(list(range(len(dataset_1))), test_size=0.2, random_state=random_seed, stratify=labels)
        train_idx = [idx[i] for i in train_idx]
        val_idx = [idx[i] for i in val_idx]

        ds_train = Subset(dataset_1, train_idx)
        ds_val = Subset(dataset_1, val_idx)
        return ds_train, ds_val 

    def train_stage_1(self, ds_train, ds_val, epochs_stage1, early_stopping, out_folder):
        train_1 = Training_Loop(epochs=epochs_stage1, bsize=self.tr_cfg["stage_1"]["bsize"],
                                bpdc=self.tr_cfg["stage_1"]["bpdc"], patience=self.tr_cfg["stage_1"]["patience"],
                                min_delta=self.tr_cfg["stage_1"]["min_delta"],early_stopping=early_stopping,
                                multi_label=True)
        train_1.set_model(self.model_stage_1)
        train_1.train(ds_train, ds_val, out_folder=out_folder)

    def train_stage_2(self, ds_train,ds_val, epochs_stage2, early_stopping, out_folder):
        train_2 = Training_Loop(epochs=epochs_stage2, bsize=self.tr_cfg["stage_2"]["bsize"],
                                bpdc=self.tr_cfg["stage_2"]["bpdc"], patience=self.tr_cfg["stage_2"]["patience"],
                                min_delta=self.tr_cfg["stage_2"]["min_delta"],early_stopping=early_stopping,
                                multi_label=False)
        train_2.set_model(self.model_stage_2)
        train_2.train(ds_train, ds_val, out_folder=out_folder)

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

    def experiment_training(self, model_variant="S", random_seed=42, epochs_stage1=20, epochs_stage2=20, early_stopping=True):
        ###Learning rate optimization
        gtsrb_ds = GTSRBDataset(self.ds_cfg, self.pth_cfg)
        ds_train, ds_val = self._setup_splits(random_seed)
        train_1 = Training_Loop(epochs=epochs_stage1, bsize=self.tr_cfg["stage_1"]["bsize"],
                                bpdc=self.tr_cfg["stage_1"]["bpdc"], patience=self.tr_cfg["stage_1"]["patience"],
                                min_delta=self.tr_cfg["stage_1"]["min_delta"],early_stopping=early_stopping,
                                multi_label=True)
        train_1.set_model(self.model_stage_1)
        train_1.train(ds_train, ds_val, out_folder="stage_1")
        ds_train = self.forward_stage_1(ds_train)
        ds_val = self.forward_stage_1(ds_val)
        for learning_rate in [0.001,0.0001,0.00001]:
            for layer in [1,2,3]:
                for dim in [32,64,128,256]:
                    model_stage_2 = stage_2_models.LabelModel(lr=learning_rate, optimizer=self.tr_cfg["stage_2"]["optimizer"],
                                                              layers=layer,hidden_dim=dim,hidden_dim2=int(dim/2))
                    train_2 = Training_Loop(epochs=epochs_stage2, bsize=self.tr_cfg["stage_2"]["bsize"],
                                        bpdc=self.tr_cfg["stage_2"]["bpdc"], patience=self.tr_cfg["stage_2"]["patience"],
                                        min_delta=self.tr_cfg["stage_2"]["min_delta"],early_stopping=early_stopping,
                                        multi_label=False)
                    train_2.set_model(model_stage_2)
                    train_2.train(ds_train, ds_val, out_folder=f"stage_2/lr-{learning_rate}/layers-{layer}/dim-{dim}")
