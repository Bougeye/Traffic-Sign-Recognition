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
from src.data.concepts_dataset import ConceptsDataset

from src.training.Training_Loop import Training_Loop
import src.models.ENV2 as stage_1_models
import src.models.LabelModel as stage_2_models

class train:
    def __init__(self, model_variant="M", early_stopping=True, configPath="config", Stage1_random_seed="", Stage2_random_seed="", Stage1_lr="", Stage2_lr="", Stage1_Epochs="", Stage2_Epochs=""):
		self.ConfigPath = configPath
        with open("{config}/training.yml".format(config=self.ConfigPath), "r") as f:
            self.tr_cfg = yaml.safe_load(f)
        self.early_stopping = early_stopping
		
		self.Stage1_random_seed = self.tr_cfg["stage_1"]["random_seed"]
		self.Stage2_random_seed = self.tr_cfg["stage_2"]["random_seed"]
		
		if Stage1_random_seed != "":
			self.Stage1_random_seed = Stage1_random_seed
		if Stage2_random_seed != "":
			self.Stage2_random_seed = Stage2_random_seed
		
		self.Stage1_Epochs = self.tr_cfg["stage_1"]["epochs"]
		self.Stage2_Epochs = self.tr_cfg["stage_2"]["epochs"]	
		
		if Stage1_Epochs != "":
			self.Stage1_Epochs = Stage1_Epochs
		if Stage2_Epochs != "":
			self.Stage2_Epochs = Stage2_Epochs
			
		self.Stage1_lr = self.tr_cfg["stage_1"]["lr"]
		self.Stage2_lr = self.tr_cfg["stage_2"]["lr"]	
		
		if Stage1_lr != "":
			self.Stage1_lr = Stage1_lr
		if Stage2_lr != "":
			self.Stage2_lr = Stage2_lr
			
		self.model_stage_1 = stage_1_models.ENV2(model_variant=model_variant, lr=self.Stage1_lr, optimizer=self.tr_cfg["stage_1"]["optimizer"])
        self.model_stage_2 = stage_2_models.LabelModel(lr=self.Stage2_lr, optimizer=self.tr_cfg["stage_2"]["optimizer"],
                                                       layers=3,hidden_dim=128,hidden_dim2=64)

    def train(self):
        self.train_1 = Training_Loop(epochs=self.Stage1_Epochs, bsize=self.tr_cfg["stage_1"]["bsize"],
                                     bpdc=self.tr_cfg["stage_1"]["bpdc"], patience=self.tr_cfg["stage_1"]["patience"],
                                     min_delta=self.tr_cfg["stage_1"]["min_delta"],early_stopping=self.early_stopping,
                                     multi_label=True, random_seed=self.Stage1_random_seed)
        self.train_2 = Training_Loop(epochs=self.Stage2_Epochs, bsize=self.tr_cfg["stage_2"]["bsize"],
                                     bpdc=self.tr_cfg["stage_2"]["bpdc"], patience=self.tr_cfg["stage_2"]["patience"],
                                     min_delta=self.tr_cfg["stage_2"]["min_delta"],early_stopping=self.early_stopping,
                                     multi_label=False, random_seed=self.Stage2_random_seed)
        
        self.train_1.set_model(self.model_stage_1)
        self.train_2.set_model(self.model_stage_2)
        
        dataset_1 = GTSRBDataset(dataset_config="{config}/dataset.yml".format(config=self.ConfigPath),
                                 path_config="{config}/paths.yml"format(config=self.ConfigPath))
        self.train_1.train(dataset_1, out_folder="stage_1")
        
        dataset_2 = self.forward_stage_1()
        self.train_2.train(dataset_2, out_folder="stage_2")

    def forward_stage_1(self):
        X_in, y_in = [],[]
        instance = self.model_stage_1.get_instance()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        gtsrb_ds = GTSRBDataset(dataset_config="{config}/dataset.yml".format(config=self.ConfigPath),
                                path_config="{config}/paths.yml".format(config=self.ConfigPath))
        loader = DataLoader(gtsrb_ds, shuffle=True, num_workers=8, persistent_workers=True,
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
        gtsrb_ds = GTSRBDataset(dataset_config="config/dataset.yml",
                                path_config="config/paths.yml")
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
