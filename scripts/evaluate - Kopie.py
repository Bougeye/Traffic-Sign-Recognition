from torch.utils.data import TensorDataset, DataLoader, Subset
import sklearn.metrics as metrics
import os
import sys
import yaml
import torch
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.gtsrb_dataset import GTSRBDataset
#from src.data.concepts_dataset import ConceptsDataset
#from src.data.test_dataset import TestDataset
from src.models.ENV2 import ENV2 as Stage_1
from src.models.LabelModel import LabelModel as Stage_2
import src.utils.plots as plots

class evaluate:
    def __init__(self, pth_model_1=None, pth_model_2=None, mode="best_model",
                 pth_data_1=None, pth_data_2=None, model_variant="M", layers=1):
        with open("config/training.yml", "r") as f:
            self.tr_cfg = yaml.safe_load(f)
        with open("config/dataset.yml", "r") as f:
            self.ds_cfg = yaml.safe_load(f)
        with open("config/paths.yml","r") as f:
            self.pth_cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage_1 = Stage_1(model_variant=model_variant)
        self.stage_2 = Stage_2(layers=3,hidden_dim=128,hidden_dim2=64)
        self.instance_1 = self.stage_1.get_instance()
        self.instance_2 = self.stage_2.get_instance()
        self.ts = str(datetime.datetime.now())[:10]+"_"+time.strftime("%H:%M:%S").replace(":","-")

        if pth_model_1 is not None:
            print("Fetching stage 1 from specified file path...")
            try:
                self.instance_1 = torch.load(pth_model_1, map_location=self.device)
            except:
                raise ValueError("Could not find model path for stage 1."
                                 "Please make sure model path relative to project root is correct.")
        else:
            print("Fetching stage 1 from standard file path...")
            tmp = torch.load(os.path.join(self.pth_cfg["registry"],"stage_1.pth"), map_location=self.device)
            self.instance_1.load_state_dict(tmp[mode])
            self.instance_1.to(self.device)
            self.instance_1.eval()
        if pth_model_2 is not None:
            print("Fetching stage 2 from specified file path...")
            try:
                self.instance_2 = torch.load(pth_model_2, map_location=self.device)
            except:
                raise ValueError("Could not find model path for stage 2."
                                 "Please make sure model path relative to project root is correct.")
        else:
            print("Fetching stage 2 from standard file path...")
            tmp = torch.load(os.path.join(self.pth_cfg["registry"],"stage_2.pth"), map_location=self.device)
            self.instance_2.load_state_dict(tmp[mode])
            self.instance_2.to(self.device)
            self.instance_2.eval()
        self.max_workers = 4+4*(self.device.type != "cuda")
        self.bsize = self.tr_cfg["stage_1"]["bsize"]

    def evaluate_set(self,mode="training"):
        start = time.time()
        print("Intializing validation split...")
        dataset = GTSRBDataset(dataset_config="config/dataset.yml",
                                   path_config="config/paths.yml",
                                   target=mode)
        labels = [dataset[i][1][1] for i in range(len(dataset))]
        if mode=="training":
            _, idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=69, stratify=labels)
            dataset = Subset(dataset, idx)
        loader = DataLoader(dataset, shuffle=True, num_workers=self.max_workers, persistent_workers=False,
                                pin_memory=(self.device.type=="cuda"), batch_size=self.bsize)
        
        print("Running evaluation on validation split...")
        all_concept_preds, all_label_preds, all_concept_targets, all_label_targets = [],[],[],[]
        with torch.no_grad():
            for i, (xb,yb) in enumerate(loader):
                xb = xb.to(self.device, non_blocking=True)
                yb[0] = yb[0].to(self.device, non_blocking=True)
                yb[1] = yb[1].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=(self.device.type=="cuda")):
                    logits = self.instance_1(xb)
                    concept_pred = torch.tensor((logits.detach().cpu() >= 0).numpy().astype(float), dtype=torch.float32)
                    logits = self.instance_2(concept_pred)
                    label_pred = logits.argmax(dim=1).detach().cpu()
                    concept_target = yb[0].detach().cpu()
                    label_target = yb[1].detach().cpu()
                
                all_concept_preds.append(concept_pred)    
                all_label_preds.append(label_pred)
                all_concept_targets.append(concept_target)
                all_label_targets.append(label_target)
                
        all_concept_preds = np.concatenate(all_concept_preds)
        all_label_preds = np.concatenate(all_label_preds)
        all_concept_targets = np.concatenate(all_concept_targets)
        all_label_targets =  np.concatenate(all_label_targets)
        concept_report = pd.DataFrame(metrics.classification_report(all_concept_targets, all_concept_preds, output_dict=True)).transpose()
        label_report = pd.DataFrame(metrics.classification_report(all_label_targets, all_label_preds, output_dict=True)).transpose()
        concept_accuracy = metrics.accuracy_score(all_concept_targets, all_concept_preds, normalize=True)
        label_accuracy = metrics.accuracy_score(all_label_targets, all_label_preds, normalize=True)
        os.makedirs(f"reports/Eval-{self.ts}", exist_ok=True)
        if mode=="training":
            concept_report.to_csv(f"reports/Eval-{self.ts}/val_concept_report.csv")
            label_report.to_csv(f"reports/Eval-{self.ts}/val_label_report.csv")
        else:
            concept_report.to_csv(f"reports/Eval-{self.ts}/test_concept_report.csv")
            label_report.to_csv(f"reports/Eval-{self.ts}/test_label_report.csv")
        
        print("Concept accuracy: ",concept_accuracy)
        print("Label accuracy: ",label_accuracy)
        end = time.time()
        print("Time spent: ",end-start)

        
    def evaluate_on_test(self):
        self.evaluate_set(mode="test")

    def evaluate_on_val(self):
        self.evaluate_set(mode="training")

if __name__ == "__main__":
    p_map = {"pth_model_1":None,
             "pth_model_2":None,
             "mode":"best_model",
             "pth_data_1":None,
             "pth_data_2":None,
             "model_variant":"S",
             "layers":3}
    
    e = evaluate(pth_model_1=p_map["pth_model_1"], pth_model_2=p_map["pth_model_2"], mode=p_map["mode"],
                 pth_data_1=p_map["pth_data_1"], pth_data_2=p_map["pth_data_2"],
                 model_variant=p_map["model_variant"], layers=p_map["layers"])
    e.evaluate_on_test()
    e.evaluate_on_val()   
