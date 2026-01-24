from torch.utils.data import TensorDataset, DataLoader, Subset
import sklearn.metrics as metrics
import os
import pandas as pd
import sys
import yaml
import torch
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.gtsrb_dataset import GTSRBDataset
from src.data.concepts_dataset import ConceptsDataset
from src.data.test_dataset import TestDataset
from src.models.ENV2 import ENV2 as Stage_1
from src.models.LabelModel import LabelModel as Stage_2
import src.utils.plots as plots
import src.utils.reports as reports

class evaluate:
    def __init__(self, pth_model_1=None, pth_model_2=None, last=False,
                 pth_data=None, model_variant="M", layers=1):
        with open("config/training.yml", "r") as f:
            self.tr_cfg = yaml.safe_load(f)
        with open("config/dataset.yml", "r") as f:
            self.ds_cfg = yaml.safe_load(f)
        with open("config/paths.yml","r") as f:
            self.pth_cfg = yaml.safe_load(f)
        if last==False:
            mode="best_model"
        else:
            mode="last_model"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage_1 = Stage_1(model_variant=model_variant)
        self.stage_2 = Stage_2(layers=self.tr_cfg["stage_2"]["layers"],hidden_dim=self.tr_cfg["stage_2"]["hidden_dim"],hidden_dim2=int(self.tr_cfg["stage_2"]["hidden_dim"]/2))
        self.instance_1 = self.stage_1.get_instance()
        self.instance_2 = self.stage_2.get_instance()
        self.ts = str(datetime.datetime.now())[:10]+"_"+time.strftime("%H:%M:%S").replace(":","-")

        if pth_model_1 is not None:
            print("Fetching stage 1 from specified file path...")
            try:
                #self.instance_1 = torch.load(pth_model_1, map_location=self.device)
                tmp = torch.load(pth_model_1, map_location=self.device)
                self.instance_1.load_state_dict(tmp[mode])
                self.instance_1.to(self.device)
                self.instance_1.eval()
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
                #self.instance_2 = torch.load(pth_model_2, map_location=self.device)
                tmp = torch.load(pth_model_2, map_location=self.device)
                self.instance_2.load_state_dict(tmp[mode])
                self.instance_2.to(self.device)
                self.instance_2.eval()
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

    def evaluate_on(self, target="training", datapath=None, random_seed=42):
        start = time.time()
        mode = target
        if mode == "training":
            mode = "val"
        print(f"Intializing {mode} split...")
        dataset = GTSRBDataset(self.ds_cfg,
                               self.pth_cfg,
                               target=target,
                               datapath=datapath)
        if target=="training":
            labels = [dataset[i][1][1] for i in range(len(dataset))]
            _, idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=random_seed, stratify=labels)
            dataset = Subset(dataset, idx)
        loader = DataLoader(dataset, num_workers=self.max_workers, persistent_workers=True,
                                pin_memory=(self.device.type=="cuda"), batch_size=self.bsize)
        
        print(f"Running evaluation on {mode} split...")
        all_concept_preds, all_label_preds, all_concept_targets, all_label_targets = [],[],[],[]
        wrongs = []
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
                for i in range(len(label_pred)):
                    if label_pred[i] != label_target[i]:
                        wrongs.append({"input":xb[i],"label_pred":label_pred[i],"label_target":label_target[i],
                                       "concept_pred":concept_pred[i],"concept_target":concept_target[i]})   
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
        acc = pd.DataFrame({"ConceptAcc":[concept_accuracy],"LabelAcc":[label_accuracy]})
        concept_cm = metrics.multilabel_confusion_matrix(all_concept_targets,all_concept_preds)
        label_cm = metrics.confusion_matrix(all_label_targets,all_label_preds)
        
        os.makedirs(f"reports/Eval-{self.ts}", exist_ok=True)
        
        #concept_report.to_csv(f"reports/Eval-{self.ts}/{mode}_concept_report.csv")
        #label_report.to_csv(f"reports/Eval-{self.ts}/{mode}_label_report.csv")
        acc.to_csv(f"reports/Eval-{self.ts}/{mode}_accuracy.csv")
        np.savetxt(f"reports/Eval-{self.ts}/{mode}_label_cm.txt", label_cm, fmt="%d")
        f = open(f"reports/Eval-{self.ts}/{mode}_concept_cm.txt","w")
        f.write("Per concept confusion matrix:\n\n{}\n".format(concept_cm))
        f.close()
        plots.class_distribution(all_label_preds,self.pth_cfg["data"]["training"],f"reports/Eval-{self.ts}")
        label_names = pd.read_csv(os.path.join(self.pth_cfg["data"]["root"],"class_map.csv"))
        concept_names = pd.read_csv(os.path.join(self.pth_cfg["data"]["root"],"concept_map.csv"))
        reports.misclassification_report(wrongs, concept_names, label_names, f"reports/Eval-{self.ts}/misclassification.pdf")
        reports.metrics_report(concept_report, concept_names, f"reports/Eval-{self.ts}/concept_report.pdf")
        reports.metrics_report(label_report, label_names, f"reports/Eval-{self.ts}/label_report.pdf")
        reports.label_cm_report(self.pth_cfg["data"]["training"],label_cm, label_names, f"reports/Eval-{self.ts}/label_cm.pdf")
        print("Concept accuracy: ",concept_accuracy)
        print("Label accuracy: ",label_accuracy)
        end = time.time()
        print("Time spent: ",end-start)


if __name__ == "__main__":        
	
    with open("config/training.yml", "r") as f:
        tr_cfg = yaml.safe_load(f)
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpth1", type=str, help="Path to Checkpoint of Stage1 Model.", default=None)
    parser.add_argument("--mpth2", type=str, help="Path to Checkpoint of Stage2 Model.", default=None)
    parser.add_argument("--datapth", type=str, help="Path to Dataset.", default=None)
    parser.add_argument("--target", type=str, help="Evaluate on Training or Test Set.", default="test")
    parser.add_argument("--rs", type=str, help="Seed for random number generators", default=tr_cfg["random_seed"])
    parser.add_argument("--last", type=bool, help="Grab last model (True) or best model (False)", default=False)
    args = parser.parse_args()
    
    e = evaluate(pth_model_1=args.mpth1, pth_model_2=args.mpth2, last=args.last, pth_data=args.datapth,
                 model_variant=tr_cfg["stage_1"]["model_variant"], layers=tr_cfg["stage_2"]["layers"])
    #e.evaluate_on_val()
    #e.evaluate_on(target="training")
    e.evaluate_on(target=args.target, datapath=args.datapth, random_seed=args.rs)
    
