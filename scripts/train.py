import os
import sys
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import src.training.Stage_1 as Stage_1

class train:
    def __init__(self, model_variant="M", early_stopping=True):
        with open("config/training.yml", "r") as f:
            tr_cfg = yaml.safe_load(f)
            
        Stage1 = Stage_1.Training_loop(model_variant=model_variant, epochs=tr_cfg["epochs"],
                                       lr=tr_cfg["lr"], bsize=tr_cfg["bsize"], optimizer=tr_cfg["optimizer"],
                                       folds=tr_cfg["folds"], bpdc=tr_cfg["bpdc"], early_stopping=early_stopping)

    def Train_First_Half(self):
        Stage1.train()

    def Experiment_Training(self, model_variant="M", early_stopping=True):
        with open("config/training.yml", "r") as f:
            tr_cfg = yaml.safe_load(f)
        ###Learning rate optimization
        for learning_rate in [0.01,0.001,0.0001,0.00001]:
            Stage1 = Stage_1.Training_loop(model_variant=model_variant, epochs=tr_cfg["epochs"],
                                           lr=learning_rate, bsize=tr_cfg["bsize"], optimizer=tr_cfg["optimizer"],
                                           folds=tr_cfg["folds"], bpdc=tr_cfg["bpdc"], early_stopping=early_stopping)
            Stage1.train(out_folder=f"lr-{learning_rate}")
        
x = train(model_variant="S", early_stopping=True)
x.Experiment_Training(model_variant="S", early_stopping=True)
