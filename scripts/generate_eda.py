import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import yaml

class EDA:
    def __init__(self, dataset_config="config/dataset.yml", paths_config="config/paths.yml"):
        with open(dataset_config,"r") as f:
            self.ds_cfg = yaml.safe_load(f)
        with open(paths_config,"r") as f:
            self.pth_cfg = yaml.safe_load(f)
        self.concepts = pd.read_csv(self.pth_cfg["data"]["concepts"])
        self.out_pth = self.pth_cfg["reports"]["EDA"]

    def class_distribution(self):
        train_pth = self.pth_cfg["data"]["training"]
        test_pth = self.pth_cfg["data"]["test"]

        c_count = [0]*len(self.concepts)
        folders = sorted(os.listdir(train_pth))
        for i in range(len(folders)):
            folder_path = os.path.join(train_pth,folders[i])
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.lower().endswith(".ppm"):
                        c_count[i]+=1

        df = pd.DataFrame({"class":self.concepts["class_id"],"class_name":self.concepts["class_name"],"count":c_count})
        df.loc[:,"Set"] = "Training"
        g = sns.catplot(x="count",
                        y="class_name",
                        height=10,
                        aspect=1,
                        data=df,
                        kind="bar")
        fig = g.fig
        fig.savefig(os.path.join(self.out_pth,"class_distribution.png"))

    def concept_distribution(self):
        labels = [e for e in self.concepts][2:]
        counts = [self.concepts[e].sum() for e in labels]
        data = pd.DataFrame({"label":labels,"count":counts}).sort_values(by="count")
        g = sns.catplot(x="count",
                        y="label",
                        height=10,
                        aspect=1,
                        data=data,
                        kind="bar")
        fig = g.fig
        fig.savefig(os.path.join(self.out_pth,"concept_distribution.png"))

X = EDA()
#X.class_distribution()
X.concept_distribution()
