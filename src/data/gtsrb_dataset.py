import os
import csv
import yaml
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class GTSRBDataset(Dataset):
    def __init__(self, dataset_config="config/dataset.yml", path_config="config/paths.yml", target="training", transform=None):
        with open(dataset_config, "r") as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg["dataset"]
        with open(path_config, "r") as f:
            cfg = yaml.safe_load(f)
        pth_cfg = cfg["data"]

        self.root_dir = pth_cfg[target]
        self.concept_csv = pth_cfg["concepts"]

        if transform is None:
            size = ds_cfg.get("image_size", 64)*ds_cfg["zoom_factor"]
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = transform

        # collect all image files + labels
        print("Collecting samples")
        if target=="training":
            self.samples = self._collect_from_training()
        elif target=="test":
            self.samples = self._collect_from_test()
        else:
            raise ValueError("Wrong mode selected. Either pick 'training' or 'test'.")
        # load concepts from CSV
        print("Collecting concepts")
        self.class_to_concepts = self._load_concept_file(self.concept_csv)
        print("Concepts collected")

    def _collect_from_training(self):
        samples = []
        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".ppm", ".jpg", ".png")):
                    samples.append((
                        os.path.join(folder_path, fname),
                        int(folder)   
                    ))
        return samples

    def _collect_from_test(self):
        path = os.path.join(self.root_dir, "GT-final_test.csv")
        gt = pd.read_csv(os.path.join(self.root_dir,"GT-final_test.csv"), sep=";")
        samples = []
        for fname in os.listdir(self.root_dir):
            if fname.lower().endswith((".ppm", ".jpg", ".png")):
                samples.append((
                    os.path.join(self.root_dir,fname),
                    int(gt[gt.Filename==fname]["ClassId"].iloc[0])))
        return samples

    def _load_concept_file(self, csv_path):
        concepts = {}

        # simple CSV reading
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  

            for row in reader:
                class_id = int(row[0])
                concept_vals = [int(v) for v in row[2:]]

                concepts[class_id] = torch.tensor(concept_vals, dtype=torch.float32)

        return concepts


    def __len__(self):
        # returns total number of images
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # load image
        img = Image.open(img_path).convert("RGB")

        # apply transform
        if self.transform is not None:
            img = self.transform(img)

        concept_vec = self.class_to_concepts[label]

        # return
        return img, (concept_vec, label)
