import os
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, dataset_config="config/dataset.yml", path_config="config/paths.yml", transform=None):
        with open(dataset_config, "r") as f:
            ds_cfg = yaml.safe_load(f)["dataset"]
        with open(path_config, "r") as f:
            pth_cfg = yaml.safe_load(f)["data"]

        
        self.root_dir = pth_cfg["test"]
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

        self.samples = []
        for fname in os.listdir(self.root_dir):
            if fname.lower().endswith((".ppm", ".jpg", ".png")):
                self.samples.append(os.path.join(self.root_dir,fname))

        labels = pd.read_csv(os.path.join(self.root_dir,"GT-final_test.csv"), sep=';')
        self.labels = torch.tensor(list(labels["ClassId"]), dtype=torch.float32)
    

    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img,(label,label)
