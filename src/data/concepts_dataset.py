import os
import csv
import yaml
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ConceptsDataset(Dataset):
    def __init__(self, X_in, y_in):
        self.X_in = X_in
        self.y_in = y_in

    def __len__(self):
        # returns total number of images
        return len(self.X_in)

    def __getitem__(self, idx):
        xb = torch.tensor(self.X_in[idx],dtype=torch.float32)
        yb = torch.tensor(self.y_in[idx],dtype=torch.long)
        return xb, (yb,yb)
