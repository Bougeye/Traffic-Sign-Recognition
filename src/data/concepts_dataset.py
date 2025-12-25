import os
import csv
import yaml
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ConceptsDataset(Dataset):
    def __init__(self, X_in, y_in):
        """
        Req.: Two array-like structures X_in and y_in are provided.
              It is assumed that X_in and y_in are of the same length.
        Eff.: A dataset based on X_in and y_in is initialized.
        Res.: -
        """
        self.X_in = X_in
        self.y_in = y_in

    def __len__(self):
        """
        Req.: -
        Eff.: -
        Res.: The minimum number of samples in the two arrays the dataset
              consists of, is returned.
        """
        return min(len(self.X_in),len(self.y_in))

    def __getitem__(self, idx):
        """
        Req.: -idx- is an index within the bounds of the dataset length.
        Eff.: -
        Res.: A torch network suitable representation of the sample
              at index -idx- is returned.
        """
        xb = torch.tensor(self.X_in[idx],dtype=torch.float32)
        yb = torch.tensor(self.y_in[idx],dtype=torch.float32)
        return xb,yb
