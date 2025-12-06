import sys
import os

# Ensure project root is in PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from torch.utils.data import DataLoader
from src.data.gtsrb_dataset import GTSRBDataset


ds = GTSRBDataset("config/dataset.yml")
loader = DataLoader(ds, batch_size=8, shuffle=True)

for batch in loader:
    imgs, (concepts, labels) = batch
    print(imgs.shape, concepts.shape, labels.shape)
    break
