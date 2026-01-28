import sys
import os
import yaml
# Ensure project root is in PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
print(ROOT)
from torch.utils.data import DataLoader
from src.data.gtsrb_dataset import GTSRBDataset

with open("config/dataset.yml","r") as f:
    ds_cfg = yaml.safe_load(f)
with open("config/paths.yml", "r") as f:
    pth_cfg = yaml.safe_load(f)
ds = GTSRBDataset(ds_cfg,pth_cfg)
loader = DataLoader(ds, batch_size=8, shuffle=True)

for batch in loader:
    imgs, (concepts, labels) = batch
    print(imgs.shape, concepts.shape, labels.shape)
    print(imgs[0], concepts[0], labels[0])
    break
