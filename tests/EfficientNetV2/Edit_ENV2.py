import torch
import torchvision
from torchvision import transforms, datasets, models
import os

model = models.efficientnet_v2_m(weights="EfficientNet_V2_M_Weights.DEFAULT")
if not os.path.exists("EfficientNet_v2_m.pth"):
    torch.save(model, "EfficientNet_v2_m.pth")
print(list(model.children()))

