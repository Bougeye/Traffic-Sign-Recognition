
from torchvision import transforms, datasets, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])

sample = "data/GTSRB/Final_Training/Images/00000/00000_00000.ppm"

img = Image.open(sample).convert("RGB")
img_tr = transform(img)
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
print("Mean and std before normalize:")
print("Mean of the image:",mean)
print("Std of the image:",std)

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

img_nor = transform_norm(img)
mean, std = img_nor.mean([1,2]), img_nor.std([1,2])
print("Mean and std after normalize:")
print("Mean of the image:",mean)
print("Std of the image:",std)
