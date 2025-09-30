import os
import torch
import torchvision 
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class KodakDataset(torch.utils.data.Dataset):
    def __init__(self, kodak_root):
        self.img_dir = kodak_root
        self.img_fname = os.listdir(self.img_dir)
        print(self.img_fname)
    def __len__(self):
        return len(self.img_fname)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_fname[idx])
        print(img_path)
        image = Image.open(img_path)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image))
        image = image.to(dtype=torch.float32) / 255.0
        return image
