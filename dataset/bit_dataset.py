from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import cv2
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import nibabel as nib
import scipy.io as scio
import h5py
from mri_tools import *
from dataset.prob_mask import *

def key(x):
    x=os.path.basename(x).split(".")[0]
    d=x.split("_")
    return int(d[0])*40+int(d[1])
    
class CustomBITDataSet(Dataset):
    def __init__(self, hr_img_path, mask_configs):
        self.hr_img_path=hr_img_path
        self.transforms = transforms.Compose([transforms.ToTensor(),])
        self.hr_img_list= []

        for file in os.listdir(self.hr_img_path):
            if file.endswith("png"):
                hr_img_path = os.path.join(self.hr_img_path, file)
                self.hr_img_list.append(hr_img_path)
        
        self.masks = [ProbMask(mask_config) for mask_config in mask_configs]

    def __getitem__(self, item):
        hr_img = Image.open(self.hr_img_list[item]).convert('L')
        hr_img_np = np.array(hr_img)

        hr_img = self.transforms(hr_img_np).squeeze()
        full_kspace=fft2(hr_img)
            
        return {0:full_kspace.unsqueeze(0), 1:torch.ones_like(full_kspace.unsqueeze(0)), 2:[mask.sample(item) for mask in self.masks], 3:self.hr_img_list[item], 4:0, 5:[mask.remask for mask in self.masks]}

    def __len__(self):
        return len(self.hr_img_list)


def get_file(path):
    files = os.listdir(path)
    list = []
    for file in files:
        if not os.path.isdir(path + file):
            f_name = str(file)
            #    print(f_name)
            tr = '\\'
            filename = path + tr + f_name
            list.append(filename)
    return (list)
