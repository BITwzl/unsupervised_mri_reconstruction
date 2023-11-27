import os
import numpy as np
import torch
import h5py
import scipy.io as sio
import cv2

class ProbMask:
    def __init__(self, mask_arg):
        self.type=mask_arg["type"]
        if mask_arg["path"].endswith("npy"):
            self.mask_prob=np.load(mask_arg["path"])
        else:
            try:
                self.mask_prob=np.array(sio.loadmat(mask_arg["path"])['mask'])
            except:
                mask = h5py.File(mask_arg["path"],"r")
                self.mask_prob=np.array(mask['mask_matrix'])
        self.fixed_mask={}
        if mask_arg["repath"].endswith("npy"):
            self.remask=np.load(mask_arg["repath"])
        else:
            try:
                self.remask=np.array(sio.loadmat(mask_arg["repath"])['mask'])
            except:
                mask = h5py.File(mask_arg["repath"],"r")
                self.remask=np.array(mask['mask_matrix'])
        self.remask=torch.tensor(cv2.resize(self.remask,self.mask_prob.shape,interpolation=cv2.INTER_NEAREST)).float()
    def sample(self,item):
        prob=torch.from_numpy(self.mask_prob).float()
        if self.type=="dynamic":
            mask = (torch.rand(prob.shape)<prob).float()
        elif self.type=="fixed":
            if item not in self.fixed_mask:
                self.fixed_mask[item]=(torch.rand(prob.shape)<prob).float()
            mask = self.fixed_mask[item]
        else:
            raise Exception("Unkown mask type {}".format(self.type))
        return mask