#!/usr/bin/python
# -*- coding: sjis -*-

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import cv2

from augmentations import SSDAugmentation

class PreProcess(object):
    def __init__(self,augment):
        self.augment = augment
    def __call__(self, img, tch):
        x1 = tch[:,:4]
        x2 = tch[:,4]
        y0, y1, y2  = self.augment(img, x1, x2)
        img = y0[:, :, ::-1].copy()
        img = img.transpose(2, 0, 1)
        y3 = y2.reshape(len(y2),1)
        an = np.concatenate([y1, y3], 1)
        return (img, an)        
   
class MyDataset(Dataset):
    def __init__(self, ansdic, dirpath, prepro):
        self.ans = ansdic
        self.dirpath = dirpath
        self.files = list(self.ans.keys())
        self.prepro = prepro
    def __len__(self):
        return len(list(self.ans.keys()))
    def __getitem__(self, idx):
        file = self.files[idx]
        filename = str(self.dirpath) + str(file) + ".jpg"
        x = cv2.imread(filename)
        y = self.ans[file]
        x, y  = self.prepro(x,y)
        return (x,y)

def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    return images, targets

