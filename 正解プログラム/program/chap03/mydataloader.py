#!/usr/bin/python
# -*- coding: sjis -*-

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import cv2

class PreProcess(object):
    def __init__(self):
        pass
    def __call__(self, x):
        x = cv2.resize(x, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = torch.from_numpy(x[:,:,(2,1,0)]).permute(2,0,1)
        return x

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
        filename = str(dirpath) + str(file) + ".jpg"
        x = cv2.imread(filename)
        y = ans[file]
        x = self.prepro(x)
        return (x,y)

prepro = PreProcess()
dirpath = './VOCdevkit/VOC2012/JPEGImages/'
ans = pickle.load(open('ans.pkl', 'rb'))
dataset = MyDataset(ans, dirpath, prepro)

def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    return images, targets


    

