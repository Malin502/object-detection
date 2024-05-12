#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import sys

argvs = sys.argv
argc = len(argvs)

# Data setting

dataset = datasets.MNIST('./data', train=False, download=True)
xtest = dataset.data.reshape(10000,-1) / 255.0
ytest = dataset.targets

# Define model

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.l1=nn.Linear(784,100)
        self.l2=nn.Linear(100,100)
        self.l3=nn.Linear(100,10)
    def forward(self,x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

# Initialize model

model = MyNN()

model.load_state_dict(torch.load(argvs[1]))

# Test
model.eval()
with torch.no_grad():
    y1 = model(xtest)
    ans = torch.argmax(y1,1)
    print(((ytest == ans).sum().float()/len(ans)).item())
