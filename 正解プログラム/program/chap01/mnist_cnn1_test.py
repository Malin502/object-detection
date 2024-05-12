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
xtest = dataset.data.reshape(10000,1,28,28) / 255.0
ytest = dataset.targets

# Define model

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cn1 = nn.Conv2d(1, 20, 5)  
        self.pool1 = nn.MaxPool2d(2)    
        self.cn2 = nn.Conv2d(20, 50, 5) 
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(50 * 8 * 8, 10)
    def forward(self, x):
        x = F.relu(self.cn1(x))
        x = self.pool1(x)
        x = F.relu(self.cn2(x))
        x = self.dropout(x)
        x = x.view(len(x), -1) # Flatten
        return self.fc(x)

# Initialize model

model = MyCNN()

# Load model

model.load_state_dict(torch.load(argvs[1]))

# Test

model.eval()
with torch.no_grad():
    y1 = model(xtest)
    ans = torch.argmax(y1,1)
    print(((ytest == ans).sum().float()/len(ans)).item())
