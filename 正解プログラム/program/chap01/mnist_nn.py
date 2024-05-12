#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
import numpy as np

# GPU/CPU device setting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data setting

dataset = datasets.MNIST('./data', train=True, download=True)
x0 = dataset.data.reshape(60000,-1) / 255.0
y0 = dataset.targets

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

model = MyNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Learn

n = len(y0)
bs = 200

for j in range(10):
    idx = np.random.permutation(n)
    for i in range(0, n, bs):
        x = x0[idx[i:(i+bs) if (i+bs) < n else n]].to(device)
        y = y0[idx[i:(i+bs) if (i+bs) < n else n]].to(device)        
        output = model(x)
        loss = criterion(output,y)
        print(j, i, loss.item())
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

    outfile = "nn-" + str(j) + ".model"
    torch.save(model.state_dict(),outfile)    
    print(outfile," saved")
    


