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
x0 = dataset.data.reshape(60000,1,28,28) / 255.0
y01 = dataset.targets
y02 = dataset.targets
for i in range(len(y02)):
    if (y02[i] in [6, 8, 9]):
        y02[i] = 1
    else:
        y02[i] = 0

# Define model

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cn1 = nn.Conv2d(1, 20, 5)  
        self.pool1 = nn.MaxPool2d(2)    
        self.cn2 = nn.Conv2d(20, 50, 5) 
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(3200, 10)  
    def forward(self, x):
        x = F.relu(self.cn1(x))
        x = self.pool1(x)
        x = F.relu(self.cn2(x))
        x = self.dropout(x)   
        x = x.view(len(x), -1) 
        return self.fc(x)

class MyCNN2(nn.Module):
    def __init__(self):
        super(MyCNN2, self).__init__()
        self.fc = nn.Linear(3200, 2)  
    def forward(self, x):
        return self.fc(x)

# Initialize model

model1 = MyCNN().to(device)
model2 = MyCNN2().to(device)
optimizer = optim.SGD([
    {'params': model1.parameters()},
    {'params': model2.parameters()}
], lr=0.01)
criterion = nn.CrossEntropyLoss()

# Learn

def fwd(model, x):
    x = F.relu(model.cn1(x))
    x = model.pool1(x)
    x = F.relu(model.cn2(x))
    x = model.dropout(x)   
    x = x.view(len(x), -1) 
    return x

n = len(y01)
bs = 200

model1.train()
model2.train()

for j in range(10):
    idx = np.random.permutation(n)
    for i in range(0, n, bs):
        x = x0[idx[i:(i+bs) if (i+bs) < n else n]].to(device)
        y1 = y01[idx[i:(i+bs) if (i+bs) < n else n]].to(device)
        y2 = y02[idx[i:(i+bs) if (i+bs) < n else n]].to(device)                
        cnnx = fwd(model1, x)   
        out1 = model1.fc(cnnx)   
        out2 = model2(cnnx)      
        loss1 = criterion(out1,y1)   
        loss2 = criterion(out2,y2)   
        loss = loss1 + loss2
        print(j, i, loss1.item(),loss2.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    outfile1 = "cnn3-m1-" + str(j) + ".model"
    torch.save(model1.state_dict(),outfile1)        
    print(outfile1," saved")

    outfile2 = "cnn3-m2-" + str(j) + ".model"
    torch.save(model2.state_dict(),outfile2)
    print(outfile2," saved")
