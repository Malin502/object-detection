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

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cn1 = nn.Conv2d(1, 20, 5)  
        self.pool1 = nn.MaxPool2d(2)    
        self.cn2 = nn.Conv2d(20, 50, 5) 
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(3200, 10)
        self.fc2 = nn.Linear(3200, 2)    ## ここが追加        
    def forward(self, x):
        x = F.relu(self.cn1(x))
        x = self.pool1(x)
        x = F.relu(self.cn2(x))
        x = self.dropout(x)
        x = x.view(len(x), -1) # Flatten
        # return self.fc(x)   ## ここを以下
        return x              ## のように変更        

# Initialize model

model = MyModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Learn

n = len(y01)
bs = 200

model.train()
for j in range(10):
    idx = np.random.permutation(n)
    for i in range(0, n, bs):
        x = x0[idx[i:(i+bs) if (i+bs) < n else n]].to(device)
        y1 = y01[idx[i:(i+bs) if (i+bs) < n else n]].to(device)
        y2 = y02[idx[i:(i+bs) if (i+bs) < n else n]].to(device)        
        cnnx = model(x)
        out1 = model.fc(cnnx)
        out2 = model.fc2(cnnx)
        loss1 = criterion(out1,y1)
        loss2 = criterion(out2,y2)
        loss = loss1 + loss2
        print(j, i, loss.item())
        optimizer.zero_grad()                
        loss.backward()
        optimizer.step()
        
    outfile = "cnn2-" + str(j) + ".model"
    torch.save(model.state_dict(),outfile)        
    print(outfile," saved")





