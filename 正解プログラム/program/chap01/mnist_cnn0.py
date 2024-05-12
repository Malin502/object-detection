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
y0 = dataset.targets

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
#        x = self.dropout(x)
        x = x.view(len(x), -1) # Flatten
        return self.fc(x)

# Initialize model

model = MyCNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Learn

n = len(y0) 
bs = 200

model.train()
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
        
    outfile = "cnn0-" + str(j) + ".model"
    torch.save(model.state_dict(),outfile) 
    print(outfile," saved")





