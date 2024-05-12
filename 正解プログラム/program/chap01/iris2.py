#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Data setting

train_x = torch.from_numpy(np.load('train-x.npy'))
# train_y = torch.from_numpy(np.load('train-y.npy'))
train_y = torch.from_numpy(np.load('train-y0.npy'))
test_x = torch.from_numpy(np.load('test-x.npy'))
test_y = torch.from_numpy(np.load('test-y.npy'))

# Define model

class MyIris(nn.Module):
    def __init__(self):
        super(MyIris, self).__init__()
        self.l1=nn.Linear(4,6)
        self.l2=nn.Linear(6,3)
    def forward(self,x):
        h1 = torch.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2
     
# model generate, optimizer and criterion setting

model = MyIris()
optimizer = optim.SGD(model.parameters(),lr=0.1)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

# Learn

n = 75
bs = 25
for i in range(1000):
    idx = np.random.permutation(n)
    for j in range(0,n,bs):
        xtm = train_x[idx[j:(j+bs) if (j+bs) < n else n]]
        ytm = train_y[idx[j:(j+bs) if (j+bs) < n else n]]
        output = model(xtm)
        loss = criterion(output,ytm)
        print(i, j, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# torch.save(model.state_dict(),'my_iris.model')     ## モデルの保存
# model.load_state_dict(torch.load('my_iris.model')) ## モデルの呼び出し
    
# Test

model.eval()
with torch.no_grad():
    output1 = model(test_x)
    ans = torch.argmax(output1,1)
    print(((test_y == ans).sum().float() / len(ans)).item())



