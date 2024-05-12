#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Data setting

train_x = torch.from_numpy(np.load('train-x.npy'))
train_y = torch.from_numpy(np.load('train-y.npy'))
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
criterion = nn.MSELoss()

# Learn

for i in range(2000):
    output = model(train_x)
    loss = criterion(output,train_y)
    print(i, loss.item())
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



