import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
import pickle
import random
import torch.nn as nn

from mydataloader import *


##(1) データの準備と設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 30
prepro = PreProcess()
dirpath = './VOCdevkit/VOC2012/JPEGImages/'
ans = pickle.load(open('ans.pkl', 'rb'))
dataset = MyDataset(ans, dirpath, prepro)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
epoch_num = 15


##(2) ネットワークの構築
from mynet import SSD

##(3) モデルの生成、損失関数、最適化関数の設定
    ##(3-1) モデルの生成
net = SSD()
vgg_weights = torch.load('vgg16_reducedfc.pth')
net.vgg.load_state_dict(vgg_weights)

net.to(device)
    
    ##(3-2) 損失関数の設定
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    ##(3-3) 最適化関数の設定
from multiboxloss import MultiBoxLoss
criterion = MultiBoxLoss(device=device)

##(4) 学習
net.train()
for ep in range(epoch_num):
    i = 0
    for xs, ys in dataloader:
        xs = [torch.FloatTensor(x) for x in xs]
        images = torch.stack(xs, dim=0)
        images = images.to(device)
        targets = [torch.FloatTensor(y).to(device) for y in ys]
        outputs = net(images)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        print(i, loss_l.item(), loss_c.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
        optimizer.step()
        loss_l, loss_c = 0, 0
        xs, ys, bc = [], [], 0
        
        i += 1
        
##(5) モデルの保存(各エポックでモデルを保存)
    outfile = 'ssd1_' + str(ep) + '.model'
    torch.save(net.state_dict(), outfile)
    print(outfile, "saved")

