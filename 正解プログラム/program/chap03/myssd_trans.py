#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import cv2
import random
from multiboxloss import MultiBoxLoss
from mydataloader2 import *
from augmentations import SSDAugmentation

## (1) データの準備と設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 30
augment = SSDAugmentation()
prepro = PreProcess(augment)
dirpath = './VOCdevkit/VOC2012/JPEGImages/'
epoch_num = 30

## (2) モデルの定義
# from mynet import SSD
from mynet import *

## (3) モデルの生成，損失関数，最適化関数の設定
##   (3.1) モデルの生成
net = SSD()
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth'))
new_conf = make_conf(num_classes=4)
net.conf = new_conf
net.num_classes = 4
net.to(device)

for param in net.parameters():
    param.requires_grad = False

for param in net.conf.parameters():
    param.requires_grad = True

##   (3.2) 損失関数 の設定

optimizer = optim.SGD(net.parameters(),
                      lr=1e-3,momentum=0.9,
                      weight_decay=5e-4)

##   (3.3) 最適化関数の設定
criterion = MultiBoxLoss(num_classes=4,device=device)

## (4) 学習
net.train()
for ep in range(epoch_num):
    i = 0
    ans = pickle.load(open('ans2.pkl', 'rb'))
    dataset = MyDataset(ans, dirpath, prepro)
    dataloader = DataLoader(dataset,batch_size=batch_size,
                            shuffle=True, collate_fn=my_collate_fn)
    for xs, ys in dataloader:
        xs  = [ torch.FloatTensor(x) for x in xs ]
        images = torch.stack(xs, dim=0)
        images = images.to(device)
        targets  = [ torch.FloatTensor(y).to(device) for y in ys ]
        outputs = net(images)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        print(i, loss_l.item(), loss_c.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
        optimizer.step()
        loss_l, loss_c  = 0, 0
        xs, ys, bc = [], [], 0
        i += 1
    outfile = "ssdTr-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
    print(outfile," saved")
