#!/usr/bin/python
# -*- coding: sjis -*-

import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mynet import SSD
from myfunctions import decode

argvs = sys.argv
argc = len(argvs)

net = SSD(phase='test',num_classes=4)
net.load_state_dict(torch.load(argvs[2]))

image = cv2.imread(argvs[1], cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0) # 平均画像を引く
x = x[:, :, ::-1].copy()  # BGR を RGB へ
x = x.transpose(2, 0, 1)  # [300,300,3]→ [3,300,300]
x = torch.from_numpy(x)
x = x.unsqueeze(0)

net.eval() 
with torch.no_grad():
    y = net(x)

# labels = ['aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor']

labels = ['cat', 'dog', 'person']

plt.figure(figsize=(10,6))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)
currentAxis = plt.gca()
detections = y.data
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):  
    j = 0
    # 確信度confが0.6以上のボックスを表示
    while detections[0,i,j,0] >= 0.1:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]  ## クラス毎に色が決まっている
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1
plt.show()


