#!/usr/bin/python
# -*- coding: sjis -*-

import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from matplotlib import pyplot as plt
from models import Darknet
from utils.utils import *

argvs = sys.argv
argc = len(argvs)

net = Darknet('config/yolov3.cfg')
net.load_darknet_weights('weights/yolov3.weights')
labels = load_classes('data/coco.names')

image = cv2.imread(argvs[1], cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
x = cv2.resize(image, (416, 416)).astype(np.float32) / 255.0
x = x[:, :, ::-1].copy()  # BGR を RGB へ
x = x.transpose(2, 0, 1)  # [416,416,3]→ [3,416,416]
x = torch.from_numpy(x)
x = x.unsqueeze(0)

net.eval() 
with torch.no_grad():
    y = net(x)
    y2 = non_max_suppression(y, conf_thres=0.001, nms_thres=0.5)
    detections = y2[0]  
    ##  (x1, y1, x2, y2, object_conf, class_score, class_pred)

plt.figure(figsize=(10,6))
colors = plt.cm.hsv(np.linspace(0, 1, len(labels)+1)).tolist()
plt.imshow(rgb_image)
currentAxis = plt.gca()
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2) / 416.0

for rk in range(len(detections)):  
    # 確信度confが0.8 以上のボックスを表示
    if (detections[rk,4] >= 0.8): 
        score = detections[rk,4].item() 
        label_pred = int(detections[rk,6].item())
        label_name = labels[label_pred]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[rk,:4]*scale).cpu().numpy().astype(np.int32)
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[label_pred]  ## クラス毎に色が決まっている
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False,
                                            edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt,
                         bbox={'facecolor':color, 'alpha':0.5})
    else:
        # 信頼度でソートされているので、0.8 以下になったら、即、終わり
        break
plt.show()



