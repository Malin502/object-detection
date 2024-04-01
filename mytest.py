import sys
import torch
import torch.nn as nn
import numpy as np
import cv2

from matplotlib import pyplot as plt
from mynet import SSD
from myfunctions import decode, nms

argvs = sys.argv
argc = len(argvs)

net = SSD(phase = 'test')
net.load_state_dict(torch.load(argvs[2]))

image = cv2.imread(argvs[1], cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#<-- これは最後の出力画像の時に使う

x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0) #平均画像を引く
x = x[:, :, ::-1].copy() #BGR -> RGB
x = x.transpose(2, 0, 1) #[300, 300, 3] -> [3, 300, 300]
x = torch.from_numpy(x)
x = x.unsqueeze(0)

net.eval()
with torch.no_grad():
    y = net(x)
    
labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
          'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
          'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

plt.figure(figsize=(10, 6))

