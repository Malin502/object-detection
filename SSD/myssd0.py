import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
import pickle
import random
import torch.nn as nn


##(1) データの準備と設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 30
epoch_num = 15

ans  = pickle.load(open('ans.pkl','rb'))
files = list(ans.keys())
datanum = len(files)
dirpath = './VOCdevkit/VOC2012/JPEGImages/'

##(2) ネットワークの構築
from mynet import SSD

##(3) モデルの生成、損失関数、最適化関数の設定
    ##(3-1) モデルの生成
net = SSD()
net.to(device)
    
    ##(3-2) 損失関数の設定
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    ##(3-3) 最適化関数の設定
from multiboxloss import MultiBoxLoss
criterion = MultiBoxLoss(device=device)

##(4) 学習
net.train()
for ep in range(epoch_num):
    random.shuffle(files)
    xs, ys, bc = [], [], 0
    
    for i in range(datanum):
        file = files[i]
        if(bc < batch_size):
            filename = dirpath + file + '.jpg'
            image = cv2.imread(filename)
            
            x = cv2.resize(image, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0) #BGRの平均値を引く   
            x = torch.from_numpy(x[:,:,(2,1,0)]).permute(2,0,1) #BGRをRGBに並び替え
            y = ans[file]
            
            xs.append(torch.FloatTensor(x))
            ys.append(torch.FloatTensor(y))
            bc += 1
            
            if((bc == batch_size) or (i == datanum - 1)):
                images = torch.stack(xs, dim = 0).to(device)
                targets = [y.to(device) for y in ys]
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
                
##(5) モデルの保存(各エポックでモデルを保存)
    outfile = 'ssd0_' + str(ep) + '.model'
    torch.save(net.state_dict(), outfile)
    print(outfile, "saved")


image = cv2.imread(filename)
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -=(104.0, 117.0, 123.0)
x = torch.from_numpy(x[:,:,(2,1,0)]).permute(2,0,1)