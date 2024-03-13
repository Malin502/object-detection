import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def make_vgg():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C',
           512, 512, 512, 'M', 512, 512, 512]
    
    layers = []
    in_channels = 3
    
    
    #NNの構築
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    
    return nn.ModuleList(layers)

def make_extras():

    layers = [
        nn.Conv2d(1024, 256, kernel_size=1),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(512, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(256, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3),
        nn.Conv2d(256, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3)
    ]
    
    return nn.ModuleList(layers)

def make_loc(num_classes=21):
    layers = [
        #out1に関する処理
        nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1), 
        #out2に関する処理
        nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1), 
        #out3に関する処理
        nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
        #out4に関する処理
        nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
        #out5に関する処理
        nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
        #out6に関する処理
        nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
    ]
    
    return nn.ModuleList(layers)

def make_conf(num_classes=21):
    layers = [
        #out1に関する処理
        nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
        #out2に関する処理
        nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
        #out3に関する処理
        nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
        #out4に関する処理
        nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
        #out5に関する処理
        nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
        #out6に関する処理
        nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
    ]
    
    return nn.ModuleList(layers)

class L2Norm(nn.Module):
    def __init__(self, n_channels = 512, scale = 20):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)
        
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).\
            unsqueeze(3).expand_as(x) * x
            
        return out