#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from math import sqrt as sqrt
from itertools import product as product
from detection import Detect

def make_vgg():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C',
           512, 512, 512, 'M', 512, 512, 512]
    layers = []
    in_channels = 3
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
       nn.Conv2d(1024, 256, kernel_size=(1)),
       nn.Conv2d(256, 512, kernel_size=(3), stride=2, padding=1),
       nn.Conv2d(512, 128, kernel_size=(1)),
       nn.Conv2d(128, 256, kernel_size=(3), stride=2, padding=1),
       nn.Conv2d(256, 128, kernel_size=(1)),
       nn.Conv2d(128, 256, kernel_size=(3)),
       nn.Conv2d(256, 128, kernel_size=(1)),
       nn.Conv2d(128, 256, kernel_size=(3))
    ]
    return nn.ModuleList(layers)

def make_loc():
    layers = [
       # out1 に対する処理
        nn.Conv2d(512, 4*4, kernel_size=3, padding=1),
       # out2 に対する処理
        nn.Conv2d(1024, 6*4, kernel_size=3, padding=1),
       # out3 に対する処理
        nn.Conv2d(512, 6*4, kernel_size=3, padding=1),
       # out4 に対する処理
        nn.Conv2d(256, 6*4, kernel_size=3, padding=1),
       # out5 に対する処理
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1),
       # out6 に対する処理
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1),
    ]
    return nn.ModuleList(layers)

def make_conf(num_classes=21):
    layers = [
       # out1 に対する処理
        nn.Conv2d(512, 4*num_classes, kernel_size=3, padding=1),
       # out2 に対する処理
        nn.Conv2d(1024, 6*num_classes, kernel_size=3, padding=1),
       # out3 に対する処理
        nn.Conv2d(512, 6*num_classes, kernel_size=3, padding=1),
       # out4 に対する処理
        nn.Conv2d(256, 6*num_classes, kernel_size=3, padding=1),
       # out5 に対する処理
        nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1),
       # out6 に対する処理
       nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1)
    ]
    return nn.ModuleList(layers)

class L2Norm(nn.Module):
    def __init__(self,n_channels=512, scale=20):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma)
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class PriorBox(object):
    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = 300
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output

class SSD(nn.Module):
    def __init__(self, phase='train',num_classes=21):
        super(SSD,self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc = make_loc()
        self.conf = make_conf(num_classes)
        dbox = PriorBox()
        self.priors = dbox.forward()
        if phase == 'test':
            self.detect = Detect()            
    def forward(self, x):
        bs = len(x)
        out, lout, cout = [], [], []
        for i in range(23):
            x = self.vgg[i](x)
        x1 = x
        out.append(self.L2Norm(x1))
        for i in range(23,len(self.vgg)):
            x = self.vgg[i](x)
        out.append(x)
        for i in range(0,8,2):
            x = F.relu(self.extras[i](x), inplace=True)
            x = F.relu(self.extras[i+1](x), inplace=True)
            out.append(x)
        for i in range(6):
            lx = self.loc[i](out[i]).permute(0,2,3,1).reshape(bs,-1,4)
            cx = self.conf[i](out[i]).permute(0,2,3,1).reshape(bs,-1,self.num_classes)
            lx = self.loc[i](out[i]).permute(0,2,3,1).reshape(bs,-1,4)
            cx = self.conf[i](out[i]).permute(0,2,3,1).reshape(bs,-1,self.num_classes)
            lout.append(lx)
            cout.append(cx)
        lout = torch.cat(lout, 1)
        cout = torch.cat(cout, 1)

        output = (lout, cout, self.priors)
        if self.phase == 'test':
            return self.detect.apply(output,self.num_classes)
        else:
            return output
