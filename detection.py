from math import sqrt
from itertools import product as product

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import matplotlib.pyplot as plt

from myfunctions import decode, nms


class Detect(Function):
    def forward(self, output, num_classes, top_k = 200, varience = [0.1, 0.2],
                conf_thresh = 0.01, nms_thresh = 0.45):
        loc_data, conf_data, prior_data = output[0], output[1], output[2]
        #conf_dataは各クラスの信頼度, [batch, 8732, num_priors]
        
        #確率部分をsoftmaxで確率に直す
        softmax = nn.softmax(conf_data, dim = 2)
        conf_data = softmax(conf_data)
        
        #numはバッチサイズ
        num = loc_data.size(0)
        #出力の形状を[batch, 21, 200, 5]にする
        output = torch.zeros(num, num_classes, top_k, 5)
        
        #conf_dataを[batch, 8732, num_classes]から[batch, num_classes, 8732]に変換
        conf_preds = conf_data.transpose(2, 1)
        
        #予測をデコードしてBBoxを作成
        for i in range(num):
            #loc_dataはDBoxからBBoxを作成
            decoded_boxes = decode(loc_data[i], prior_data, varience)
            
            #conf_predsをconf_scoresにディープコピー
            conf_scores = conf_preds[i].clone()
            
            for cl in range(1, num_classes): #各クラスの処理
                c_mask = conf_scores[cl].gt(conf_thresh)
                #conf_threshを以上の信頼度の集合を作る
                
                scores = conf_scores[cl][c_mask]
                #その集合の要素数が0、つまりconf_thresh以上はない
                #これ以上の処理はない、次のクラスへ
                
                if scores.size(0) == 0:
                    continue
                
                #c_maskをdecoded_boxesに適用できるように変形
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                
                #l_maskをdecoded_boxesに適用,1次元へ
                #view(-1, 4)でサイズを戻す
                boxes = decoded_boxes[l_mask].view(-1, 4)
                
                #boxesに対してNMSを適用
                #idsはNMSを通過したBBoxのインデックス
                #countはNMSを通過したBBoxの数
                ids, count = nms(boxes, scores, nms_thresh, top_k)
                
                #上記の結果をoutputに格納
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
                    
        return output