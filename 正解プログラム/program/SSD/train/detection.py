#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from myfunctions import decode, nms

class Detect(Function):
    def forward(self, output, num_classes, 
                top_k=200, variance=[0.1,0.2], 
                conf_thresh=0.01, nms_thresh=0.45):    
        loc_data, conf_data, prior_data = output[0], output[1], output[2]
        # conf_data は各クラスの信頼度、[ bs, 8732, num_classes ]
        # 信頼度の部分を softmax で確率に直す
        softmax = nn.Softmax(dim=-1)
        conf_data = softmax(conf_data)       
        # num  は batch の大きさ 
        num = loc_data.size(0)  
        # 出力の配列を準備、中身は今は 0 、[ bs, 21, 200, 5 ]
        output = torch.zeros(num, num_classes, top_k, 5)
        # conf_data [bs, 8732, num_classes] を 
        # [bs,num_classes,8732]  に変形して conf_preds と名付ける
        conf_preds = conf_data.transpose(2, 1)
        # Decode predictions into bboxes.
        for i in range(num):  # バッチ内の各データの処理
            # loc_data と DBox から BBox を作成
            decoded_boxes = decode(loc_data[i], prior_data, variance)
            # conf_preds を conf_scores にハードコピー
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes): # 各クラスの処理
                # conf_scores で信頼度が conf_thresh 以上の index を求める
                c_mask = conf_scores[cl].gt(conf_thresh)
                # conf_thresh 以上の信頼度の集合を作る
                scores = conf_scores[cl][c_mask]
                # その集合の要素数が 0、つまりconf_thresh 以上はない
                # これ以降の処理はなしで、次のクラスへ
                if scores.size(0) == 0:
                    continue
                # c_mask を decoded_boxes に適用できるようにサイズ変更
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask を decoded_boxes に適用、、、１次元になる
                # view(-1, 4) でサイズを戻す
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # boxes に対して nms を適用、
                # ids は nms を通過した BBox の index 
                # count は nms を通過した BBox の数
                ids, count = nms(boxes, scores, nms_thresh, top_k)
                # 上記の結果を output に格納
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output

