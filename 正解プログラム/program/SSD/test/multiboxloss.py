#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from myfunctions import match

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes=21, overlap_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        self.device = device

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)  # batch size
        priors = priors[:loc_data.size(1), :]  ## 8732*4 
        num_priors = (priors.size(0))  ## 8732  カッコ必要？
        num_classes = self.num_classes
        ## オフセットを入れる箱 
        loc_t = torch.Tensor(num, num_priors, 4).to(self.device) 
        ## conf_t : torch.Size([4, 8732]) どの DBox か
        conf_t = torch.LongTensor(num, num_priors).to(self.device)
        for idx in range(num):   ## num は bs、各教師 image データに対して
            ## BB の位置情報 4次元、複数個  
            truths = targets[idx][:, :-1].to(self.device) 
            ## BB のラベル、複数個
            labels = targets[idx][:, -1].to(self.device)
            ## DBox を gpu へ
            defaults = priors.to(self.device)  
            match(self.threshold, truths, defaults,
                  self.variance, labels, loc_t, conf_t, idx)
        pos = conf_t > 0  ## 背景ではないもの
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
#        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
#        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = F.cross_entropy(batch_conf, conf_t.view(-1), reduction='none')
        # Hard Negative Mining
        num_pos = pos.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
#        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
