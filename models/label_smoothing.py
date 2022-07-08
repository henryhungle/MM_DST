import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import pdb 

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, state_size, res_size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.state_size = state_size
        self.res_size = res_size 
        self.true_dist = None
        self.log_softmax = True

    def _no_log_softmax(self):
        self.log_softmax = False 

    def forward(self, x, target, pred_type):
        if self.log_softmax:
            scores = F.log_softmax(x, dim=-1)
        else:
            scores = x 
        if pred_type == 'state':
            size = self.state_size 
        elif pred_type == 'res': 
            size = self.res_size 
        assert scores.size(1) == size
        
        true_dist = scores.data.clone()
        true_dist.fill_(self.smoothing/(size-1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.padding_idx>0:
            true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.sum()>0 and len(mask)>0: 
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        loss = self.criterion(scores, Variable(true_dist, requires_grad=False))
        norm = (target!=self.padding_idx).sum()
        
        return loss/norm 

