#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class NeuralSort (torch.nn.Module):
    def __init__(self, tau=1.0, device="cuda"):
        super(NeuralSort, self).__init__()
        self.device = device
        self.tau = tau

    def forward(self, scores: Tensor):
        
        # score shape: batch size * policies * number of pnls * pnl size

        init_size = scores.size()
        bsize = scores.size()[0]
        scores = scores.unsqueeze(-1)
        dim = scores.size()[-2]
        one = torch.FloatTensor(dim, 1).fill_(1).to(self.device)

        A_scores = torch.abs(scores - scores.permute(0, 1, 2,4,3))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(torch.FloatTensor).to(self.device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 1, 2, 4, 3)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)
        

        return P_hat
    
class Discriminator(nn.Module):
    def __init__(self,pnl_size,device="cuda"):
        super(Discriminator, self).__init__()
                    
        self.neural_sort = NeuralSort(tau=1.0, device=device)
        self.dense1 = nn.Linear(pnl_size,pnl_size)  
        self.dense2 = nn.Linear(pnl_size,2) 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = torch.matmul(self.neural_sort(x),x.unsqueeze(-1))
        x = self.dense2(self.relu(self.dense1(x)))
        x = self.sigmoid(x)
                    
        return x


