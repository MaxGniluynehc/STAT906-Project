#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


# input shape: batch size * policies * number of pnls * pnl size

class NeuralSort (torch.nn.Module):
    def __init__(self, tau=1.0, device="cuda"): # set device to "mps" on Mac
        super(NeuralSort, self).__init__()
        self.device = device
        self.tau = tau

    def forward(self, scores: Tensor):

        init_size = scores.size()
        bsize = scores.size()[0]
        scores = scores.reshape(bsize,-1,1)
        scores = scores.unsqueeze(-1)
        dim = scores.size()[1]
        one = torch.FloatTensor(dim, 1).fill_(1).to(self.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(torch.FloatTensor).to(self.device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)
        
        P_hat = P_hat.reshape(init_size)

        return P_hat
    
class Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self, pnl_size, device="cuda"):
        super(Discriminator, self).__init__()
                    
        self.neural_sort = NeuralSort(tau=1.0, device=device)
        self.dense1 = nn.Linear(pnl_size, pnl_size)
        self.dense2 = nn.Linear(pnl_size,2)  
        
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = torch.matmul(self.neural_sort(x),x)
        x = self.dense2(self.dense1(x))
        x = self.sigmoid(x)
                    
        return x

