#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from Tradining_Strategies import TradingStrategy
from utils import *



# input shape: batch size * policies * number of pnls * pnl size
# The differentiable version of sorting introduced in the paper
class NeuralSort (torch.nn.Module):
    def __init__(self, tau=1.0, device="cuda"): # set device to "mps" on Mac
        super(NeuralSort, self).__init__()
        self.device = device
        self.tau = tau

    def forward(self, scores: Tensor): # score: [batch_size, 1, batch_time_size]

        init_size = scores.size()
        bsize = scores.size()[0]
        scores = scores.unsqueeze(-1) # score: [batch_size, batch_time_size]
        dim = scores.size()[1] # dim is the batch_time_size, or n in Eq(28)
        one = torch.FloatTensor(dim, 1).fill_(1).to(self.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(torch.FloatTensor).to(self.device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        return P_hat



class Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self, pnl_size, device):
        super(Discriminator, self).__init__()
                    
        self.neural_sort = NeuralSort(tau=1.0, device=device)
        self.dense1 = nn.Linear(pnl_size, pnl_size, device=device)
        self.dense2 = nn.Linear(pnl_size, pnl_size, device=device)
        self.dense3 = nn.Linear(pnl_size,2, device=device)
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.matmul(self.neural_sort(x),x.unsqueeze(-1)).squeeze(-1)
        x = self.dense2(self.leakyrelu(self.dense1(x)))
        x = self.dense3(self.leakyrelu(x))
        # x = self.sigmoid(x)
        # x = self.tanh(x)
        return x

    def loss(self, lbd, ps_real, ps_fake, strategies:list|TradingStrategy, reinforce=False):
        disc_loss1 = 0
        disc_loss2 = 0
        for trade_strategy in strategies:
            pnl_real = trade_strategy.get_strategy_PnL(ps_real)  # ,tc.ones_like(ps_real))
            pnl_fake = trade_strategy.get_strategy_PnL(ps_fake)  # ,tc.ones_like(ps_fake))

            fake_ve = self.forward(pnl_fake)
            fake_v, fake_e = fake_ve[:, 0], fake_ve[:, 1]
            real_ve = self(pnl_real)
            real_v, real_e = real_ve[:, 0], real_ve[:, 1]

            disc_loss1 += tc.abs(tc.mean(score(fake_v, fake_e, pnl_real, 0.05)) - \
                                  lbd * tc.mean(score(real_v, real_e, pnl_real, 0.05)))

            # constraint the output to the real VaR and ES
            if reinforce:
                true_v = VaR(0.05, pnl_real)
                true_e = ES(0.05, pnl_real)
                disc_loss2 += tc.abs(tc.mean(score(real_v, real_e, pnl_real, 0.05)) - \
                                      lbd * tc.mean(score(true_v, true_e, pnl_real, 0.05)))

        disc_loss = -disc_loss1 + disc_loss2

        return disc_loss/len(strategies)

