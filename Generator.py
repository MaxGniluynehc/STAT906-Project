import torch as tc
import torch.nn as nn
import numpy as np
from Tradining_Strategies import TradingStrategy
from Discriminator import Discriminator
from utils import *

class Generator(nn.Module):
    def __init__(self, noise_size, pnl_size, market_size, noise_type="gaussian", device="cuda"):
        super(Generator, self).__init__()

        self.device = device

        self.noise_size = noise_size
        self.pnl_size = pnl_size # this is T
        self.market_size = market_size # this is M

        self.noise_type = noise_type

        self.dense1 = nn.Linear(noise_size, 256, device=self.device)
        self.dense2 = nn.Linear(256, 256, device=self.device)
        self.dense3 = nn.Linear(256, pnl_size, device=self.device)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def generate_noise_input(self, type="gaussian", **kwargs):
        if type == "gaussian":
            nz = tc.tensor(np.random.normal(loc=kwargs["mean"], scale=kwargs["std"],
                                  size=[self.market_size, self.noise_size])).to(dtype=tc.float32, device=self.device)
            # nz = tc.normal(mean=kwargs["mean"], std=kwargs["std"],
            #                size=[self.market_size, self.noise_size],
            #                dtype=tc.float32, device=self.device)
            assert not tc.any(tc.isnan(nz)), AssertionError("gen: nz returns nan!")
            return nz
        else:
            ValueError("Wrong error type!")

    def forward(self, hidden_layer_num =1, **kwargs):
        nz = self.generate_noise_input(type=self.noise_type, **kwargs)
        assert not tc.any(tc.isnan(nz)), AssertionError("gen: nz returns nan!")

        x = self.dense1(nz)
        x = self.leakyrelu(x)
        assert not tc.any(tc.isnan(x)), AssertionError("gen: x after dense1 returns nan!")

        for i in range(hidden_layer_num):
            x = self.dense2(x)
            x = self.leakyrelu(x)
            assert not tc.any(tc.isnan(x)), AssertionError("gen: x in hidden dense{} returns nan!".format(i+1))

        x = self.dense3(x)
        assert not tc.any(tc.isnan(x)), AssertionError("gen: x in output layer returns nan!")

        return x

    def loss(self, ps_real, ps_fake, strategies:list|TradingStrategy, discriminator:Discriminator, reinforce=False):
        gen_loss1 = 0
        gen_loss2 = 0
        for trade_strategy in strategies:
            pnl_real = trade_strategy.get_strategy_PnL(ps_real)
            pnl_fake = trade_strategy.get_strategy_PnL(ps_fake)
            fake_ve = discriminator.forward(pnl_fake)
            fake_v, fake_e = fake_ve[:, 0], fake_ve[:, 1]

            # abs mean score loss
            gen_loss1 += tc.abs(tc.mean(score(fake_v, fake_e, pnl_real, 0.05)))

            if reinforce:
                true_v = VaR(0.05, pnl_real)
                true_e = ES(0.05, pnl_real)
                gen_loss2 += tc.mean(tc.pow(true_v- fake_v, 2)) + tc.mean(tc.pow(true_e- fake_e, 2))

        gen_loss = gen_loss1 + gen_loss2
        return gen_loss/len(strategies)



if __name__ == '__main__':
    G = Generator(noise_size=10,
                  pnl_size=100,
                  market_size=5)
    G.device
    G.generate_noise_input(mean=0, std=1).shape
    G.forward(mean=0, std=1).shape




















