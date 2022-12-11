import torch as tc
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, noise_size, pnl_size, market_size, noise_type="gaussian", device="cuda"):
        super(Generator, self).__init__()

        self.device = device
        noise_size = 1000

        self.noise_size = noise_size
        self.pnl_size = pnl_size # this is T
        self.market_size = market_size # this is M

        self.noise_type = noise_type

        self.dense1 = nn.Linear(noise_size, 1000, device=self.device)
        self.dense2 = nn.Linear(1000, 128, device=self.device)
        self.dense3 = nn.Linear(128, 256, device=self.device)
        self.dense4 = nn.Linear(256, 512, device=self.device)
        self.dense5 = nn.Linear(512, 1024, device=self.device)
        self.dense6 = nn.Linear(1024, pnl_size, device=self.device)

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(pnl_size)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def generate_noise_input(self, type="gaussian", **kwargs):
        if type == "gaussian":
            nz = tc.normal(mean=kwargs["mean"], std=kwargs["std"],
                           size=[self.market_size, self.noise_size],
                           dtype=tc.float32, device=self.device)
            return nz
        else:
            ValueError("Wrong error type!")

    def forward(self, hidden_layer_num =1, **kwargs):
        nz = self.generate_noise_input(type=self.noise_type, **kwargs)
        x = self.dense1(nz)
        x = self.leakyrelu(x)
        x = self.leakyrelu(self.dense2(x))
        x = self.leakyrelu(self.dense3(x))
        x = self.leakyrelu(self.dense4(x))
        x = self.leakyrelu(self.dense5(x))
        x = self.dense6(x)
        # for i in range(hidden_layer_num):
        #     x = self.dense2(x)
        #     x = self.leakyrelu(x)
        # x = self.dense3(x)
        # x = self.tanh(x)
        return x
        # nn.ModuleList([self.dense2, nn.ReLU()]*hidden_layer_num)


if __name__ == '__main__':
    G = Generator(noise_size=10,
                  pnl_size=100,
                  market_size=5)
    G.device
    G.generate_noise_input(mean=0, std=1).shape
    G.forward(mean=0, std=1).shape


















