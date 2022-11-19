import torch as tc
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, noise_size, pnl_size, market_size, noise_type="gaussian", device="mps"):
        super(Generator, self).__init__()
        if tc.backends.mps.is_available():
            self.device = tc.device(device)
        else:
            Warning("GPU unavailable.")
            self.device = tc.device("cpu")

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
            nz = tc.normal(mean=kwargs["mean"], std=kwargs["std"],
                           size=[self.market_size, self.noise_size],
                           dtype=tc.float32, device=self.device)
            return nz
        else:
            ValueError("Wrong error type!")

    def forward(self, hidden_layer_num =3, **kwargs):
        nz = self.generate_noise_input(type=self.noise_type, **kwargs)
        x = self.dense1(nz)
        x = self.relu(x)
        for i in range(hidden_layer_num):
            x = self.dense2(x)
            x = self.relu(x)
        x = self.dense3(x)
        return x
        # nn.ModuleList([self.dense2, nn.ReLU()]*hidden_layer_num)


if __name__ == '__main__':
    G = Generator(noise_size=10,
                  pnl_size=100,
                  market_size=5)
    G.device
    G.generate_noise_input(mean=0, std=1).shape
    G.forward(mean=0, std=1).shape


















