from dataloader import PriceScenarioDataset
from utils import VaR, ES, score
from Discriminator import Discriminator
from Generator import Generator
from Tradining_Strategies import TradingStrategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from preprocess.acf import *
# from preprocess.gaussianize import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from model.torch_tcn import *

import torch.optim as optim
from tqdm import tqdm

from scipy.stats import norm



num_epochs = 100
batch_size = 128
lr = 0.01
noise_size=100
pnl_size=101
market_size=5

if torch.cuda.is_available():
    dev = "cuda"
elif torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"

device = torch.device(dev)

def toy_sampler(n, T=100, p0=1):
    pt = torch.ones([n, T+1]) * p0
    ep_s = torch.normal(mean=0, std=1e-3,size=(n, T))
    u_s1 = -1.5e-3 + torch.rand(size=(n, int(T/2))) * (-8e-4 - (-1.5e-3))
    u_s2 = 4e-4 + torch.rand(size=(n, int(T/2))) * (8e-4 - 4e-4)
    u_s = torch.cat((u_s1, u_s2), dim=1)
    dp_s = ep_s + u_s
    pt[:, 1:] += torch.cumsum(dp_s, dim=1)
    return pt.to(device)

toy_sample_n = 5000
toy_sample = toy_sampler(toy_sample_n, T=100)

# Show toy sample
plt.figure()
for i in torch.randint(0, toy_sample_n, [100]):
    plt.plot(list(range(101)), toy_sample[i,:].to("cpu"), color="gray", alpha=0.1)

toy_sample_log = torch.log(toy_sample) # log paths log(pt/p0)
plt.figure()
for i in torch.randint(0, toy_sample_n, [100]):
    plt.plot(list(range(101)), toy_sample_log[i,:].to("cpu"), color="gray", alpha=0.1)


# define dataloader
dataloader = torch.utils.data.DataLoader(toy_sample, batch_size=batch_size, drop_last=True, shuffle=True)

# define GAN model
generator = Generator(noise_size=noise_size, pnl_size=pnl_size, market_size=batch_size, device=dev) #.to(device)
discriminator = Discriminator(pnl_size=pnl_size, device=dev)# .to(device)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=100, eta_min=0)
gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=100, eta_min=0)


# define singals and trading strategies
signal = (toy_sample.max() - toy_sample.min())/20
look_back = 10
trade_strategy_1 = TradingStrategy("buy-hold", look_back, (0, 0), (signal, signal), device=device)
trade_strategy_2 = TradingStrategy("MA", look_back, (0, 0), (signal, signal), device=device)
trade_strategy_3 = TradingStrategy("MOM", look_back, (0, 0), (signal, signal), device=device)


# Train
def train(epochs=tqdm(range(100)), lbd=0.5,
          logs_PATH = "/Users/y222chen/Documents/Max/Study/STAT906_Comp_Intense_Models_in_Finance/Project/project/logs20221206/"):
    generator.train()
    discriminator.train()
    # logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT906_Comp_Intense_Models_in_Finance/Project/project/logs20221206/"

    gen_loss_logs = torch.empty(0)
    disc_loss_logs = torch.empty(0)

    # epoch = next(iter(t))
    # ps_real = next(iter(dataloader))
    print("====================================== Start Training ================================================")
    for epoch in epochs:
        for idx, ps_real in enumerate(dataloader, 0):
            # Train discriminator
            for _ in range(1):  # (idx<=10 and epoch ==0):
                discriminator.zero_grad()

                ps_fake = generator.forward(mean=0, std=1).reshape(batch_size, -1).detach()

                disc_loss = 0
                for trade_strategy in [trade_strategy_1, trade_strategy_2, trade_strategy_3]:
                    pnl_real = trade_strategy.get_strategy_PnL(ps_real)  # ,torch.ones_like(ps_real))
                    pnl_fake = trade_strategy.get_strategy_PnL(ps_fake)  # ,torch.ones_like(ps_fake))

                    fake_ve = discriminator.forward(pnl_fake)
                    fake_v, fake_e = fake_ve[:, 0], fake_ve[:, 1]
                    real_ve = discriminator(pnl_real)
                    real_v, real_e = real_ve[:, 0], real_ve[:, 1]

                    disc_loss += torch.mean(score(fake_v, fake_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05)) - \
                                 lbd * torch.mean(score(real_v, real_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05))

                    # VaR(0.95, ps_real.cpu())
                    # torch.sort(ps_real.cpu())
                    # VaR(0.05, pnl_real)
                    # ES(0.05, pnl_real)
                    # torch.max(score(VaR(0.05, pnl_real), ES(0.05, pnl_real), pnl_real, 0.05))
                    # score(fake_v, fake_e, ps_real, 0.05).shape

                disc_loss /= 3
                disc_loss.backward()
                disc_optimizer.step()
                scheduler_disc.step()
                disc_loss_logs = torch.cat((disc_loss_logs, disc_loss.detach().unsqueeze(-1).cpu()))

            for _ in range(1):
                generator.zero_grad()
                # discriminator.zero_grad()
                ps_fake = generator(mean=0, std=1).reshape(batch_size, -1)

                gen_loss = 0
                for trade_strategy in [trade_strategy_1, trade_strategy_2, trade_strategy_3]:
                    pnl_real = trade_strategy.get_strategy_PnL(ps_real)
                    pnl_fake = trade_strategy.get_strategy_PnL(ps_fake)

                    fake_ve = discriminator(pnl_fake)
                    fake_v, fake_e = fake_ve[:, 0], fake_ve[:, 1]
                    gen_loss += torch.mean(score(fake_v, fake_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05))

                gen_loss /= 3
                gen_loss.backward()
                gen_optimizer.step()
                scheduler_gen.step()
                gen_loss_logs = torch.cat((gen_loss_logs, gen_loss.detach().unsqueeze(-1).cpu()))

        if (epoch + 1) % 10 == 0:
            plt.figure()
            ps_fake = generator(mean=0, std=1).reshape(batch_size, -1).detach().cpu()
            for i in range(len(ps_fake)):
                plt.plot(list(range(101)), ps_fake[i], color="blue", alpha=0.1)
            plt.savefig(logs_PATH+"fake_ps_at_epoch={}.png".format(epoch))

            # Save model
            torch.save(generator, logs_PATH+"trained_generator_at_epoch_{}.pth".format(epoch))
            torch.save(discriminator, logs_PATH + "trained_discriminator_at_epoch_{}.pth".format(epoch))

        epochs.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))




# Load trained generator
# generator = torch.load(logs_PATH+"trained_generator_at_epoch_{}.pth".format(9))
# discriminator = torch.load(logs_PATH+"trained_discriminator_at_epoch_{}.pth".format(9))
# generator.eval()
# generator.forward(mean=0, std=1)


if __name__ == '__main__':
    train()













