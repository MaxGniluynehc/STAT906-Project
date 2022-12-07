from dataloader import PriceScenarioDataset
from utils import moving_average, VaR, ES, score
from Discriminator import Discriminator
from Generator import Generator
from Tradining_Strategies import TradingStrategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from preprocess.acf import *
# from preprocess.gaussianize import *

import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from model.torch_tcn import *

import torch.optim as optim
from tqdm import tqdm

from scipy.stats import norm



num_epochs = 100
batch_size = 128
lr = 1e-5
noise_size=100
pnl_size=101
market_size=5
# logs_PATH = "/Users/y222chen/Documents/Max/Study/STAT906_Comp_Intense_Models_in_Finance/Project/project/logs20221206/"
logs_PATH = "/Users/y222chen/Documents/Max/Study/STAT906_Comp_Intense_Models_in_Finance/Project/project/logs20221206-reinforce/"
# logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT906_Comp_Intense_Models_in_Finance/Project/project/logs20221206/"


if tc.cuda.is_available():
    dev = "cuda"
elif tc.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"

device = tc.device(dev)

def toy_sampler(n, T=100, p0=1):
    pt = tc.ones([n, T+1]) * p0
    ep_s = tc.normal(mean=0, std=1e-3,size=(n, T))
    u_s1 = -1.5e-3 + tc.rand(size=(n, int(T/2))) * (-8e-4 - (-1.5e-3))
    u_s2 = 4e-4 + tc.rand(size=(n, int(T/2))) * (8e-4 - 4e-4)
    u_s = tc.cat((u_s1, u_s2), dim=1)
    dp_s = ep_s + u_s
    pt[:, 1:] += tc.cumsum(dp_s, dim=1)
    return pt.to(device)

toy_sample_n = 5000
toy_sample = toy_sampler(toy_sample_n, T=100)
toy_sample_log = tc.log(toy_sample) # log paths log(pt/p0)

# Show toy sample
plt.figure()
for i in tc.randint(0, toy_sample_n, [100]):
    plt.plot(list(range(101)), toy_sample[i,:].to("cpu"), color="gray", alpha=0.1)

plt.figure()
for i in tc.randint(0, toy_sample_n, [100]):
    plt.plot(list(range(101)), toy_sample_log[i,:].to("cpu"), color="gray", alpha=0.1)


# define dataloader
dataloader = tc.utils.data.DataLoader(toy_sample, batch_size=batch_size, drop_last=True, shuffle=True)

# define GAN model
generator = Generator(noise_size=noise_size, pnl_size=pnl_size, market_size=batch_size, device=dev) #.to(device)
discriminator = Discriminator(pnl_size=pnl_size, device=dev)# .to(device)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
scheduler_disc = tc.optim.lr_scheduler.CosineAnnealingWarmRestarts(disc_optimizer, T_0=50, eta_min=1e-7) # CosineAnnealingLR(disc_optimizer, T_max=100, eta_min=1e-7)
gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
scheduler_gen = tc.optim.lr_scheduler.CosineAnnealingWarmRestarts(gen_optimizer, T_0=50, eta_min=1e-7) #.CosineAnnealingLR(gen_optimizer, T_max=100, eta_min=1e-7)


# define singals and trading strategies
look_back = 20
signal = tc.std(toy_sample - moving_average(toy_sample, look_back))*0.5 # (toy_sample.max() - toy_sample.min())/20
trade_strategy_1 = TradingStrategy("buy-hold", look_back, (0, 0), (signal, signal), device=device)
trade_strategy_2 = TradingStrategy("MA", look_back, (0, 0), (signal, signal), device=device)
trade_strategy_3 = TradingStrategy("MOM", look_back, (0, 0), (signal, signal), device=device)


# pnl = trade_strategy_2.get_strategy_PnL(toy_sample)
# plt.figure()
# for i in tc.randint(0, pnl.shape[0], size=[5]):
#     plt.plot(list(range(pnl.shape[1])), pnl[i, :].cpu(), color="gray", alpha=0.1)


# Train
def train(epochs=tqdm(range(100)), lbd=1, logs_PATH = logs_PATH):
    generator.train()
    discriminator.train()

    gen_loss_logs = tc.empty(0)
    disc_loss_logs = tc.empty(0)

    # epoch = next(iter(epochs))
    # ps_real = next(iter(dataloader))
    print("====================================== Start Training ================================================")
    for epoch in epochs:
        for idx, ps_real in tqdm(enumerate(dataloader, 0)):

            # Train discriminator
            for _ in range(3):  # (idx<=10 and epoch ==0):
                # disc_optimizer.zero_grad()
                generator.zero_grad()
                discriminator.zero_grad()

                ps_fake = generator.forward(mean=0, std=1).reshape(batch_size, -1).detach()
                assert not tc.any(tc.isnan(ps_fake)), AssertionError("training_disc: ps_fake returns nan!")

                disc_loss = discriminator.loss(lbd, ps_real, ps_fake, [trade_strategy_1, trade_strategy_2, trade_strategy_3], reinforce=True)

                # disc_loss = 0
                # for trade_strategy in [trade_strategy_1, trade_strategy_2, trade_strategy_3]:
                #     pnl_real = trade_strategy.get_strategy_PnL(ps_real)  # ,tc.ones_like(ps_real))
                #     assert not tc.any(tc.isnan(pnl_real)), AssertionError("training_disc: pnl_real returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #     pnl_fake = trade_strategy.get_strategy_PnL(ps_fake)  # ,tc.ones_like(ps_fake))
                #     assert not tc.any(tc.isnan(pnl_fake)), AssertionError("training_disc: pnl_fake returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #     fake_ve = discriminator.forward(pnl_fake)
                #     assert not tc.any(tc.isnan(fake_ve)), AssertionError("training_disc: fake_ve returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #     fake_v, fake_e = fake_ve[:, 0], fake_ve[:, 1]
                #
                #     real_ve = discriminator(pnl_real)
                #     real_v, real_e = real_ve[:, 0], real_ve[:, 1]
                #
                #     assert not tc.any(tc.isnan(score(fake_v, fake_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05))), AssertionError(
                #         "training_disc: fake_score returns nan with strategy = {}!".format(trade_strategy.strategy))
                #     assert not tc.any(tc.isnan(score(real_v, real_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05))), AssertionError(
                #         "training_disc: real_score returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #
                #     disc_loss -= tc.mean(score(fake_v, fake_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05)) - \
                #                  lbd * tc.mean(score(real_v, real_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05))
                # #     disc_loss += tc.abs(tc.mean(score(fake_v, fake_e, pnl_real, 0.05)) - \
                # #                  lbd * tc.mean(score(real_v, real_e, pnl_real, 0.05)))
                # disc_loss /= 3

                disc_loss.backward()
                disc_optimizer.step()
                scheduler_disc.step()
                disc_loss_logs = tc.cat((disc_loss_logs, disc_loss.detach().unsqueeze(-1).cpu()))

            for _ in range(3):
                # gen_optimizer.zero_grad()
                generator.zero_grad()
                discriminator.zero_grad()
                ps_fake = generator(mean=0, std=1).reshape(batch_size, -1)
                assert not tc.any(tc.isnan(ps_fake)), AssertionError("training_gen: ps_fake returns nan!")

                gen_loss = generator.loss(ps_real, ps_fake, [trade_strategy_1, trade_strategy_2, trade_strategy_3], discriminator, reinforce=True)

                # gen_loss = 0
                # for trade_strategy in [trade_strategy_1, trade_strategy_2, trade_strategy_3]:
                #     pnl_real = trade_strategy.get_strategy_PnL(ps_real)
                #     assert not tc.any(tc.isnan(pnl_real)), AssertionError("training_gen: pnl_real returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #     pnl_fake = trade_strategy.get_strategy_PnL(ps_fake)
                #     assert not tc.any(tc.isnan(pnl_fake)), AssertionError("training_gen: pnl_fake returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #
                #     fake_ve = discriminator(pnl_fake)
                #     assert not tc.any(tc.isnan(fake_ve)), AssertionError("training_gen: fake_ve returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #     fake_v, fake_e = fake_ve[:, 0], fake_ve[:, 1]
                #
                #     assert not tc.any(tc.isnan(score(fake_v, fake_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05))), AssertionError(
                #         "training_gen: fake_score returns nan with strategy = {}!".format(trade_strategy.strategy))
                #
                #     gen_loss += tc.mean(score(fake_v, fake_e, pnl_real, 0.05))
                #
                #     # gen_loss += tc.mean(score(fake_v, fake_e, pnl_real[:, round(0.05*pnl_real.shape[-1])-1], 0.05))
                # gen_loss /= 3

                gen_loss.backward()
                gen_optimizer.step()
                scheduler_gen.step()
                gen_loss_logs = tc.cat((gen_loss_logs, gen_loss.detach().unsqueeze(-1).cpu()))

        if (epoch + 1) % 10 == 0:
            plt.figure()
            ps_fake = generator(mean=0, std=1).reshape(batch_size, -1).detach().cpu()
            for i in range(len(ps_fake)):
                plt.plot(list(range(101)), ps_fake[i], color="blue", alpha=0.1)
            plt.savefig(logs_PATH+"fake_ps_at_epoch={}.png".format(epoch))

            # Save model
            tc.save(generator, logs_PATH+"trained_generator_at_epoch_{}.pth".format(epoch))
            tc.save(discriminator, logs_PATH + "trained_discriminator_at_epoch_{}.pth".format(epoch))

        epochs.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))

    return gen_loss_logs, disc_loss_logs


# Load trained generator
# generator = tc.load(logs_PATH+"trained_generator_at_epoch_{}.pth".format(9))
# discriminator = tc.load(logs_PATH+"trained_discriminator_at_epoch_{}.pth".format(9))
# generator.eval()
# generator.forward(mean=0, std=1)


if __name__ == '__main__':
    num_epochs = 100
    batch_size = 128
    lr = 1e-5
    noise_size = 100
    pnl_size = 101
    market_size = 5
    # logs_PATH = "/Users/y222chen/Documents/Max/Study/STAT906_Comp_Intense_Models_in_Finance/Project/project/logs20221206/"
    logs_PATH = "/Users/y222chen/Documents/Max/Study/STAT906_Comp_Intense_Models_in_Finance/Project/project/logs20221206-reinforce/"

    dataloader = tc.utils.data.DataLoader(toy_sample, batch_size=batch_size, drop_last=True, shuffle=True)
    # define GAN model
    generator = Generator(noise_size=noise_size, pnl_size=pnl_size, market_size=batch_size, device=dev)  # .to(device)
    discriminator = Discriminator(pnl_size=pnl_size, device=dev)  # .to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    scheduler_disc = tc.optim.lr_scheduler.CosineAnnealingWarmRestarts(disc_optimizer, T_0=50,
                                                                       eta_min=1e-7)  # CosineAnnealingLR(disc_optimizer, T_max=100, eta_min=1e-7)
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    scheduler_gen = tc.optim.lr_scheduler.CosineAnnealingWarmRestarts(gen_optimizer, T_0=50,
                                                                      eta_min=1e-7)  # .CosineAnnealingLR(gen_optimizer, T_max=100, eta_min=1e-7)


    # Load trained generator
    generator = tc.load(logs_PATH+"trained_generator_at_epoch_{}.pth".format(89))
    discriminator = tc.load(logs_PATH+"trained_discriminator_at_epoch_{}.pth".format(89))

    # generator.eval()
    # pnl = generator.forward(mean=0, std=1)
    # plt.figure()
    # for i in tc.randint(0, pnl.shape[0], size=[5]):
    #     plt.plot(list(range(pnl.shape[1])), pnl[i, :].detach().cpu(), color="gray", alpha=0.1)
    # plt.ylim([-0.1, 0.1])

    gen_loss_logs, disc_loss_logs = train(epochs=tqdm(range(90,150)))
    print("gen_loss_logs: {} \n disc_loss_logs:{}".format(gen_loss_logs, disc_loss_logs))

















