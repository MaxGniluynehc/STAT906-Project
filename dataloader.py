import torch as tc
from torch.utils.data import Dataset, DataLoader
from utils import VaR, ES, score


class PriceScenarioDataset(Dataset):
    def __init__(self, price_scenario, batch_time_size, alpha=None,
                 return_risk_measures=False):
        '''
        :param price_scenario: [N,1,T] or simply [N, T]
            - In the paper, price_scenario is [N,M,T], but here we have already
              pre-constructed the portfolio, so M = 1 is given.
            - In the simpliest case when N = 1, make sure price scenario is [1,T]
              rather than [T].
        :param batch_time_size: the length of time in each batched sample.
            (Since the data is time series, we cannot randomly sample over time. So the dataloader
            has to randomly select a single starting point (t between 0 to T), from which a batch
            of time-series chunk of length batch_time_size is sampled as a whole.)
        :param alpha: the significance level to compute VaR and ES
        '''
        self.T = price_scenario.shape[-1]
        self.num_preconstruced_portfolios = price_scenario.shape[0]
        self.price_scenario = price_scenario.reshape([-1, self.T])
        self.batch_time_size = batch_time_size
        self.alpha = alpha
        self.return_risk_measures = return_risk_measures

    def __len__(self):
        return self.T - self.batch_time_size + 1

    def __getitem__(self, index):

        ps = self.price_scenario[:, index: index+self.batch_time_size]

        if self.return_risk_measures:
            v = VaR(self.alpha, ps)
            e = ES(self.alpha, ps)
            s = score(v,e,ps, self.alpha)
            return ps, v, e, s
        else:
            return ps
















