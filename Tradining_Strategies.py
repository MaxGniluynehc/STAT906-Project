import torch as tc
import numpy as np
import utils

tc.device("mps")

# 3 strategies: buy-hold, MA, MOM

# Price process: [M, T], M is market dim, T is total time steps
# Portfolio weight: [M, 1] or [M, T], assuming static portfolio weights


class TradingStrategy(object):
    def __init__(self, strategy: str, look_back: int or tuple, investment_horizon:tuple, buy_signal_bounds:tuple):
        assert strategy in ["buy-hold", "MA", "MOM"], ValueError("Invalid strategy!")
        self.strategy = strategy
        self.look_back = look_back # takes int for strategy == ["buy-hold", "MA"]; takes tuple for strategy == ["MOM"]
        self.start, self.end = investment_horizon
        self.signal_upper, self.signal_lower = buy_signal_bounds

    def get_portfolio_process(self, market_price_process, portfolio_weight):

        if portfolio_weight.shape() == market_price_process.shape(): # all [M, T]
            Vt = np.multiply(portfolio_weight, market_price_process).sum(axis=0) # [1, T]
        elif portfolio_weight.shape()[0] == market_price_process.shape()[0]:
            Vt = np.matmul(portfolio_weight.T, market_price_process) # [1,T]
        else:
            ValueError("market_price_process and portfolio_weight has mismatched dimension!")

        Rt = (Vt[1:-1] - Vt[0:-2])/Vt[0:-2] # [1, T-1]
        return Vt, Rt

    def get_strategy_PnL(self, market_price_process, portfolio_weight, return_signal=True):

        if self.strategy == "buy-hold":
            Vt, Rt = self.get_portfolio_process(market_price_process, portfolio_weight)
            PnL = Rt[self.start : self.end]

            return PnL

        elif self.strategy == "MA":
            Vt, Rt = self.get_portfolio_process(market_price_process, portfolio_weight)
            Rt_ma = utils.moving_average(Rt, window_size=self.look_back) # [1, T]

            sell_signal = -((Rt[self.start : self.end] - Rt_ma[self.start : self.end]) > self.signal_upper).astype(int)
            buy_signal = ((Rt[self.start : self.end] - Rt_ma[self.start : self.end]) < - self.signal_lower).astype(int)
            signal = np.linspace(0,0,num=len(buy_signal)) + buy_signal + sell_signal
            PnL = np.multiply(Rt[self.start : self.end], signal)
            if return_signal:
                return PnL, signal
            else:
                return PnL

        elif self.strategy == "MOM":
            Vt, Rt = self.get_portfolio_process(market_price_process, portfolio_weight)

            if len(self.look_back) == 1:
                Rt_ma_long = utils.moving_average(Rt, window_size=self.look_back)  # [1, T]
                Rt_ma_short = Rt

            else:
                Rt_ma_long = utils.moving_average(Rt, window_size=max(self.look_back))  # [1, T]
                Rt_ma_short = utils.moving_average(Rt, window_size=min(self.look_back))  # [1, T]

            sell_signal = -((Rt_ma_long[self.start : self.end] - Rt_ma_short[self.start : self.end]) > self.signal_upper).astype(int)
            buy_signal = ((Rt_ma_long[self.start : self.end] - Rt_ma_short[self.start : self.end]) < -self.signal_lower).astype(int)
            signal = np.linspace(0, 0, num=len(buy_signal)) + buy_signal + sell_signal
            PnL = np.multiply(Rt[self.start : self.end], signal)

            if return_signal:
                return PnL, signal
            else:
                return PnL











