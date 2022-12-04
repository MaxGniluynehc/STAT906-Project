import torch as tc
import numpy as np
import utils

tc.device("cuda")

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

        # if tc.backends.mps.is_available(): # on MacOS
        #     self.device = tc.device("mps")
        # elif tc.cuda.is_available(): # on WindowOS
        #     self.device = tc.device("cuda")
        # else:
        #     self.device= tc.device("cpu")

        self.device = tc.device("cuda")

    def get_portfolio_process(self, market_price_process: tc.Tensor,
                              portfolio_weight:tc.Tensor):
        '''
        :param market_price_process: [M, T]
        :param portfolio_weight: [1, M] or [M, T], the way a portfolio is constructed.
        :return: the portfolio value process (V_t) and return process (R_t), where
        R_t = (S_t - S_{t-1})/S_{t-1}.
        In our analysis, V_t is the price scenario and R_t is the PnL process.
        For now, we assume that V_t and R_t are given, because portfolios are pre-constructed.
        '''

        if portfolio_weight.shape == market_price_process.shape: # all [M, T]
            Vt = tc.multiply(portfolio_weight, market_price_process).sum(dim=0) # [1, T]
        elif portfolio_weight.shape[0] == market_price_process.shape[0]:
            Vt = tc.matmul(portfolio_weight.T, market_price_process) # [1,T]
        else:
            ValueError("market_price_process and portfolio_weight has mismatched dimension!")


        Rt = (Vt[1:-1] - Vt[0:-2])/Vt[0:-2] # [1, T-1]
        return Vt, Rt

    def _get_strategy_PnL(self, market_price_process, portfolio_weight, return_signal=False):
        '''
        (This is used if PnL process is not given, for now this is deprecated, please use
        the new function get_strategy_PnL() below. )
        :param market_price_process: [M, T]
        :param portfolio_weight: [1, M] or [M, T]
        :param return_signal: if true, also return the trade signals, default is true.
        :return: the PnL process if we trade following the self.strategy strategy.
        '''
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

    def get_strategy_PnL(self, prices, return_signal=False):
        '''
        :param Rt: The given return process of the pre-constructed portfolio
        :param return_signal: if true, also return the trade signals, default is true.
        :return: the PnL process if we trade following the self.strategy strategy, over the
        trading horizon from self.strat to self.end.
        '''

        Rt = prices[:,:]/prices[:,0].reshape(-1,1)

        if self.strategy == "buy-hold":
            PnL = tc.as_tensor(Rt, dtype=tc.float32, device=self.device)
            return PnL

        elif self.strategy == "MA":
            Rt_ma = utils.moving_average(Rt, window_size=self.look_back) # [1, T]

            sell_signal = -((Rt - Rt_ma) > self.signal_upper).long()
            buy_signal = ((Rt - Rt_ma) < - self.signal_lower).long()
            signal = (tc.linspace(0,0, steps=buy_signal.shape[-1]).cuda() + buy_signal + sell_signal).to(dtype=tc.int)
            PnL = tc.multiply(Rt, signal).to(dtype=tc.float32, device=self.device)
            if return_signal:
                return PnL, signal  # [1, T]
            else:
                return PnL  # [1, T]

        elif self.strategy == "MOM":
            if type(self.look_back) == int:
                Rt_ma_long = utils.moving_average(Rt, window_size=self.look_back)  # [1, T]
                Rt_ma_short = Rt  # [1, T]

            else:
                Rt_ma_long = utils.moving_average(Rt, window_size=max(self.look_back))  # [1, T]
                Rt_ma_short = utils.moving_average(Rt, window_size=min(self.look_back))  # [1, T]

            sell_signal = -((Rt_ma_long - Rt_ma_short) > self.signal_upper).long()
            buy_signal = ((Rt_ma_long - Rt_ma_short) < -self.signal_lower).long()
            signal = (tc.linspace(0, 0, steps=buy_signal.shape[-1]).cuda() + buy_signal + sell_signal).to(dtype=tc.int)
            PnL = tc.multiply(Rt, signal).to(dtype=tc.float32, device=self.device)

            if return_signal:
                return PnL, signal  # [1, T]
            else:
                return PnL  # [1, T]










