
import numpy as np
import torch as tc



def moving_average(array, window_size):
    s = tc.cumsum(array, dim=0)
    s[window_size:] = s[window_size:] - s[:-window_size]
    s[window_size-1:] = s[window_size-1:]/window_size
    s[:(window_size - 1)] = s[:(window_size - 1)] / tc.arange(1, window_size)
    return s


def VaR(alpha, x):
    x = tc.reshape(x, shape=[-1, x.shape[-1]])
    return tc.sort(x, dim=-1).values[:, round(alpha*x.shape[-1])-1]


def ES(alpha, x):
    x = tc.reshape(x, shape=[-1, x.shape[-1]])
    x_sorted = tc.sort(x, dim=-1).values
    ES = tc.mean(x_sorted[:, : round(alpha*x.shape[-1])], dim=-1)
    return ES


def score(v,e,x,alpha):
    # Setting W needs some discussion
    W = tc.max(tc.column_stack([ES(alpha,x)/VaR(alpha,x), tc.ones(size=ES(alpha,x).shape)]), dim=-1).values
    # can also use W = np.random.uniform(1, ES(alpha,x)/VaR(alpha,x), num=1)
    v_ = v.repeat(x.shape[-1], 1).T
    e_ = e.repeat(x.shape[-1], 1).T
    W_ = W.repeat(x.shape[-1], 1).T

    s_ = W_ * ((x <= v_).long() - alpha) * (x ** 2 - v_ ** 2) / 2 + (x <= v_).long() * e_ * (v_ - x) + alpha * e_ * (
                e_ / 2 - v_)

    return s_ # W*((x <= v).long() - alpha) * (x**2 - v**2)/2 + (x <= v).long() * e * (v-x) + alpha*e*(e/2 - v)



if __name__ == '__main__':
    array = tc.arange(20, dtype=tc.float32)
    moving_average(array, window_size=3)

    x = tc.normal(mean=0, std=1, size=[100])

    # x = tc.randn(size=[4,50])
    alpha = 0.1
    v = VaR(0.05, x)
    v.shape
    e = ES(alpha, x)
    s = score(v, e, x, alpha)
    s.shape

