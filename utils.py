
import numpy as np
import torch as tc



def moving_average(array, window_size):
    s = tc.cumsum(array, dim=0)
    s[window_size:] = s[window_size:] - s[:-window_size]
    s[window_size-1:] = s[window_size-1:]/window_size
    s[:(window_size - 1)] = s[:(window_size - 1)] / tc.arange(1, window_size)
    return s


def VaR(alpha, x):
    return tc.sort(x)[0][round(alpha*len(x))-1]


def ES(alpha, x):
    var = VaR(alpha, x)
    x_sorted = tc.sort(x)[0]
    return tc.mean(x_sorted[x_sorted <= var])


def score(v,e,x,alpha):
    # Setting W needs some discussion
    W = tc.max(tc.Tensor([ES(alpha,x)/VaR(alpha,x), 1]), dim=0).values # can also use W = np.random.uniform(1, ES(alpha,x)/VaR(alpha,x), num=1)
    return W*((x <= v).long() - alpha) * (x**2 - v**2)/2 + (x <= v).long() * e * (v-x) + alpha*e*(e/2 - v)






if __name__ == '__main__':
    array = tc.arange(20, dtype=tc.float32)
    moving_average(array, window_size=3)

    x = tc.normal(mean=0, std=1, size=[100])
    alpha = 0.05
    v = VaR(0.05, x)
    e = ES(alpha, x)
    s = score(v, e, x, alpha)

