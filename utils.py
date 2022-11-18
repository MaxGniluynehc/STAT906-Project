
import numpy as np


def moving_average(array, window_size):
    s = np.cumsum(array)
    s[window_size:] = s[window_size:] - s[:-window_size]
    s[window_size-1:] = s[window_size-1:]/window_size
    s[:(window_size - 1)] = s[:(window_size - 1)] / np.arange(1, window_size)
    return s


def VaR(alpha, x):
    return np.sort(x)[round(alpha*len(x))-1]


def ES(alpha, x):
    var = VaR(alpha, x)
    x_sorted = np.sort(x)
    return np.mean(x_sorted[x_sorted <= var])


def score(v,e,x,alpha):
    # Setting W needs some discussion
    W = max(ES(alpha,x)/VaR(alpha,x), 1) # can also use W = np.random.uniform(1, ES(alpha,x)/VaR(alpha,x), num=1)
    return W*((x <= v).astype(int) - alpha) * (x**2 - v**2)/2 + (x <= v).astype(int) * e * (v-x) + alpha*e*(e/2 - v)






if __name__ == '__main__':
    array = np.arange(20, dtype=float)
    moving_average(array, window_size=3)