
import numpy as np
import torch as tc
import torch




# def moving_average(array, window_size):
#     s = tc.cumsum(array, dim=1)
#     # print(s.shape,s)
#     s[:,window_size:] = s[:,window_size:] - s[:,-window_size].reshape(-1,1)
#     s[:,window_size-1:] = s[:,window_size-1:]/window_size
#     s[:,(window_size - 1)] = s[:,(window_size - 1)] / tc.arange(1, window_size)
#     return s

if torch.cuda.is_available():
    dev = "cuda"
elif torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"

PYTORCH_ENABLE_MPS_FALLBACK=1
dev = tc.device(dev)


def moving_average(x, window_size):
    '''
    :param x: [*, N, T] or [N, T], where N is the number of price scenarios, and T is the total time length
    :return:
    '''
    N=window_size
    zeros = torch.zeros(x.size()[0]).unsqueeze(1).to(device=dev)
    ma = torch.cat((zeros,x),dim=1)
    ma = torch.cumsum(ma.to("cpu"),dim=1).to(dev) # cumsum is not working on mps
#     print(ma)
#     print(ma[:,N:] - ma[:,:-N])
    ma = (ma[:,N:] - ma[:,:-N]) / float(N)
    ma = torch.cat((x[:,:N-1],ma), dim=1)
    return ma


def VaR(alpha, x):
    x = tc.reshape(x, shape=[-1, x.shape[-1]]).to("cpu")
    return tc.sort(x, dim=-1).values[:, round(alpha*x.shape[-1])-1].to(dev)




def ES(alpha, x):
    x = tc.reshape(x, shape=[-1, x.shape[-1]]).to("cpu")
    x_sorted = tc.sort(x, dim=-1).values
    ES = tc.mean(x_sorted[:, : round(alpha*x.shape[-1])], dim=-1).to(dev)
    return ES


def score(v,e,x,alpha=0.05):
    '''
    :param v: [N] VaR
    :param e: [N] ES
    :param x: [N, T] price scenario
    :return: [N, T, 1] score at each time step for each price scenario
    '''
    # print(v,e)
    # Setting W needs some discussion
    W = tc.max(tc.column_stack([e/v, tc.ones(size=x.shape).to(device=dev)]), dim=-1).values
    # can also use W = np.random.uniform(1, ES(alpha,x)/VaR(alpha,x), num=1)
    v_ = v.repeat(x.shape[-1], 1).T.unsqueeze(-1)
    e_ = e.repeat(x.shape[-1], 1).T.unsqueeze(-1)
    W_ = W.repeat(x.shape[-1], 1).T.unsqueeze(-1)
    x = x.unsqueeze(-1)

    # print(x.shape,v_.shape,e_.shape,W_.shape)

    s_ = W_ * ((x <= v_).long() - alpha) * (x ** 2 - v_ ** 2) / 2 + (x <= v_).long() * e_ * (v_ - x) + alpha * e_ * (e_ / 2 - v_)
    # print(tc.sum(s_))
    # print(s_.shape)
    return s_ # W*((x <= v).long() - alpha) * (x**2 - v**2)/2 + (x <= v).long() * e * (v-x) + alpha*e*(e/2 - v)




if __name__ == '__main__':
    array = tc.arange(20, dtype=tc.float32)
    array.shape
    moving_average(array, window_size=3)

    x = tc.normal(mean=0, std=1, size=[10,100], device=dev)
    ma = moving_average(x, window_size=3)
    ma.shape

    # x = tc.randn(size=[4,50])
    alpha = 0.1
    tc.sort(x, dim=-1)

    v = VaR(0.05, x)
    tc.sort(x, stable=True)
    e = ES(alpha, x)
    s = score(v, e, x, alpha)
    s.shape

