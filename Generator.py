import torch as tc
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, input_size, device="mps"):
        super(Generator, self).__init__()
        if tc.backends.mps.is_available():
            self.device = tc.device(device)
        else:
            Warning("GPU unavailable.")
            self.device = tc.device("cpu")





















