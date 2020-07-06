import copy
import math

import torch
from torch import nn as nn
from torch.nn import functional as F

ACTIVATION_DICT = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'prelu': torch.nn.PReLU
}

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

