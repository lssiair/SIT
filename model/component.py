import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation_Fun(nn.Module):

    def __init__(self, act_name):
        super(Activation_Fun, self).__init__()
        if act_name == 'relu':
            self.act = nn.ReLU()
        if act_name == 'prelu':
            self.act = nn.PReLU()
        if act_name == 'sigmoid':
            self.act == nn.Sigmoid()

    def forward(self, x):
        return self.act(x)


class MLP(nn.Module):
    def __init__(self, in_size, out_size=None, normalization=False, act_name='prelu'):
        super(MLP, self).__init__()
        if out_size is None:
            out_size = in_size
        self.linear = nn.Linear(in_size, out_size)
        self.ln = LayerNorm(out_size) if normalization else nn.Sequential()
        self.activation = Activation_Fun(act_name)

    def forward(self, x):
        x = self.linear(x)
        x = self.ln(x)
        x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):

    def __init__(self, in_size, hidden_size=256, out_size=64, non_linear=True):
        super(SelfAttention, self).__init__()

        self.query = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        ) if non_linear else nn.Linear(in_size, out_size)

        self.key = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        ) if non_linear else nn.Linear(in_size, out_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, mask=None, interaction=True):

        assert len(query.shape) == 3

        query = self.query(query)  # batch_size seq_len d_model
        query = query / float(math.sqrt(query.shape[-1]))
        key = self.key(key)      # batch_size seq_len d_model
        attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)
        if mask is None and interaction is True:
            return attention  # for path scoring
        if mask is not None and interaction is True:
            attention = F.softmax(attention, dim=-1)
            attention = attention * mask  # setting the attention score of pedestrian who are not in the scene to zero
            attention = F.normalize(attention, p=1, dim=-1)  # normalizing the non-zero value

            return attention