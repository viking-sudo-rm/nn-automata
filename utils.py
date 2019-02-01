from abc import ABCMeta, abstractmethod, abstractproperty

import torch
from torch import nn


class RNNModel(nn.Module):

    """Model that uses RandomizedDiscreteLSTM."""

    def __init__(self, rnn_module):
        super(RNNModel, self).__init__()
        self.rnn_module = rnn_module
        self.linear = nn.Linear(rnn_module.hidden_size, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)
        self.rnn_module.init_hidden(batch_size)
        ys = []
        for t in range(sequence_length):
            h, _ = self.rnn_module(inputs[:, t])
            y = self.linear(h)
            ys.append(y)
        return torch.stack(ys, 1)


class RNNModule(nn.Module):

    @abstractmethod
    def init_hidden(self, batch_size):
        raise NotImplementedError
