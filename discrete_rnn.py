from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import RNNModule


class RandomizedDiscreteRNN(RNNModule):

    def __init__(self, input_size, hidden_size, min_value, max_value):
        super(RandomizedDiscreteRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # TODO: Pass in distribution as an object.
        self.distribution = torch.distributions.uniform.Uniform(min_value, max_value)

    def compute_gate(self, operand):
        coefficients = self.distribution.sample(operand.shape)
        return torch.sigmoid(coefficients * operand)


class RandomizedDiscreteLSTM(RandomizedDiscreteRNN):

    """An LSTM with a random variable regularizer that is meant to discretize the computation."""

    def __init__(self, input_size, hidden_size, min_value, max_value):
        super(RandomizedDiscreteLSTM, self).__init__(input_size, hidden_size, min_value, max_value)

        self.h, self.c = None, None

        # LSTM weights.
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_cx = nn.Linear(input_size, hidden_size)
        self.weight_ox = nn.Linear(input_size, hidden_size)
        self.weight_fh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight_ih = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight_ch = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight_oh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        f = self.compute_gate(self.weight_fx(x) + self.weight_fh(self.h))
        i = self.compute_gate(self.weight_ix(x) + self.weight_ih(self.h))
        o = self.compute_gate(self.weight_ox(x) + self.weight_oh(self.h))
        c_tilde = torch.tanh(self.weight_cx(x) + self.weight_ch(self.h))
        self.c = f * self.c + i * c_tilde
        self.h = o * torch.tanh(self.c)
        return self.h, self.c

    def init_hidden(self, batch_size):
        self.h = Variable(torch.zeros(batch_size, self.hidden_size))
        self.c = Variable(torch.zeros(batch_size, self.hidden_size))


class RandomizedDiscreteSRN(RandomizedDiscreteRNN):

    def __init__(self, input_size, hidden_size, min_value, max_value):
        super(RandomizedDiscreteSRN, self).__init__(input_size, hidden_size, min_value, max_value)
        self.h = None
        self.weights_x = nn.Linear(input_size, hidden_size)
        self.weights_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        self.h = self.compute_gate(self.weights_x(x) + self.weights_h(self.h))
        return self.h, self.h

    def init_hidden(self, batch_size):
        self.h = Variable(torch.zeros(batch_size, self.hidden_size))


class NormalizedDiscreteSRN(RNNModule):

    def __init__(self, input_size, hidden_size):
        super(NormalizedDiscreteSRN, self).__init__()
        self.hidden_size = hidden_size
        self.h = None
        self.weights_x = nn.Linear(input_size, hidden_size)
        self.weights_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        vectors = self.weights_x(x) + self.weights_h(self.h)
        norms = vectors.norm(dim=1)
        self.h = torch.sigmoid(vectors / norms.unsqueeze(1))
        return self.h, self.h

    def init_hidden(self, batch_size):
        self.h = Variable(torch.zeros(batch_size, self.hidden_size))
