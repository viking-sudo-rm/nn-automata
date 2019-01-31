import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class ReLULSTM(nn.Module):

    """An LSTM with ReLU activations instead of sigmoid."""

    def __init__(self, input_size, hidden_size):
        super(ReLULSTM, self).__init__()

        self.hidden_size = hidden_size
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
        f = F.relu(self.weight_fx(x) + self.weight_fh(self.h))
        i = F.relu(self.weight_ix(x) + self.weight_ih(self.h))
        o = F.relu(self.weight_ox(x) + self.weight_oh(self.h))
        c_tilde = torch.tanh(self.weight_cx(x) + self.weight_ch(self.h))
        self.c = f * self.c + i * c_tilde
        self.h = o * torch.tanh(self.c)
        return self.h, self.c


    def init_hidden(self, batch_size):
        self.h = Variable(torch.zeros(batch_size, self.hidden_size))
        self.c = Variable(torch.zeros(batch_size, self.hidden_size))
