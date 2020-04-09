from overrides import overrides
import torch


class NoisyRNN(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 rnn_cell_type=torch.nn.RNNCell,
                 std=.1,
                 batch_first=None):
        # batch_first is an unused argument to accomodate the interface.
        super().__init__()
        self._hidden_dim = hidden_dim
        self._std = std
        self._rnn_cell = rnn_cell_type(input_dim, hidden_dim)

    def _make_means_and_stds(self, batch_size):
        means = torch.zeros(batch_size, self._hidden_dim)
        stds = torch.Tensor(batch_size, self._hidden_dim)
        stds.fill_(self._std)
        return means, stds

    @overrides
    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape
        state = torch.zeros(batch_size, self._hidden_dim)
        cell_state = torch.zeros(batch_size, self._hidden_dim)
        states = []
        for t in range(sequence_length):
            means, stds = self._make_means_and_stds(batch_size)
            noise = torch.normal(means, stds)
            if isinstance(self._rnn_cell, torch.nn.LSTMCell):
                cell_state = cell_state + noise
                state, cell_state = self._rnn_cell(inputs[:, t, :],
                                                   [state, cell_state])
            else:
                state = state + noise
                state = self._rnn_cell(inputs[:, t, :], state)
            states.append(state)
        return torch.stack(states, dim=1), state


def make_noisy_rnn_type(rnn_cell_type):
    class CustomNoisyRNN(NoisyRNN):
        def __init__(self, *args, **kwargs):
            kwargs["rnn_cell_type"] = rnn_cell_type
            super().__init__(*args, **kwargs)
    return CustomNoisyRNN
