from __future__ import print_function

from collections import defaultdict
import numpy as np
import pickle

import torch
from torch import nn
from torch.autograd import Variable

from relu_lstm import ReLULSTM


class BrokenModel(nn.Module):

    """LSTM model with ReLU activations."""

    def __init__(self, hidden_size):
        super(BrokenModel, self).__init__()
        self.relu_lstm = ReLULSTM(1, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)
        self.relu_lstm.init_hidden(batch_size)
        ys = []
        for t in range(sequence_length):
            h, _ = self.relu_lstm(inputs[:, t])
            y = self.linear(h)
            ys.append(y)
        return torch.stack(ys, 1)


class IntactModel(nn.Module):

    """LSTM model with sigmoid activations."""

    def __init__(self, hidden_size):
        super(IntactModel, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)
        h, _ = self.lstm(inputs)
        y = self.linear(h)
        return y


def make_strings(num_examples, string_length):
    probabilities = torch.Tensor(num_examples, string_length)
    probabilities.fill_(.5)
    return torch.bernoulli(probabilities).int()


def get_counts(strings):
    batch_size = strings.size(0)
    sequence_length = strings.size(1)
    counts = []
    count = torch.zeros(batch_size).int()
    for t in range(sequence_length):
        strip = torch.squeeze(strings[:, t])
        count = count + (2 * (strip.float() - .5)).int()
        counts.append(count > 0)
    return torch.stack(counts, 1)


def make_dataset(num_examples, string_length):
    strings = make_strings(num_examples, string_length)
    counts = get_counts(strings)
    strings = strings.unsqueeze(2).float()
    counts = counts.unsqueeze(2).float()
    return strings, counts


def main(epochs=10, length=32, hidden_size=4):
    # Make training data.
    strings, counts = make_dataset(1000, length)
    strings_test, counts_test = make_dataset(100, length)

    # Create model.
    model = BrokenModel(hidden_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Hyperparameters.
    batch_size = 10

    for epoch in range(epochs):
        print("=" * 10, "EPOCH", epoch, "=" * 10)
        perm = torch.randperm(len(strings))
        strings = strings[perm]
        counts = counts[perm]

        for batch, i in enumerate(range(0, len(strings) - batch_size, batch_size)):
            # print("Batch", batch)
            string_batch = strings[i : i + batch_size]
            count_batch = counts[i : i + batch_size]

            optimizer.zero_grad()
            predicted_count_batch = model(string_batch)
            loss = criterion(predicted_count_batch, count_batch)
            loss.backward()
            optimizer.step()

            accuracy = ((predicted_count_batch > 0) == count_batch.byte()).float()

            # print("\tLoss: %.2f" % torch.mean(loss).item())
            # print("\tAcc: %.2f" % torch.mean(accuracy).item())

        predicted_counts_test = model(strings_test)
        accuracy = ((predicted_counts_test > 0) == counts_test.byte()).float()
        print("Test Acc: %.2f" % torch.mean(accuracy).item())

    return torch.mean(accuracy).item()


def try_many_lengths():
    accuracies = defaultdict(list)
    for length in range(5, 35):
        for trial in range(3):
            print("=" * 20, "TRIAL", length, ".", trial, "=" * 20)
            accuracies[length].append(main(epochs=30, length=length, hidden_size=2))

    for length, accs in accuracies.items():
        print("%d: med=%.2f, max=%.2f" % (length, np.median(accs), max(accs)))

    with open("accuracies.dat", "w") as fh:
        pickle.dump(accuracies, fh)


if __name__ == "__main__":
    try_many_lengths()
