from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


def make_strings(num_examples, string_length):
    probabilities = torch.Tensor(num_examples, string_length)
    probabilities.fill_(.5)
    return torch.bernoulli(probabilities).int()


def compute_parities(strings):
    parities = []
    parity = torch.zeros(strings.size(0)).int()
    for time in range(strings.size(1)):
        # Using the in-place thing modifies the object itself.
        parity = parity ^ strings[:, time]
        parities.append(parity)
    return torch.stack(parities, 1)


class Model(nn.Module):

    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self._lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self._linear = nn.Linear(hidden_size, 1)

    def forward(self, strings):
        hidden, _ = self._lstm(strings)
        return self._linear(hidden)


def main():
    dataset_length = 1000
    string_length = 15
    hidden_size = 64
    batch_size = 16

    # Generate the data.
    strings = make_strings(dataset_length, string_length)
    parities = compute_parities(strings)
    strings = strings.unsqueeze(2).float()
    parities = parities.unsqueeze(2).float()

    # Create model.
    model = Model(hidden_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        print("=" * 10, "EPOCH", epoch, "=" * 10)
        perm = torch.randperm(dataset_length)
        strings = strings[perm]
        parities = parities[perm]
        print(strings.size())
        for batch, i in enumerate(range(0, len(strings) - batch_size, batch_size)):
            print("Batch", batch)
            string_batch = strings[i : i + batch_size]
            parity_batch = parities[i : i + batch_size]

            optimizer.zero_grad()
            predicted_parity_batch = model(string_batch)
            loss = criterion(predicted_parity_batch, parity_batch)
            loss.backward()
            optimizer.step()

            accuracy = ((predicted_parity_batch > 0) == parity_batch.byte()).float()

            print("\tLoss: %.2f" % torch.mean(loss).item())
            print("\tAcc: %.2f" % torch.mean(accuracy).item())

if __name__ == "__main__":
    main()
