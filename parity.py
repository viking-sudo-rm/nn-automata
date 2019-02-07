from __future__ import division, print_function

import torch
from torch import nn
import torch.nn.functional as F

from discrete_rnn import NormalizedDiscreteSRN, RandomizedDiscreteSRN
from utils import RNNModel


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


def make_dataset(dataset_length, string_length):
    strings = make_strings(dataset_length, string_length)
    parities = compute_parities(strings)  # compute_parities
    strings = strings.unsqueeze(2).float()
    parities = parities.unsqueeze(2).float()
    return strings, parities


class BasicLSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BasicLSTMModel, self).__init__()
        self._lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self._linear = nn.Linear(hidden_size, 1)

    def forward(self, strings):
        hidden, _ = self._lstm(strings)
        return self._linear(hidden)


def main():
    dataset_length = 1000
    string_length = 128
    hidden_size = 16
    batch_size = 16

    # Generate the data.
    strings, parities = make_dataset(dataset_length, string_length)
    strings_test, parities_test = make_dataset(dataset_length // 10, string_length)

    # Create model.
    # rnn_module = NormalizedDiscreteSRN(1, hidden_size)
    rnn_module = RandomizedDiscreteSRN(1, hidden_size, min_value=1, max_value=1)
    model = RNNModel(rnn_module)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(200):
        print("=" * 10, "EPOCH", epoch, "=" * 10)
        perm = torch.randperm(dataset_length)
        strings = strings[perm]
        parities = parities[perm]
        for batch, i in enumerate(range(0, len(strings) - batch_size, batch_size)):
            # print("Batch", batch)
            string_batch = strings[i : i + batch_size]
            parity_batch = parities[i : i + batch_size]

            optimizer.zero_grad()
            predicted_parity_batch = model(string_batch)
            loss = criterion(predicted_parity_batch, parity_batch)
            loss.backward()
            optimizer.step()

            # accuracy = ((predicted_parity_batch > 0) == parity_batch.byte()).float()
            # print("\tLoss: %.2f" % torch.mean(loss).item())
            # print("\tAcc: %.2f" % torch.mean(accuracy).item())

        predicted_parities_test = model(parities_test)
        accuracy = ((predicted_parities_test > 0) == parities_test.byte()).float()
        print("Test Acc: %.5f" % torch.mean(accuracy).item())

        # print("=" * 10, "PARAMS", epoch, "=" * 10)
        # print(model.state_dict())
        save_path = "models/parity-temp/epoch%d.dat" % epoch
        print("Saved parameters to", save_path)
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
