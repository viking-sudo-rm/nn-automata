from __future__ import print_function

import sys

import torch

from discrete_rnn import NormalizedDiscreteSRN, RandomizedDiscreteSRN
from utils import RNNModel


def main(save_path):
    rnn_module = RandomizedDiscreteSRN(1, 2, min_value=0, max_value=1)
    model = RNNModel(rnn_module)
    model.load_state_dict(torch.load(save_path))
    for key, value in model.state_dict().items():
        print(key, value)  # TODO: If scalar: value.item.


# TODO: Implement circuit here.


if __name__ == "__main__":
    main(sys.argv[1])  # TODO: Use argparse here.
