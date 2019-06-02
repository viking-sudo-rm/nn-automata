import argparse
from collections import defaultdict
import numpy as np
import os

import torch
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer

from data_readers.wwr import WWRDatasetReader
from seq2seq import Seq2Seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--rnn_dim", type=int, default=10)
    parser.add_argument("--disable_attention", action="store_true")
    parser.add_argument("--feedforward_decoder", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=2)
    return parser.parse_args()


def evaluate(model, vocab, test_dataset):
    iterator = BucketIterator(batch_size=len(test_dataset),
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)
    val_generator = iterator(test_dataset)
    model(**next(val_generator))
    return model.get_metrics(reset=True)


def main(args):
    # Data parameters for train, val, and test taken from:
    # https://github.com/viking-sudo-rm/StackNN/blob/master/configs.py
    # lexicon = [str(idx) for idx in range(args.vocab_size)]
    lexicon = ["a", "b"]
    train_dataset_reader = WWRDatasetReader(800, 10, 2,
                                            max_length=12,
                                            vocabulary=lexicon,
                                            seq2seq=True)
    val_dataset_reader = WWRDatasetReader(100, 10, 2,
                                          max_length=12,
                                          vocabulary=lexicon,
                                          seq2seq=True)

    train_dataset = train_dataset_reader.build()
    val_dataset = val_dataset_reader.build()
    vocab = Vocabulary.from_instances(train_dataset + val_dataset)

    model = Seq2Seq(vocab,
                    word_embedding_dim=args.vocab_size,
                    rnn_dim=args.rnn_dim,
                    disable_attention=args.disable_attention,
                    feedforward_decoder=args.feedforward_decoder,
                    masked=False)

    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16,
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    filename = get_filename(args)

    if not args.no_train:
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          patience=args.patience,
                          num_epochs=args.num_epochs,
                          validation_dataset=val_dataset)
        metrics = trainer.train()
        with open("models/%s.th" % filename, "wb") as fh:
            torch.save(model.state_dict(), fh)

    else:
        metrics = {}
        model.load_state_dict(torch.load("models/%s.th" % filename))

    test_dataset_reader = WWRDatasetReader(100, 10, 2,
                                           max_length=80,
                                           vocabulary=lexicon,
                                           seq2seq=True)
    test_dataset = test_dataset_reader.build()
    test_metrics = evaluate(model, vocab, test_dataset)

    gen_dataset_reader = WWRDatasetReader(100, 50, 5,
                                          max_length=80,
                                          vocabulary=lexicon,
                                          seq2seq=True)
    gen_dataset = gen_dataset_reader.build()
    gen_metrics = evaluate(model, vocab, gen_dataset)

    metrics.update(test_acc=test_metrics["acc"])
    metrics.update(gen_acc=gen_metrics["acc"])
    return metrics


def get_filename(args):
    if args.filename is not None:
        return args.filename
    elif args.disable_attention:
        return "seq2seq-reverse"
    else:
        return "seq2seq-attn-reverse"


if __name__ == "__main__":
    args = parse_args()

    if args.num_trials is None:
        metrics = main(args)
        print(metrics)

    else:
        all_metrics = defaultdict(list)
        base_filename = get_filename(args)

        for i in range(args.num_trials):
            print("Starting trial #%d" % i)
            args.filename = os.path.join(base_filename, str(i))
            metrics = main(args)
            print(metrics)
            for key in metrics:
                all_metrics[key].append(metrics[key])

        print("metric", "mean", "max")
        test_accs = all_metrics["test_acc"]
        gen_accs = all_metrics["gen_acc"]
        print("test_acc", np.mean(test_accs), np.max(test_accs))
        print("gen_acc", np.mean(gen_accs), np.max(gen_accs))
