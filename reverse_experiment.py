import argparse

import torch
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer

from data_readers.wwr import WWRDatasetReader
from seq2seq import Seq2Seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--word_embedding_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=4)
    parser.add_argument("--disable_attention", action="store_true")
    parser.add_argument("--feedforward_decoder", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    return parser.parse_args()


def main(args):
    train_dataset_reader = WWRDatasetReader(100000, 10, 15, seq2seq=True)
    test_dataset_reader = WWRDatasetReader(100, 13, 15, seq2seq=True)

    train_dataset = train_dataset_reader.build()
    test_dataset = test_dataset_reader.build()
    vocab = Vocabulary.from_instances(train_dataset + test_dataset)

    model = Seq2Seq(vocab,
                    word_embedding_dim=args.word_embedding_dim,
                    rnn_dim=args.rnn_dim,
                    disable_attention=args.disable_attention,
                    feedforward_decoder=args.feedforward_decoder)

    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16,
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    filename = "seq2seq-reverse"

    if not args.no_train:
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          num_epochs=args.epochs,
                          validation_dataset=test_dataset)
        trainer.train()
        with open("models/%s.th" % filename, "wb") as fh:
            torch.save(model.state_dict(), fh)

    else:
        model.load_state_dict(torch.load("models/%s.th" % filename))

    # TODO: Can test the model here.


if __name__ == "__main__":
    args = parse_args()
    main(args)
