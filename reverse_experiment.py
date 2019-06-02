import argparse

import torch
from allennlp.common.tqdm import Tqdm
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


def evaluate(model, vocab, test_dataset):
    iterator = BucketIterator(batch_size=len(test_dataset),
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)
    val_generator = iterator(test_dataset)
    model(**next(val_generator))
    return model.get_metrics(reset=True)


def main(args):
    train_dataset_reader = WWRDatasetReader(1000000, 20, 25, seq2seq=True)
    val_dataset_reader = WWRDatasetReader(100, 26, 27, seq2seq=True)

    train_dataset = train_dataset_reader.build()
    val_dataset = val_dataset_reader.build()
    vocab = Vocabulary.from_instances(train_dataset + val_dataset)

    model = Seq2Seq(vocab,
                    word_embedding_dim=args.word_embedding_dim,
                    rnn_dim=args.rnn_dim,
                    disable_attention=args.disable_attention,
                    feedforward_decoder=args.feedforward_decoder,
                    masked=False)

    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16,
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    if args.disable_attention:
        filename = "seq2seq-reverse"
    else:
        filename = "seq2seq-attn-reverse"

    if not args.no_train:
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          num_epochs=args.epochs,
                          validation_dataset=val_dataset)
        trainer.train()
        with open("models/%s.th" % filename, "wb") as fh:
            torch.save(model.state_dict(), fh)

    else:
        model.load_state_dict(torch.load("models/%s.th" % filename))

    test_dataset_reader = WWRDatasetReader(100, 50, 51, seq2seq=True)
    test_dataset = test_dataset_reader.build()
    metrics = evaluate(model, vocab, test_dataset)
    print(metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
