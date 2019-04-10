from overrides import overrides

import torch
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Average, CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from data_readers.anbn import ANBNDatasetReader
from data_readers.wwr import WWRDatasetReader


_NUM_EMBEDDINGS = 5


class LanguageModel(Model):

    def __init__(self,
                 vocab,
                 embedding_dim=4,
                 rnn_dim=4,
                 rnn_type=torch.nn.LSTM):
        super().__init__(vocab)
        self._vocab_size = vocab.get_vocab_size()
        self._c_idx = vocab.get_token_index("c")
        embedding = torch.nn.Embedding(_NUM_EMBEDDINGS, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})
        self._rnn = torch.nn.LSTM(embedding_dim, rnn_dim, batch_first=True)
        self._ff = torch.nn.Linear(rnn_dim, self._vocab_size)

        self._accuracy = CategoricalAccuracy()
        self._c_accuracy = CategoricalAccuracy()
    
    def forward(self, sentence, labels=None):
        mask = get_text_field_mask(sentence)
        embeddings = self._embedder(sentence)
        rnn_states, _ = self._rnn(embeddings)
        logits = self._ff(rnn_states)

        predictions = torch.argmax(logits, dim=2).float()
        results = {"predictions": predictions}

        if labels is not None:
            c_mask = (labels == self._c_idx).long()
            self._accuracy(logits, labels, mask)
            self._c_accuracy(logits, labels, mask * c_mask)
            loss = sequence_cross_entropy_with_logits(logits, labels, mask)
            results["loss"] = loss

        return results

    @overrides
    def get_metrics(self, reset):
        return {
            "accuracy": self._accuracy.get_metric(reset),
            "c_accuracy": self._c_accuracy.get_metric(reset)
        }


def main():
    # TODO: Try self attention: https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2seq_encoders/stacked_self_attention.py.

    # Counting task.
    # train_dataset = ANBNDatasetReader(5, 1000).build()
    # test_dataset = ANBNDatasetReader(2000, 2200).build()

    # Reverse task.
    train_dataset = WWRDatasetReader(1000, 50).build()
    test_dataset = WWRDatasetReader(100, 100).build()

    vocab = Vocabulary.from_instances(train_dataset + test_dataset)  # This is just {a, b}.

    model = LanguageModel(vocab, rnn_type=torch.nn.LSTM)
    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=100,
                      validation_dataset=test_dataset,
                      patience=10
                     )
    trainer.train()

if __name__ == "__main__":
    main()
