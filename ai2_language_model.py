from overrides import overrides

import torch
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Average, CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_readers.anbn import ANBNDatasetReader
from data_readers.wwr import WWRDatasetReader
from predictor import LanguageModelPredictor


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

        self._acc = CategoricalAccuracy()
        self._c_acc = CategoricalAccuracy()
        self._second_half_acc = CategoricalAccuracy()
    
    def forward(self, sentence, labels=None):
        mask = get_text_field_mask(sentence)
        embeddings = self._embedder(sentence)
        rnn_states, _ = self._rnn(embeddings)
        logits = self._ff(rnn_states)

        predictions = torch.argmax(logits, dim=2).float()
        results = {
            "predictions": predictions,
            "rnn_states": rnn_states,
        }

        if labels is not None:
            c_mask = (labels == self._c_idx).long()
            midpoint_idx = labels.size(1) // 2
            second_half_mask = torch.zeros_like(mask)
            second_half_mask[:, midpoint_idx + 1:] = 1
            self._acc(logits, labels, mask)
            self._c_acc(logits, labels, mask * c_mask)
            self._second_half_acc(logits, labels, mask * second_half_mask)
            loss = sequence_cross_entropy_with_logits(logits, labels, mask)
            results["loss"] = loss

        return results

    @overrides
    def get_metrics(self, reset):
        return {
            "acc": self._acc.get_metric(reset),
            "c_acc": self._c_acc.get_metric(reset),
            "second_half_acc": self._second_half_acc.get_metric(reset),
        }


def plot_rnn_states(predictor, sentence):
    results = predictor.predict(sentence)
    rnn_states = results["rnn_states"]
    cell_series_iter = zip(*rnn_states)

    for cell_series in cell_series_iter:
        plt.plot(cell_series)


def main():
    # Counting task.
    train_dataset = ANBNDatasetReader(5, 1000).build()
    test_dataset = ANBNDatasetReader(2000, 2200).build()

    # Reverse task.
    # train_dataset = WWRDatasetReader(1000, 50).build()
    # test_dataset = WWRDatasetReader(100, 100).build()

    vocab = Vocabulary.from_instances(train_dataset + test_dataset)  # This is just {a, b}.

    model_name = "srn-2"
    model = LanguageModel(vocab, rnn_type=torch.nn.RNN, rnn_dim=2)
    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    # trainer = Trainer(model=model,
    #                   optimizer=optimizer,
    #                   iterator=iterator,
    #                   train_dataset=train_dataset,
    #                   num_epochs=5,
    #                   validation_dataset=test_dataset,
    #                  )
    # trainer.train()

    with open("models/%s.th" % model_name, "wb") as fh:
        torch.save(model.state_dict(), fh)

    model.load_state_dict(torch.load("models/%s.th" % model_name))

    predictor = LanguageModelPredictor(model, ANBNDatasetReader(1, 1))
    n = 10
    sentence = " ".join(["a" for _ in range(n)] + ["b" for _ in range(n)])
    plot_rnn_states(predictor, sentence)
    plt.savefig("plots/%s.png" % model_name)

if __name__ == "__main__":
    main()
