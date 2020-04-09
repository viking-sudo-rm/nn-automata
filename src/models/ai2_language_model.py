from overrides import overrides
import argparse
import random

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.dataloader import DataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits

from src.modules.noisy_rnn import make_noisy_rnn_type


_NUM_EMBEDDINGS = 5
_MODELS = {
    "srn": torch.nn.RNN,
    "gru": torch.nn.GRU,
    "lstm": torch.nn.LSTM,
    "noisy-srn": make_noisy_rnn_type(torch.nn.RNNCell),
    "noisy-lstm": make_noisy_rnn_type(torch.nn.LSTMCell),
    "noisy-gru": make_noisy_rnn_type(torch.nn.GRUCell),
    "seq2seq": None,
}


@Model.register("djanky_lm")
class LanguageModel(Model):

    def __init__(self,
                 vocab,
                 embedding_dim: int = 1,
                 rnn_dim: int = 4,
                 rnn_type: str = "srn"):
        super().__init__(vocab)
        self._vocab_size = vocab.get_vocab_size()
        self._c_idx = vocab.get_token_index("c")
        embedding = torch.nn.Embedding(_NUM_EMBEDDINGS, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})
        rnn_type = _MODELS[rnn_type]
        self._rnn = rnn_type(embedding_dim, rnn_dim, batch_first=True)
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
            # "second_half_acc": self._second_half_acc.get_metric(reset),
        }


# def plot_rnn_states(predictor, sentence):
#     results = predictor.predict(sentence)
#     rnn_states = results["rnn_states"]
#     cell_series_iter = zip(*rnn_states)

#     for cell_series in cell_series_iter:
#         plt.plot(cell_series)

#     plt.xlabel("Token Index")
#     plt.ylabel("Cell Value")


# def main(task_name="count",
#          model_name="srn",
#          rnn_dim=2,
#          num_epochs=5,
#          no_train=False,
#          plot=False):

#     if task_name == "count":
#         train_dataset_reader = ANBNDatasetReader(5, 1000)
#         test_dataset_reader = ANBNDatasetReader(2000, 2200)
#     elif task_name == "reverse":
#         train_dataset_reader = WWRDatasetReader(100000, 3, 20)
#         test_dataset_reader = WWRDatasetReader(100, 30, 32)
#     elif task_name == "copy":
#         train_dataset_reader = WWDatasetReader(100000, 3, 20)
#         test_dataset_reader = WWDatasetReader(100, 30, 32)

#     train_dataset = train_dataset_reader.build()
#     test_dataset = test_dataset_reader.build()
#     vocab = Vocabulary.from_instances(train_dataset + test_dataset)

#     if model_name == "seq2seq":
#         model = Seq2Seq(vocab)
#     else:
#         rnn_type = _MODELS[model_name]
#         model = LanguageModel(vocab, rnn_type=rnn_type, rnn_dim=rnn_dim)

#     optimizer = torch.optim.Adam(model.parameters())
#     iterator = BucketIterator(batch_size=16,
#                               sorting_keys=[("sentence", "num_tokens")])
#     iterator.index_with(vocab)

#     filename = "%s-%d" % (model_name, rnn_dim)
#     if task_name == "reverse":
#         filename += "-reverse"

#     if not no_train:
#         trainer = Trainer(model=model,
#                           optimizer=optimizer,
#                           iterator=iterator,
#                           train_dataset=train_dataset,
#                           num_epochs=num_epochs,
#                           validation_dataset=test_dataset)
#         trainer.train()
#         with open("models/%s.th" % filename, "wb") as fh:
#             torch.save(model.state_dict(), fh)

#     model.load_state_dict(torch.load("models/%s.th" % filename))

#     n = 11
#     if task_name == "count":
#         sentence = " ".join(["a" for _ in range(n)] + ["b" for _ in range(n)])
#     elif task_name == "reverse":
#         random.seed(2)
#         tokens = train_dataset_reader.get_random_tokens()
#         sentence = " ".join(tokens)

#     if plot:
#         predictor = LanguageModelPredictor(model, train_dataset_reader)
#         plot_rnn_states(predictor, sentence)
#         plt.savefig("plots/%s.png" % filename)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("task", choices=["count", "reverse", "copy"])
#     parser.add_argument("model", choices=_MODELS.keys())
#     parser.add_argument("--dim", type=int, default=2)
#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--notrain", action="store_true")
#     parser.add_argument("--plot", action="store_true")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     main(task_name=args.task,
#          model_name=args.model,
#          rnn_dim=args.dim,
#          num_epochs=args.epochs,
#          no_train=args.notrain,
#          plot=args.plot)
