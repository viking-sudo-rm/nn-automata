from overrides import overrides
import torch

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.nn.util import sequence_cross_entropy_with_logits

from src.modules.attention import attention


_MAX_WORDS = 5
_MAX_LEN = 1000


class PositionalEmbedder(torch.nn.Module):

    def __init__(self, vocab_size, word_embedding_dim, pos_embedding_dim):
        super().__init__()
        word_embedding = torch.nn.Embedding(vocab_size, word_embedding_dim)
        self._word_embedder = BasicTextFieldEmbedder({"tokens":
                                                      word_embedding})
        if pos_embedding_dim is not None:
            self.feature_dim = word_embedding_dim + pos_embedding_dim
            self._pos_embedding = torch.nn.Embedding(_MAX_LEN,
                                                     pos_embedding_dim)
        else:
            self.feature_dim = word_embedding_dim
            self._pos_embedding = None

    @overrides
    def forward(self, sentence):
        mask = get_text_field_mask(sentence)
        word_embeddings = self._word_embedder(sentence)

        if self._pos_embedding is not None:
            batch_size = word_embeddings.size(0)
            seq_len = word_embeddings.size(1)
            positions = torch.arange(seq_len)
            pos_embeddings = self._pos_embedding(positions)
            pos_embeddings = pos_embeddings.view(1, seq_len,
                                                 pos_embeddings.size(1))
            pos_embeddings = pos_embeddings.repeat(batch_size, 1, 1)

            features = torch.cat([word_embeddings, pos_embeddings], dim=2)
            return features, mask

        else:
            return word_embeddings, mask


class Encoder(torch.nn.Module):

    def __init__(self, feature_dim, rnn_dim):
        super().__init__()
        self._rnn = torch.nn.LSTM(feature_dim, rnn_dim, batch_first=True)

    @overrides
    def forward(self, features):
        encodings, rnn_state = self._rnn(features)
        rnn_state = [tensor.squeeze(dim=0) for tensor in rnn_state]
        return encodings, rnn_state


class Decoder(torch.nn.Module):

    def __init__(self, vocab_size, feature_dim, rnn_dim, masked, feedforward):
        super().__init__()
        self._masked = masked
        self._feedforward = feedforward

        if feedforward:
            self._query_transform = torch.nn.Linear(feature_dim + rnn_dim,
                                                    rnn_dim,
                                                    bias=False)
        else:
            self._rnn_cell = torch.nn.LSTMCell(1, rnn_dim)
            self._query_transform = torch.nn.Linear(rnn_dim, rnn_dim,
                                                    bias=False)
            self._key_transform = torch.nn.Linear(rnn_dim, rnn_dim, bias=False)

        self._classifier = torch.nn.Linear(rnn_dim, vocab_size)

    @overrides
    def forward(self, features, encodings, rnn_state):
        batch_size = encodings.size(0)
        seq_len = encodings.size(1)
        logits = []

        keys = self._key_transform(encodings)
        zeros = torch.zeros(batch_size, 1)

        for idx in range(seq_len):
            rnn_state = self._rnn_cell(zeros, rnn_state)
            query = self._query_transform(rnn_state[0]).unsqueeze(dim=1)
            attention_vector = attention(query, keys, encodings)
            logits.append(self._classifier(attention_vector))

        # for idx in range(seq_len):
        #     feature = features[:, idx, :]
        #     encoding = encodings[:, idx, :]
        #     full_feature = torch.cat([feature, encoding], dim=1)

        #     if self._feedforward:
        #         query = self._query_transform(full_feature)
        #     else:
        #         rnn_state = self._rnn_cell(full_feature, rnn_state)
        #         query = self._query_transform(rnn_state[0])
        #     query = query.unsqueeze(dim=1)

        #     if self._masked:
        #         seen_encodings = encodings[:, :idx + 1, :]
        #     else:
        #         seen_encodings = encodings

        #     attention_vector = attention(query, seen_encodings, seen_encodings)
        #     logits.append(self._classifier(attention_vector))

        return torch.stack(logits, dim=1)


class LSTMDecoder(torch.nn.Module):

    def __init__(self, vocab_size, feature_dim, rnn_dim):
        super().__init__()
        self._vocab_size = vocab_size
        self._feature_dim = feature_dim
        self._rnn_dim = rnn_dim

        self._rnn = torch.nn.LSTM(1, rnn_dim,
                                  batch_first=True)
        self._classifier = torch.nn.Linear(rnn_dim, vocab_size)

    @overrides
    def forward(self, features, encodings, rnn_state):
        batch_size = encodings.size(0)
        seq_len = encodings.size(1)
        # full_features = torch.cat([features, encodings], dim=2)
        zeros = torch.zeros(batch_size, seq_len, 1)
        rnn_state = [tensor.unsqueeze(dim=0) for tensor in rnn_state]
        hidden_states, _ = self._rnn(zeros, rnn_state)
        return self._classifier(hidden_states)


# TODO: http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html
class Seq2Seq(Model):

    def __init__(self,
                 vocab,
                 word_embedding_dim=5,
                 pos_embedding_dim=None,
                 rnn_dim=16,
                 masked=True,
                 disable_attention=False,
                 feedforward_decoder=False):
        super().__init__(vocab)
        vocab_size = vocab.get_vocab_size()
        self._pos_embedder = PositionalEmbedder(vocab_size,
                                                word_embedding_dim,
                                                pos_embedding_dim)

        feature_dim = self._pos_embedder.feature_dim
        self._encoder = Encoder(feature_dim, rnn_dim)
        if not disable_attention:
            self._decoder = Decoder(vocab_size,
                                    feature_dim,
                                    rnn_dim,
                                    masked,
                                    feedforward_decoder)
        else:
            self._decoder = LSTMDecoder(vocab_size, feature_dim, rnn_dim)

        self._acc = CategoricalAccuracy()
        # self._second_half_acc = CategoricalAccuracy()

    @overrides
    def forward(self, sentence, labels=None):
        # sentence_lens = torch.sum((sentence != 0).int(), axis=-1)

        # Call all the seq2seq modules.
        features, mask = self._pos_embedder(sentence)
        encodings, rnn_state = self._encoder(features)
        logits = self._decoder(features, encodings, rnn_state)

        predictions = torch.argmax(logits, dim=2).float()
        results = {
            "predictions": predictions,
        }

        if labels is not None:
            # midpoint_idx = labels.size(1) // 2
            # second_half_mask = torch.zeros_like(mask)
            # second_half_mask[:, midpoint_idx + 1:] = 1

            self._acc(logits, labels, mask)
            # self._second_half_acc(logits, labels, mask * second_half_mask)

            loss = sequence_cross_entropy_with_logits(logits, labels, mask)
            results["loss"] = loss

        return results

    @overrides
    def get_metrics(self, reset):
        return {
            "acc": self._acc.get_metric(reset),
            # "second_half_acc": self._second_half_acc.get_metric(reset),
        }
