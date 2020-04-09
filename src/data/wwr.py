import numpy as np
import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class WWRDatasetReader(DatasetReader):

    def __init__(self,
                 num_strings,
                 mean_length,
                 std_length,
                 min_length=1,
                 max_length=None,
                 vocabulary=["a", "b"],
                 seq2seq=False):
        super().__init__(lazy=False)
        self._num_strings = num_strings
        self._mean_length = mean_length
        self._std_length = std_length
        self._min_length = min_length
        self._max_length = max_length
        self._vocabulary = vocabulary
        self._seq2seq = seq2seq
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    def build(self):
        return self.read(None)

    def _get_random_length(self):
        length = np.random.normal(self._mean_length, self._std_length)
        length = int(length)
        if self._min_length is not None:
            length = max(self._min_length, length)
        if self._max_length is not None:
            length = min(length, self._max_length)
        return length

    def get_random_tokens(self):
        length = self._get_random_length()
        tokens = [random.choice(self._vocabulary)
                  for _ in range(length)]
        return tokens

    def _read(self, options):
        for n in range(self._num_strings):
            tokens = self.get_random_tokens()
            yield self.text_to_instance(tokens)

    def text_to_instance(self, text):
        if self._seq2seq:
            reversed_text = list(reversed(text))
            sentence = TextField([Token(word) for word in text],
                                 self._token_indexers)
            labels = SequenceLabelField(reversed_text, sequence_field=sentence)

        else:
            text.append("#")
            text.extend(reversed(text))
            sentence = TextField([Token(word) for word in text[:-1]],
                                 self._token_indexers)
            labels = SequenceLabelField(text[1:], sequence_field=sentence)

        return Instance({
            "sentence": sentence,
            "labels": labels,
        })
