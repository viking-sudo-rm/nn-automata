import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class WWRDatasetReader(DatasetReader):

    _TOKENS = ["a", "b"]

    def __init__(self, num_strings, string_length):
        super().__init__(lazy=False)
        self._num_strings = num_strings
        self._string_length = string_length
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    def build(self):
        return self.read(None)

    def _read(self, options):
        for n in range(self._num_strings):
            tokens = [random.choice(self._TOKENS)
                      for _ in range(self._string_length)]
            tokens.append("#")
            tokens.extend(reversed(tokens))
            yield self.text_to_instance(tokens)

    def text_to_instance(self, text):
        sentence = TextField([Token(word) for word in text[:-1]], self._token_indexers)
        labels = SequenceLabelField(text[1:], sequence_field=sentence)
        return Instance({
            "sentence": sentence,
            "labels": labels,
        })
