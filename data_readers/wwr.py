import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class WWRDatasetReader(DatasetReader):

    TOKENS = ["a", "b"]

    def __init__(self, num_strings, min_length, max_length):
        super().__init__(lazy=False)
        self._num_strings = num_strings
        self._min_length = min_length
        self._max_length = max_length
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    def build(self):
        return self.read(None)

    def get_random_tokens(self):
        length = random.randint(self._min_length, self._max_length)
        tokens = [random.choice(self.TOKENS)
                  for _ in range(length)]
        tokens.append("#")
        tokens.extend(reversed(tokens))
        return tokens

    def _read(self, options):
        for n in range(self._num_strings):
            tokens = self.get_random_tokens()
            yield self.text_to_instance(tokens)

    def text_to_instance(self, text):
        sentence = TextField([Token(word) for word in text[:-1]], self._token_indexers)
        labels = SequenceLabelField(text[1:], sequence_field=sentence)
        return Instance({
            "sentence": sentence,
            "labels": labels,
        })
