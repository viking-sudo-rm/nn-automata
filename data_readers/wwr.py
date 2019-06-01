import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class WWRDatasetReader(DatasetReader):

    TOKENS = ["a", "b"]

    def __init__(self, num_strings, min_length, max_length, seq2seq=False):
        super().__init__(lazy=False)
        self._num_strings = num_strings
        self._min_length = min_length
        self._max_length = max_length
        self._seq2seq = seq2seq
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    def build(self):
        return self.read(None)

    def get_random_tokens(self):
        length = random.randint(self._min_length, self._max_length)
        tokens = [random.choice(self.TOKENS)
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
