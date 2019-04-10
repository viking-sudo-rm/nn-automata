from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class ANBNDatasetReader(DatasetReader):

    def __init__(self, min_n, max_n):
        super().__init__(lazy=False)
        self._min_n = min_n
        self._max_n = max_n
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    def build(self):
        return self.read(None)

    def _read(self, _):
        for n in range(self._min_n, self._max_n):
            tokens = ["a" for _ in range(n)]
            tokens.extend("b" for _ in range(n))
            tokens.append("c")
            yield self.text_to_instance(tokens)

    def text_to_instance(self, text):
        sentence = TextField([Token(word) for word in text[:-1]], self._token_indexers)
        labels = SequenceLabelField(text[1:], sequence_field=sentence)
        return Instance({
            "sentence": sentence,
            "labels": labels,
        })
