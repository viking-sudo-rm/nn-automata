# nn-automata

This repository implements the formal language tasks that are referenced in my senior thesis/DELFOL submission.

Example usage for the anbn experiments:

```shell
DIM=4 allennlp train configs/anbn.jsonnet -s /tmp/rnn32 --include-package=src
```

Experiments with string reversal are unmaintained, but the original code is still there.
