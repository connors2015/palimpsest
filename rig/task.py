"""Synthetic sequence task for the toy transformer.

Delayed-copy: target[t] = tokens[t - LAG]. Predicting a token a fixed number
of positions back requires the model to (a) use position to locate the source
token and (b) copy it through the attention OV path — so both the positional
embedding and the attention head must do real work, unlike a bigram task a
lookup table could solve. Positions t < LAG are ignored in loss and metrics.
"""

import numpy as np

VOCAB = 8
CONTEXT = 8
LAG = 1


def make_batch(rng: np.random.Generator, batch: int,
               vocab: int = VOCAB, context: int = CONTEXT, lag: int = LAG):
    tokens = rng.integers(0, vocab, size=(batch, context))
    targets = np.empty_like(tokens)
    targets[:, lag:] = tokens[:, :context - lag]
    targets[:, :lag] = tokens[:, :lag]   # ignored by the loss mask below
    mask = np.zeros((batch, context), dtype=bool)
    mask[:, lag:] = True
    return tokens, targets, mask
