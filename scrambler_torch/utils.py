"""Sequence encoding helpers (counterparts of scrambler.utils)."""
import numpy as np

DNA = "ACGT"


def one_hot_encode(seqs, alphabet=DNA, seq_len=None):
    """Encode strings as (N, L, C) float32. Unknown letters (e.g. 'N', '$') are all-zero rows."""
    if isinstance(seqs, str):
        seqs = [seqs]
    seq_len = seq_len or max(len(s) for s in seqs)
    lut = np.full(256, -1, dtype=np.int64)
    for i, c in enumerate(alphabet):
        lut[ord(c.upper())] = lut[ord(c.lower())] = i
    out = np.zeros((len(seqs), seq_len, len(alphabet)), dtype=np.float32)
    for n, s in enumerate(seqs):
        codes = lut[np.frombuffer(s[:seq_len].encode(), dtype=np.uint8)]
        pos = np.nonzero(codes >= 0)[0]
        out[n, pos, codes[pos]] = 1.0
    return out


def sequence_template(template, alphabet=DNA):
    """'$' marks free positions; letters are fixed. Returns the (L, C) one-hot template."""
    return one_hot_encode(template, alphabet)[0]


def sequence_mask(template):
    """1 at free ('$') positions, 0 at fixed template letters."""
    return np.array([c == "$" for c in template], dtype=np.float32)


def background_from_data(x, pseudo_count=1.0):
    """Per-position letter frequencies of one-hot data x (N, L, C) with a pseudo count."""
    x = np.asarray(x)
    return (x.sum(0) + pseudo_count) / (x.shape[0] + x.shape[-1] * pseudo_count)
