"""Sequence normalization, validation, and one-hot encoding.

Order matters and is enforced by `prepare`: normalize, then *validate*, then encode.
The encoding LUT maps every byte outside ACGT to an all-zero column, which is exactly
how N is represented — so encoding an unvalidated sequence silently converts IUPAC
codes, digits and stray punctuation into N and predicts confidently on garbage.
"""

from __future__ import annotations

import re

import numpy as np

from .errors import InvalidAlphabet, SequenceTooLong, SequenceTooShort

# Anything not ACGT stays all-zero, which is the in-distribution encoding of N
# (baskerville's dna_1hot with n_uniform=False). Lowercase is folded to uppercase
# during normalization: soft-masked bases from SGD/UCSC are real bases, not N.
_LUT = np.zeros((256, 4), dtype=np.float32)
for _k, _b in enumerate(b"ACGT"):
    _LUT[_b, _k] = 1.0

_FASTA_HEADER = re.compile(r"^>.*$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
ALPHABET = frozenset("ACGTN")

_ALLOWED = np.zeros(256, dtype=bool)
for _b in b"ACGTN":
    _ALLOWED[_b] = True


def normalize(raw: str) -> str:
    """Strip FASTA headers and all whitespace, then uppercase."""
    return _WHITESPACE.sub("", _FASTA_HEADER.sub("", raw)).upper()


def validate(seq: str, *, min_len: int, max_len: int, field: str) -> None:
    """Raise a typed error if `seq` is not something the model can be fed."""
    if len(seq) < min_len:
        raise SequenceTooShort(
            f"sequence is {len(seq)} bp; the minimum is {min_len} bp",
            field=field,
            length=len(seq),
            min_length=min_len,
        )
    if len(seq) > max_len:
        raise SequenceTooLong(
            f"sequence is {len(seq)} bp; the maximum is {max_len} bp",
            field=field,
            length=len(seq),
            max_length=max_len,
        )
    try:
        codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError as exc:
        # Curly quotes and non-breaking spaces survive a paste from a word processor.
        pos = exc.start
        raise InvalidAlphabet(
            f"unsupported character {seq[pos]!r} at position {pos}; expected A, C, G, T or N",
            field=field,
            character=seq[pos],
            position=pos,
        ) from None

    offending = ~_ALLOWED[codes]
    if offending.any():
        pos = int(offending.argmax())
        raise InvalidAlphabet(
            f"unsupported character {seq[pos]!r} at position {pos}; expected A, C, G, T or N",
            field=field,
            character=seq[pos],
            position=pos,
        )


def dna_1hot(seq: str) -> np.ndarray:
    """(4, L) float32, channels-first, ACGT order. Requires a *validated* sequence."""
    codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return np.ascontiguousarray(_LUT[codes].T)


def prepare(raw: str, *, min_len: int, max_len: int, field: str) -> tuple[str, np.ndarray]:
    """normalize -> validate -> encode, in that order. The only encoding entrypoint
    callers on the request path should use."""
    seq = normalize(raw)
    validate(seq, min_len=min_len, max_len=max_len, field=field)
    return seq, dna_1hot(seq)


def decode_1hot(onehot: np.ndarray) -> str:
    """(4, L) or (L, 4) one-hot -> DNA string. All-zero columns decode to N.

    Used to turn stored one-hot fixtures back into API input for equivalence tests.
    """
    arr = onehot if onehot.shape[0] == 4 else onehot.T
    covered = arr.sum(axis=0) > 0
    idx = arr.argmax(axis=0)
    bases = np.array(list("ACGT"))
    out = np.where(covered, bases[idx], "N")
    return "".join(out.tolist())
