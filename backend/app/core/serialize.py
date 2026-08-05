"""Turning prediction arrays into wire formats.

JSON floats are rounded to significant digits, not decimals. Rounding coverage
predictions to 3 *decimals* is a data-destroying bug rather than a size knob: the
value distribution is dominated by small positive numbers, so ~0.1% of values round
to exactly 0.0 and the bottom decile is mangled (max relative error 1.0). Rounding to
3 *significant* digits is both smaller on the wire and ~200x more accurate.
"""

from __future__ import annotations

import io
import json

import numpy as np

# Values below this are reported as-is rather than scaled; softplus output is strictly
# positive but can be arbitrarily close to zero.
_TINY = 1e-12


def sig(x: np.ndarray, digits: int = 3) -> np.ndarray:
    """Round to `digits` significant digits. Input must be non-negative."""
    arr = np.asarray(x, dtype=np.float64)
    safe = np.maximum(arr, _TINY)
    scale = 10.0 ** (digits - 1 - np.floor(np.log10(safe)))
    return np.where(arr <= _TINY, 0.0, np.round(arr * scale) / scale)


def to_json_values(values: np.ndarray, digits: int = 3) -> list:
    """(n_tracks, n_bins) -> nested lists of rounded Python floats."""
    return sig(values, digits).tolist()


def build_npz(
    per_sequence: list[dict],
    *,
    track_indices: list[int],
    track_ids: list[str],
    meta: dict,
) -> bytes:
    """Serialize a prediction batch as an uncompressed .npz.

    `per_sequence` items carry `id`, `bin_start_bp` and `values` of shape
    (n_tracks, n_bins). When every sequence has the same bin count the arrays are
    stacked into a single `predictions` member of shape (n_seq, n_tracks, n_bins);
    otherwise each lands in its own `pred_%03d` member. `meta_json["npz_layout"]`
    records which, so a reader never has to guess.

    Not compressed: gzip on float32 buys ~1.1x for real CPU time, and the response is
    served with Content-Encoding: identity for the same reason.
    """
    bin_counts = {item["values"].shape[1] for item in per_sequence}
    stacked = len(bin_counts) == 1

    members: dict[str, np.ndarray] = {
        "track_indices": np.asarray(track_indices, dtype=np.int32),
        "track_ids": np.asarray(track_ids),
        "bin_start_bp": np.asarray([i["bin_start_bp"] for i in per_sequence], dtype=np.int32),
        "sequence_ids": np.asarray([i["id"] for i in per_sequence]),
    }
    if stacked:
        members["predictions"] = np.stack(
            [np.asarray(i["values"], dtype=np.float32) for i in per_sequence]
        )
    else:
        for n, item in enumerate(per_sequence):
            members[f"pred_{n:03d}"] = np.asarray(item["values"], dtype=np.float32)

    meta = {**meta, "npz_layout": "stacked" if stacked else "per_sequence"}
    members["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )

    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()
