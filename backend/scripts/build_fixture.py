"""Extract a small, committable equivalence fixture from the golden TF predictions.

    uv run python scripts/build_fixture.py --src /path/to/shorkie-pytorch/data

The full oracle is ~16.8 GB of gitignored HDF5 that exists on one laptop, which makes
the load-bearing correctness test unrunnable anywhere else. This carves out a ~1 MB
slice that still fails loudly on a coordinate shift, a transposed layout, a wrong
track offset, a missing .eval(), or a precision regression.

Sequences come from chunk 009. Chunks 000-004 of that dataset are synthetic random
DNA and only 005-009 are genomic; genomic activations run ~12x larger (mean 3.6 vs
0.26, max 122 vs 18), which is the regime where a precision bug actually shows up.

Tracks are sampled across the whole 0..5214 range rather than taken as a prefix, so a
track-offset error cannot hide in an untested block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tests" / "fixtures"

N_SEQ = 4
N_TRACKS = 64
CHUNK = 9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="shorkie-pytorch/data directory")
    args = ap.parse_args()

    src = args.src / f"predictions_f0_chunk{CHUNK:03d}.h5"
    if not src.exists():
        raise SystemExit(f"missing {src}")

    track_idx = np.linspace(0, 5214, N_TRACKS).astype(np.int64)
    track_idx = np.unique(track_idx)

    with h5py.File(src, "r") as fh:
        try:
            onehot = fh["sequences"][:N_SEQ]          # (n, 16384, 4)
            preds = fh["predictions"][:N_SEQ]         # (n, 896, 5215)
        except OSError as exc:  # some chunks carry corrupted HDF5 blocks
            raise SystemExit(f"read failed: {exc}")
        attrs = {k: int(v) for k, v in fh.attrs.items()}

    bases = np.array(list("ACGT"))
    covered = onehot.sum(axis=2) > 0
    seqs = [
        "".join(np.where(covered[i], bases[onehot[i].argmax(axis=1)], "N").tolist())
        for i in range(N_SEQ)
    ]
    if any(len(s) != 16384 or "N" in s for s in seqs):
        raise SystemExit("unexpected fixture sequences (wrong length or contains N)")

    gold = preds[:, :, track_idx].astype(np.float32)  # (n, 896, n_tracks)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "shorkie_golden.npz",
        predictions=gold,
        track_indices=track_idx.astype(np.int32),
        sequences=np.array(seqs),
        # The exact tensor the TensorFlow model was fed, so the test can pin the
        # service's encoder against it directly. Without this the test decodes and
        # re-encodes with the same table, and any permutation of the channel order
        # cancels out — it would pass on a wrong encoder.
        onehot=onehot.astype(np.uint8),
    )
    (OUT / "shorkie_golden.json").write_text(
        json.dumps(
            {
                "source": f"predictions_f0_chunk{CHUNK:03d}.h5 (genomic)",
                "source_attrs": attrs,
                "fold": 0,
                "n_sequences": N_SEQ,
                "n_tracks": len(track_idx),
                "seq_length": 16384,
                "n_bins": 896,
                "generator": "TensorFlow Shorkie via modal_extract_weights.py",
                "stats": {
                    "mean": float(gold.mean()),
                    "max": float(gold.max()),
                    "min": float(gold.min()),
                },
            },
            indent=2,
        )
        + "\n"
    )
    size = (OUT / "shorkie_golden.npz").stat().st_size
    print(
        f"wrote {N_SEQ} sequences x {len(track_idx)} tracks x 896 bins "
        f"({size:,} bytes); gold mean {gold.mean():.3f} max {gold.max():.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
