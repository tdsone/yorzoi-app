"""Convert a Shorkie TensorFlow checkpoint to safetensors.

    uv run python scripts/convert_weights.py \
        --checkpoint ../../shorkie-pytorch/data/checkpoint_f0.h5 \
        --params ../../shorkie-pytorch/data/shorkie_params.json

Converting ahead of time keeps h5py off the serving path and makes container start a
state_dict load instead of an HDF5 walk. The conversion is asserted to be exact: the
safetensors model must reproduce the checkpoint-loaded model bit for bit.

Upstream checkpoints (one per cross-validation fold) live at
https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f{0..7}/model_best.h5
Note the `shorkie_models/` segment — the bucket layout moved, and the paths in the
shorkie-pytorch README still point at the old location, which returns NoSuchKey.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--fold", default="f0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    import sys

    sys.path.insert(0, str(HERE.parent))
    from app.models._shorkie_net import Shorkie

    config = json.loads(args.params.read_text())["model"]
    model = Shorkie.from_tf_checkpoint(config, str(args.checkpoint))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {n_params:,} parameters from {args.checkpoint.name}")

    out = args.out or DATA / f"shorkie_{args.fold}.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out))

    # Round-trip: a conversion that silently drops or reshapes a buffer would produce
    # a model that still runs and still looks plausible.
    check = Shorkie(config)
    check.load_state_dict(load_file(str(out)))
    check.eval()
    for (name, a), (_, b) in zip(model.state_dict().items(), check.state_dict().items()):
        if not torch.equal(a, b):
            raise SystemExit(f"round-trip mismatch in {name}")

    (DATA / "shorkie_params.json").write_text(args.params.read_text())
    meta = {
        "fold": args.fold,
        "source_checkpoint": args.checkpoint.name,
        "source_sha256": sha256(args.checkpoint),
        "safetensors_sha256": sha256(out),
        "n_parameters": n_params,
    }
    (DATA / f"shorkie_{args.fold}_provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out.name} ({out.stat().st_size:,} bytes); round-trip exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
