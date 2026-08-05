"""Shorkie adapter.

Two properties of the upstream model make this more than a thin wrapper:

* `Shorkie.from_tf_checkpoint` returns a model in **train** mode. One forward pass in
  that state produces noise *and* permanently rewrites the BatchNorm running
  statistics, so every later request on the same container is quietly degraded.
  `load()` therefore calls `.eval()` and asserts it.
* The network has no shape validation of any kind. A length other than `seq_length`
  can crash, return a silently empty tensor, or return the right bin count with a
  wrong prediction — the attention layer derives its relative-position basis from the
  runtime token count, so any other length silently leaves the trained regime.
  `forward()` asserts the exact input shape before every call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from app.core.spec import SHORKIE, ModelSpec

# S. cerevisiae index into the pretrained 165-species encoding. Upstream hardcodes
# this and the reference TensorFlow predictions were generated with the same value,
# so the equivalence result cannot validate it. Surfaced in the response meta.
SPECIES_INDEX = 109


class ShorkieAdapter:
    spec: ModelSpec = SHORKIE

    def __init__(self, weights: str | Path, params: str | Path, fold: str = "f0") -> None:
        self.weights = Path(weights)
        self.params = Path(params)
        self.fold = fold
        self._model = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str = "cpu") -> None:
        import torch

        from ._shorkie_net import Shorkie

        config = json.loads(self.params.read_text())["model"]

        if self.weights.suffix == ".safetensors":
            from safetensors.torch import load_file

            model = Shorkie(config)
            model.load_state_dict(load_file(str(self.weights)))
        else:
            model = Shorkie.from_tf_checkpoint(config, str(self.weights))

        model.eval()
        if model.training:
            raise RuntimeError("model did not enter eval mode")
        model.to(device)

        if model.head.out_features != self.spec.n_tracks:
            raise RuntimeError(
                f"checkpoint has {model.head.out_features} output tracks, "
                f"spec declares {self.spec.n_tracks}"
            )

        # The port's fidelity to the TensorFlow reference is the whole reason to use
        # it; TF32 would trade that away for speed this model does not need.
        torch.backends.cuda.matmul.allow_tf32 = os.getenv("SHORKIE_ALLOW_TF32") == "1"
        torch.backends.cudnn.allow_tf32 = os.getenv("SHORKIE_ALLOW_TF32") == "1"

        self._model = model
        self._device = device

    def forward(self, batch: np.ndarray, track_idx: np.ndarray | None = None) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("adapter is not loaded")
        # The network has no shape validation and a wrong length does not necessarily
        # raise: 16380..16383 return the right bin count with a silently wrong
        # prediction, because attention derives its positional basis from the runtime
        # token count. Check the exact shape, not the output bin count.
        if batch.ndim != 3 or batch.shape[1:] != (4, self.spec.seq_length):
            raise ValueError(
                f"expected (B, 4, {self.spec.seq_length}), got {tuple(batch.shape)}"
            )

        tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self._device)
        with torch.inference_mode():
            out = self._model(tensor)          # (B, n_bins, n_tracks), fp32, no autocast
            if out.shape[1] != self.spec.n_bins:
                # Nearby input lengths can return a plausible bin count on a stretched
                # grid, so this is a second, independent check on the geometry.
                raise RuntimeError(
                    f"model returned {out.shape[1]} bins, spec declares {self.spec.n_bins}"
                )
            if track_idx is not None:
                # Slice on the compute device: a full shorkie tile is 18.7 MB and a
                # 64-track selection is 0.23 MB, so this is the difference between a
                # viable and a non-viable transfer.
                idx = torch.as_tensor(np.asarray(track_idx), dtype=torch.long, device=out.device)
                out = out.index_select(2, idx)
        return out.detach().to("cpu").numpy()

    def describe(self) -> dict:
        return {
            "model": self.spec.name,
            "fold": self.fold,
            "weights": self.weights.name,
            "species_index": SPECIES_INDEX,
            "species_index_verified": False,
        }
