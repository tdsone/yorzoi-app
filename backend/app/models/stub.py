"""A deterministic stand-in used to test the API and engine without weights.

The output is a function of the *input bases at that bin's own coordinates*, which is
what makes it useful: a coordinate bug, an off-by-one crop, a transposed layout or a
mis-stitched tile all change the numbers, so the API tests catch them without loading
a 55 MB checkpoint or waiting on a CPU forward pass.
"""

from __future__ import annotations

import numpy as np

from app.core.spec import ModelSpec


class StubAdapter:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "cpu") -> None:
        self._loaded = True

    def forward(self, batch: np.ndarray, track_idx: np.ndarray | None = None) -> np.ndarray:
        spec = self.spec
        if batch.ndim != 3 or batch.shape[1:] != (4, spec.seq_length):
            raise ValueError(f"expected (B, 4, {spec.seq_length}), got {batch.shape}")

        # Drop the cropped flanks, then reduce each bin to its G+C count.
        core = batch[:, :, spec.crop_bp : spec.seq_length - spec.crop_bp]
        gc = core[:, 1, :] + core[:, 2, :]                       # (B, predicted_bp)
        binned = gc.reshape(batch.shape[0], spec.n_bins, spec.bin_size).sum(axis=2)

        channels = (
            np.arange(spec.n_tracks, dtype=np.float32)
            if track_idx is None
            else np.asarray(track_idx, dtype=np.float32)
        )
        return (binned[:, :, None] + channels[None, None, :] * 1e-3).astype(np.float32)

    def describe(self) -> dict:
        return {"model": self.spec.name, "weights": "stub", "fold": None}
