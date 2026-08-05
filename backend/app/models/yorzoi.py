"""Yorzoi adapter.

Three things this adapter has to get right, all of which the previous implementation
got wrong or left implicit:

* **Input layout.** Yorzoi takes (B, L, 4), channels-last — its own docstring claims
  (N, 4, L) and is wrong; `get_embs_after_crop` permutes before a Conv1d(in_channels=4).
  The engine's normal form is (B, 4, L), so the adapter transposes. A wrong layout
  raises loudly here rather than corrupting silently.

* **The inverse transform must run in fp32, outside autocast.** `_borzoi_transform_inv`
  computes `384 + (y - 384) ** 2` before a `** (1/0.75)`. In fp16 that overflows to inf
  once `y - 384` exceeds ~256. The deployed service runs it inside
  `torch.autocast(device_type="cuda")`, which resolves to fp16.

* **Bins, not per-bp.** The network emits 300 bins of 10 bp. Un-binning is
  `repeat_interleave(10) / 10` — a pure upsample that multiplies the payload tenfold
  and adds nothing. The adapter stops after the inverse transform and leaves the model
  on its native 10 bp grid; per-bp expansion is a presentation concern.

  The inverse transform is nonlinear, so it must be applied per tile *before* overlap
  averaging. That is why it lives here rather than in the engine.

UNVERIFIED: `seq_length=4992` and `crop_bp=996` are inherited from the deployed
service. Yorzoi's own dataset code describes a nominal 5000 bp window with 1000 bp of
context, but 5000 cannot pass through the network (a U-Net skip mismatches at 625 vs
624 positions), so the served geometry is not the nominal one and `crop_bp` may carry a
4 bp systematic offset. There is no golden data for yorzoi in this repo, so nothing
here can check it. Surfaced in the response meta as `geometry_verified: false`.
"""

from __future__ import annotations

import numpy as np

from app.core.spec import YORZOI, ModelSpec

DEFAULT_REPO = "tom-ellis-lab/yorzoi"


class YorzoiAdapter:
    spec: ModelSpec = YORZOI

    def __init__(self, repo: str = DEFAULT_REPO, revision: str | None = None) -> None:
        self.repo = repo
        # Pin this. Without a revision the weights behind a rebuild can change with no
        # signal, and there is nothing for a cache key or a response to bind to.
        self.revision = revision
        self._model = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str = "cpu") -> None:
        import torch
        from yorzoi.model.borzoi import Borzoi

        kwargs = {"revision": self.revision} if self.revision else {}
        model = Borzoi.from_pretrained(self.repo, **kwargs)
        model.eval()
        if model.training:
            raise RuntimeError("model did not enter eval mode")
        model.to(device)

        self._model = model
        self._device = device
        self._verify_geometry()

    def _verify_geometry(self) -> None:
        """One forward on a dummy input, so a geometry mismatch fails at load rather
        than silently mislabelling coordinates on every response."""
        probe = np.zeros((1, 4, self.spec.seq_length), dtype=np.float32)
        out = self.forward(probe)
        if out.shape[1] != self.spec.n_bins:
            raise RuntimeError(
                f"model returned {out.shape[1]} bins, spec declares {self.spec.n_bins}; "
                "the served geometry does not match the ModelSpec"
            )
        if out.shape[2] != self.spec.n_tracks:
            raise RuntimeError(
                f"model returned {out.shape[2]} tracks, spec declares {self.spec.n_tracks}"
            )

    @staticmethod
    def _untransform(out):
        """Invert the training transform, staying on the model's 10 bp grid.

        yorzoi exposes only `untransform_then_unbin`, which also upsamples 10x. The
        inverse transform itself is the private `_borzoi_transform_inv`; if that name
        ever moves, fall back to the public function and decimate, since un-binning is
        `repeat_interleave(r) / r` and is therefore exactly reversible.
        """
        try:
            from yorzoi.utils import _borzoi_transform_inv

            return _borzoi_transform_inv(out)
        except ImportError:
            from yorzoi.utils import untransform_then_unbin

            resolution = YORZOI.bin_size
            return untransform_then_unbin(out, resolution=resolution)[..., ::resolution] * resolution

    def forward(self, batch: np.ndarray, track_idx: np.ndarray | None = None) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("adapter is not loaded")
        if batch.ndim != 3 or batch.shape[1:] != (4, self.spec.seq_length):
            raise ValueError(
                f"expected (B, 4, {self.spec.seq_length}), got {tuple(batch.shape)}"
            )

        tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
        tensor = tensor.permute(0, 2, 1).contiguous().to(self._device)   # -> (B, L, 4)

        with torch.inference_mode():
            out = self._model(tensor)                    # (B, n_tracks, n_bins), squashed
            if track_idx is not None:
                # Tracks are dim 1 here; slicing before the transform is both cheaper
                # and keeps the expensive elementwise work proportional to the request.
                idx = torch.as_tensor(np.asarray(track_idx), dtype=torch.long, device=out.device)
                out = out.index_select(1, idx)
            out = self._untransform(out.float())         # fp32; never under autocast
            # untransform_then_unbin divides by the resolution after repeat_interleave,
            # so dividing here makes each bin the per-bp average. Expanding to per-bp is
            # then a pure repeat_interleave and reproduces the deployed numbers exactly.
            out = out / self.spec.bin_size
            out = out.permute(0, 2, 1)                   # -> (B, n_bins, n_selected)
        return out.detach().to("cpu").float().numpy()

    def describe(self) -> dict:
        return {
            "model": self.spec.name,
            "weights": self.repo,
            "revision": self.revision,
            "geometry_verified": False,
        }
