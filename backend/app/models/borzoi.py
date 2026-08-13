"""Borzoi and Flashzoi adapter, backed by johahi/borzoi-pytorch.

One adapter serves both variants: Flashzoi is a distillation of Borzoi with the same
input window, the same 6144 x 32 bp output grid and the same 7611-track human head,
differing only in the attention implementation and the precision it runs at.

* **Input layout.** borzoi-pytorch takes (B, 4, L) channels-first — exactly the
  engine's normal form, so unlike yorzoi there is no transpose here.

* **Output layout.** The model returns (B, n_tracks, n_bins); the engine's normal form
  is (B, n_bins, n_tracks). Track slicing happens on dim 1 *before* the device-to-host
  copy: a full tile is 7611 x 6144 floats (187 MB) and a 64-track selection is 1.6 MB.

* **Precision.** Borzoi runs fp32. Flashzoi's attention is flash_attn's MHA, which
  only implements fp16/bf16, so its forward must run under bf16 autocast — the
  upstream module disables autocast around the final head conv itself, so the head
  still computes in fp32. flash_attn also means Flashzoi is CUDA-only; load() refuses
  a CPU device rather than failing later inside the first request.

* **Value space.** Outputs stay in the model's native training space: each track was
  scaled and soft-clip squashed when the targets were built, and nothing here inverts
  that (the transform differs per track and its parameters live in the targets sheet,
  not the checkpoint). Positions and sequences are comparable; absolute magnitudes
  across tracks are not. The spec advertises this as value_space and
  depth_normalized=False.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from app.core.spec import BORZOI, FLASHZOI, ModelSpec

DEFAULT_REPOS = {
    "borzoi": "johahi/borzoi-replicate-0",
    "flashzoi": "johahi/flashzoi-replicate-0",
}

_SPECS: dict[str, ModelSpec] = {"borzoi": BORZOI, "flashzoi": FLASHZOI}


class BorzoiAdapter:
    def __init__(
        self,
        variant: str = "borzoi",
        repo: str | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in _SPECS:
            raise ValueError(f"variant must be one of {sorted(_SPECS)}, got {variant!r}")
        self.variant = variant
        self.spec = _SPECS[variant]
        # `repo` may also be a local snapshot directory; the Modal image bakes the
        # pinned revision at build so containers do no network I/O.
        self.repo = repo or DEFAULT_REPOS[variant]
        self.revision = revision
        self._model = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _autocast(self):
        import torch

        if self.variant != "flashzoi":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def load(self, device: str = "cpu") -> None:
        from borzoi_pytorch import Borzoi

        if self.variant == "flashzoi" and not device.startswith("cuda"):
            raise RuntimeError("flashzoi uses flash_attn and can only load on CUDA")

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

    def forward(self, batch: np.ndarray, track_idx: np.ndarray | None = None) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("adapter is not loaded")
        if batch.ndim != 3 or batch.shape[1:] != (4, self.spec.seq_length):
            raise ValueError(
                f"expected (B, 4, {self.spec.seq_length}), got {tuple(batch.shape)}"
            )

        tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
        tensor = tensor.to(self._device)

        with torch.inference_mode(), self._autocast():
            out = self._model(tensor)                    # (B, n_tracks, n_bins)
            if out.shape[2] != self.spec.n_bins:
                raise RuntimeError(
                    f"model returned {out.shape[2]} bins, spec declares {self.spec.n_bins}"
                )
            if track_idx is not None:
                idx = torch.as_tensor(np.asarray(track_idx), dtype=torch.long, device=out.device)
                out = out.index_select(1, idx)
            out = out.float().permute(0, 2, 1)           # -> (B, n_bins, n_selected)
        return out.detach().to("cpu").numpy()

    def describe(self) -> dict:
        return {
            "model": self.spec.name,
            "weights": self.repo,
            "revision": self.revision,
            "head": "human",
            "output_transform": "not_inverted",
        }
