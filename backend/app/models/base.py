"""Adapter contract.

An adapter owns everything model-specific: how weights load, what device the tensors
live on, and any output reshaping or un-transforming. It hands the engine one shape:

    forward(batch: (B, 4, seq_length) float32) -> (B, n_bins, n_tracks) float32

Normalising to (bins, tracks) means the engine never branches on which model it is
serving. Shorkie is already in that layout; yorzoi's adapter unbins and transposes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.core.spec import ModelSpec


@runtime_checkable
class ModelAdapter(Protocol):
    spec: ModelSpec

    @property
    def loaded(self) -> bool: ...

    def load(self, device: str = "cpu") -> None:
        """Load weights. Must leave the model in eval mode."""

    def forward(self, batch: np.ndarray, track_idx: np.ndarray | None = None) -> np.ndarray:
        """(B, 4, seq_length) -> (B, n_bins, n_selected).

        `track_idx` selects output channels *on the compute device*, before any copy
        back to the host. Passing None returns all channels.
        """

    def describe(self) -> dict:
        """Provenance for the response meta block."""
