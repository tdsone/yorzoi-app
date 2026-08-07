"""Prediction pipeline: encode -> tile -> batch -> merge -> reduce.

Model-agnostic. Everything that differs between models lives in the `ModelSpec` or
behind the adapter, so this file has no per-model branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core import coords, encode
from app.core.errors import ResponseBudgetExceeded, TooManySequences
from app.core.spec import ModelSpec
from app.core.tracks import TrackSelection
from app.models.base import ModelAdapter

# Context below these thresholds makes absolute magnitudes untrustworthy: the model
# was never trained on windows that are mostly N, and padding measurably distorts the
# prediction over the real sequence, not just over the padding.
WARN_LOW_CONTEXT = 0.5
WARN_VERY_LOW_CONTEXT = 0.85
WARN_EXTREME_LOW_CONTEXT = 0.98


@dataclass
class Limits:
    max_sequences: int = 8
    max_total_bp: int = 262_144
    max_tiles: int = 64
    max_output_elements: int = 25_000_000
    micro_batch: int = 4

    @classmethod
    def for_spec(cls, spec: ModelSpec) -> "Limits":
        # Budget in tiles and output elements rather than bp: the same bp count means
        # wildly different compute and payload for a 4992 bp model with 162 tracks and
        # a 16384 bp model with 5215.
        return cls(max_total_bp=min(262_144, spec.max_seq_len * 2))


@dataclass
class SequenceInput:
    id: str
    seq: str


@dataclass
class Prediction:
    id: str
    length: int
    n_fraction: float
    window: str
    tiles: int
    bin_size_bp: int
    start_bp: int
    n_bins: int
    last_bin_partial: bool
    last_bin_bp: int
    pad_fraction: float
    warnings: list[str]
    values: np.ndarray                      # (n_rows, n_bins) or (n_rows,) when aggregated
    aggregated: bool = False


@dataclass
class PredictResult:
    predictions: list[Prediction]
    n_tiles: int


def _warnings_for(pad_fraction: float, n_fraction: float) -> list[str]:
    out = []
    if pad_fraction >= WARN_EXTREME_LOW_CONTEXT:
        out.append("extreme_low_context")
    elif pad_fraction >= WARN_VERY_LOW_CONTEXT:
        out.append("very_low_context")
    elif pad_fraction >= WARN_LOW_CONTEXT:
        out.append("low_context")
    # After encoding, an N is the same all-zero column as tile padding, so a
    # mostly-N sequence looks like context the model does not have.
    if n_fraction >= 0.5:
        out.append("mostly_n")
    return out


def plan(
    sequences: list[SequenceInput],
    spec: ModelSpec,
    selection: TrackSelection,
    *,
    window: str = "auto",
    step: int | None = None,
    aggregate: bool = False,
    limits: Limits | None = None,
) -> list[dict]:
    """Validate, encode and lay out the work. Raises before anything large is allocated."""
    limits = limits or Limits.for_spec(spec)

    if not sequences:
        raise TooManySequences("at least one sequence is required", field="sequences")
    if len(sequences) > limits.max_sequences:
        raise TooManySequences(
            f"{len(sequences)} sequences requested; the limit is {limits.max_sequences}",
            field="sequences",
            count=len(sequences),
            limit=limits.max_sequences,
        )

    plans: list[dict] = []
    total_bp = 0
    total_tiles = 0
    for n, item in enumerate(sequences):
        seq, onehot = encode.prepare(
            item.seq,
            min_len=spec.min_seq_len,
            max_len=spec.max_seq_len,
            field=f"sequences[{n}]",
        )
        total_bp += len(seq)
        if total_bp > limits.max_total_bp:
            raise ResponseBudgetExceeded(
                f"total input is {total_bp} bp; the limit is {limits.max_total_bp} bp",
                field="sequences",
                total_bp=total_bp,
                limit=limits.max_total_bp,
            )
        origins, mode = coords.plan_origins(
            len(seq), spec, window, step, max_tiles=limits.max_tiles
        )
        total_tiles += len(origins)
        plans.append(
            {"id": item.id, "seq": seq, "onehot": onehot, "origins": origins, "mode": mode}
        )

    if total_tiles > limits.max_tiles:
        raise ResponseBudgetExceeded(
            f"request needs {total_tiles} tiles; the limit is {limits.max_tiles}",
            field="sequences",
            tiles=total_tiles,
            limit=limits.max_tiles,
        )

    if not aggregate:
        elements = sum(
            selection.n_rows * coords.n_bins_for(len(p["seq"]), spec)
            if p["mode"] != "native"
            else selection.n_rows * spec.n_bins
            for p in plans
        )
        if elements > limits.max_output_elements:
            raise ResponseBudgetExceeded(
                f"response would hold {elements:,} values; the limit is "
                f"{limits.max_output_elements:,}. Narrow the track selection or use "
                "mode=aggregate.",
                field="tracks",
                requested_values=elements,
                limit=limits.max_output_elements,
            )
    return plans


def run(
    plans: list[dict],
    adapter: ModelAdapter,
    selection: TrackSelection,
    *,
    aggregate: bool = False,
    limits: Limits | None = None,
) -> PredictResult:
    """Execute a plan produced by `plan`."""
    spec = adapter.spec
    limits = limits or Limits.for_spec(spec)

    # Every tile from every sequence goes through the model together, so a request for
    # eight short sequences costs one batched pass rather than eight.
    tiles: list[np.ndarray] = []
    for item in plans:
        for origin in item["origins"]:
            tiles.append(coords.tile_onehot(item["onehot"], origin, spec))

    outputs: list[np.ndarray] = []
    for start in range(0, len(tiles), limits.micro_batch):
        batch = np.stack(tiles[start : start + limits.micro_batch])
        outputs.extend(adapter.forward(batch, selection.indices))

    predictions: list[Prediction] = []
    cursor = 0
    for item in plans:
        count = len(item["origins"])
        tile_preds = outputs[cursor : cursor + count]
        cursor += count

        length = len(item["seq"])
        values, start_bp = coords.merge(
            tile_preds,
            item["origins"],
            spec,
            coords.plan_clip(length, item["mode"], spec),
        )
        rows = selection.reduce(values.T)                    # (n_rows, n_bins)

        last_bin_bp = spec.bin_size
        last_bin_partial = False
        if item["mode"] != "native":
            remainder = length % spec.bin_size
            if remainder:
                last_bin_partial = True
                last_bin_bp = remainder

        pad = coords.pad_fraction(length, item["origins"], spec)
        n_fraction = float(1.0 - item["onehot"].sum() / max(length, 1))
        predictions.append(
            Prediction(
                id=item["id"],
                length=length,
                n_fraction=round(n_fraction, 4),
                window=item["mode"],
                tiles=count,
                bin_size_bp=spec.bin_size,
                start_bp=start_bp,
                n_bins=rows.shape[1],
                last_bin_partial=last_bin_partial,
                last_bin_bp=last_bin_bp,
                pad_fraction=round(pad, 4),
                warnings=_warnings_for(pad, n_fraction),
                values=rows.sum(axis=1) if aggregate else rows,
                aggregated=aggregate,
            )
        )

    return PredictResult(predictions=predictions, n_tiles=len(tiles))
