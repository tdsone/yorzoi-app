"""The metric u: one scalar per (mutant) sequence, computed from binned predictions.

A metric resolves once per request into a `ResolvedMetric` — track indices the
adapter must produce plus a pure-numpy reduction — and is then applied to every
batch. The registry is the extension point: a future metric is a new resolver
receiving (params, MetricContext) and returning a ResolvedMetric; nothing else in
the pipeline changes.

Profile mean-normalization from `TrackSelection.reduce` is deliberately NOT used
here: it rescales by the window mean of each *mutant's own* prediction, so a
mutation that changed all tracks uniformly would score zero. The metric reads raw
per-bin values and averages across the selected channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.core.errors import (
    IntervalOutsidePredictedSpan,
    StrandAverageUnsupported,
    UnknownMetric,
)
from app.core.spec import ModelSpec
from app.core.tracks import DEFAULT_PRESET, TrackCatalog
from app.schemas import IsmMetricSpec


@dataclass(frozen=True)
class MetricContext:
    spec: ModelSpec
    catalog: TrackCatalog
    origin: int
    """Sequence-coordinate of the first predicted bp (see app/core/coords.py)."""
    seq_len: int


@dataclass(frozen=True)
class ResolvedMetric:
    track_indices: np.ndarray
    """Union of channels the adapter must produce (fwd plus, if strand-averaging on a
    stranded model, their strand pairs), ascending."""

    fwd_positions: np.ndarray      # positions into track_indices read by the fwd pass
    rc_positions: np.ndarray | None    # ... by the revcomp pass; None => no RC pass
    bin_lo: int
    bin_hi: int
    bin_weights: np.ndarray        # (bin_hi - bin_lo,) overlap bp per bin, fwd order
    normalize: bool                # True: divide by total overlap (agg="mean")
    describe: dict = field(default_factory=dict)

    def score(self, y: np.ndarray) -> np.ndarray:
        """(B, n_bins, n_selected) forward-orientation predictions -> (B,) u values."""
        return self._score(y, self.fwd_positions, self.bin_lo, self.bin_hi, self.bin_weights)

    def score_rc(self, y_rc: np.ndarray) -> np.ndarray:
        """Predictions for the reverse-complemented window.

        RC bin k covers forward bin n_bins-1-k, so the slice mirrors and the
        weights reverse; stranded channels were swapped to their pairs at resolve
        time via `rc_positions`.
        """
        n_bins = y_rc.shape[1]
        return self._score(
            y_rc,
            self.rc_positions,
            n_bins - self.bin_hi,
            n_bins - self.bin_lo,
            self.bin_weights[::-1],
        )

    def _score(self, y, positions, lo, hi, weights) -> np.ndarray:
        block = y[:, lo:hi, positions]                    # (B, n_interval_bins, n_pos)
        u = block.mean(axis=2, dtype=np.float64) @ weights
        return u / weights.sum() if self.normalize else u


def resolve_interval_mean(params: IsmMetricSpec, ctx: MetricContext) -> ResolvedMetric:
    spec, catalog = ctx.spec, ctx.catalog

    preset = params.preset
    if not (params.tracks or params.track_groups or preset):
        preset = DEFAULT_PRESET.get(spec.name)
    selection = catalog.resolve(
        spec,
        tracks=params.tracks,
        track_groups=params.track_groups,
        preset=preset,
        max_rows=None,
        normalize_profiles=False,
    )
    fwd = [int(i) for i in selection.indices]

    rc_positions: np.ndarray | None = None
    track_indices = np.asarray(fwd, dtype=np.int64)
    fwd_positions = np.arange(len(fwd), dtype=np.int64)
    strand_pairs_added = 0
    if params.strand_average and spec.stranded:
        pairs = []
        for i in fwd:
            pair = catalog.tracks[i].get("strand_pair")
            if pair is None:
                raise StrandAverageUnsupported(
                    f"track {i} ({catalog.tracks[i]['id']!r}) has no strand_pair; "
                    "strand averaging on a stranded model needs one for every "
                    "selected track",
                    field="metric.strand_average",
                    index=i,
                )
            pairs.append(int(pair))
        union = sorted(set(fwd) | set(pairs))
        strand_pairs_added = len(union) - len(fwd)
        position = {v: p for p, v in enumerate(union)}
        track_indices = np.asarray(union, dtype=np.int64)
        fwd_positions = np.asarray([position[i] for i in fwd], dtype=np.int64)
        rc_positions = np.asarray([position[i] for i in sorted(set(pairs))], dtype=np.int64)
    elif params.strand_average:
        # Unstranded model: the RC pass reads the same channels.
        rc_positions = fwd_positions

    span_lo = max(0, ctx.origin)
    span_hi = min(ctx.seq_len, ctx.origin + spec.predicted_bp)
    iv = params.interval
    if iv.start < span_lo or iv.end > span_hi:
        raise IntervalOutsidePredictedSpan(
            f"interval [{iv.start}, {iv.end}) must lie within the predicted span "
            f"[{span_lo}, {span_hi}) of this sequence",
            field="metric.interval",
            interval=[iv.start, iv.end],
            predicted_span=[span_lo, span_hi],
        )

    bin_lo = (iv.start - ctx.origin) // spec.bin_size
    bin_hi = -(-(iv.end - ctx.origin) // spec.bin_size)
    starts = ctx.origin + np.arange(bin_lo, bin_hi) * spec.bin_size
    weights = (
        np.minimum(starts + spec.bin_size, iv.end) - np.maximum(starts, iv.start)
    ).astype(np.float64)

    return ResolvedMetric(
        track_indices=track_indices,
        fwd_positions=fwd_positions,
        rc_positions=rc_positions,
        bin_lo=int(bin_lo),
        bin_hi=int(bin_hi),
        bin_weights=weights,
        normalize=params.agg == "mean",
        describe={
            "name": params.name,
            "interval": [iv.start, iv.end],
            "bins": [int(bin_lo), int(bin_hi)],
            "edge_weights": [float(weights[0]), float(weights[-1])],
            "interval_bp": float(weights.sum()),
            "n_tracks_selected": len(fwd),
            "strand_average": params.strand_average,
            "strand_pairs_added": strand_pairs_added,
            "agg": params.agg,
        },
    )


METRICS: dict[str, Callable[[IsmMetricSpec, MetricContext], ResolvedMetric]] = {
    "interval_mean": resolve_interval_mean,
}


def resolve_metric(params: IsmMetricSpec, ctx: MetricContext) -> ResolvedMetric:
    try:
        resolver = METRICS[params.name]
    except KeyError:
        raise UnknownMetric(
            f"unknown metric {params.name!r}; available: {', '.join(sorted(METRICS))}",
            field="metric.name",
        ) from None
    return resolver(params, ctx)
