"""ISM planning and execution: place -> mutate -> batch -> score -> assemble.

Single-window v1: one sequence, no tiling, `len(seq) <= spec.seq_length`. Every
sequence position lands inside the model input, so mutations outside the predicted
span are legal (they change predictions inside it); only the metric interval is
constrained to the predicted span, by the metric resolver.

Semantics mirror tangermeme's `saturation_mutagenesis`: `raw[a, p]` is u for the
sequence with position p set to base a (the reference base's "edit" is the unmutated
sequence, so its entry is u0); `centered` subtracts the per-position mean over the
four bases (the "hypothetical" view); `projected` is `centered` at the observed
bases, identically `u0 - raw.mean(axis=0)`. Deliberately not z-scored. A windowed
run is bit-identical to slicing a full run because edits are independent — asserted
in tests, not here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from app.core import coords, encode
from app.core.errors import (
    IsmSequenceTooLong,
    IsmUnsupportedModel,
    IsmWindowContainsN,
    IsmWindowInvalid,
    SequenceTooLong,
)
from app.core.spec import ModelSpec
from app.core.tracks import TrackCatalog
from app.ism import ISM_MODELS
from app.ism.metrics import MetricContext, ResolvedMetric, resolve_metric
from app.models.base import ModelAdapter
from app.schemas import IsmRequest

# Larger than predict's micro_batch: ISM inputs are all the same shape and arrive in
# thousands, so the GPU should see full batches. Overridable via ISM_BATCH_SIZE for
# tuning without a redeploy. The host copy per batch is
# batch * n_bins * n_selected_tracks * 4 bytes — an all-track shorkie selection moves
# ~19 MB per mutant, which is the real reason to select fewer tracks, not GPU memory.
ISM_BATCH = {"yorzoi": 64, "shorkie": 32}


def place(length: int, spec: ModelSpec) -> int:
    """Origin (sequence-coordinate of the first predicted bp) for one window.

    For `length <= predicted_bp` this reproduces `coords.plan_origins`'s single-tile
    placement exactly (tested against it); for lengths up to `seq_length` — which
    plan_origins would tile — the sequence is centred in the model input instead, and
    `length == seq_length` lands on `crop_bp`, the native path.
    """
    if length <= spec.predicted_bp:
        offset = ((spec.predicted_bp - length) // 2 // spec.bin_size) * spec.bin_size
        return -offset
    return spec.crop_bp - (spec.seq_length - length) // 2


@dataclass
class IsmPlan:
    id: str
    seq: str
    onehot: np.ndarray          # (4, L)
    window: np.ndarray          # (4, seq_length) reference model input
    origin: int
    ism_start: int              # sequence coords, half-open
    ism_end: int
    metric: ResolvedMetric
    strand_average: bool
    batch_size: int

    @property
    def n_mutants(self) -> int:
        return 3 * (self.ism_end - self.ism_start)

    @property
    def n_forwards(self) -> int:
        return (self.n_mutants + 1) * (2 if self.strand_average else 1)


def plan_ism(
    model: str, req: IsmRequest, *, spec: ModelSpec, catalog: TrackCatalog
) -> IsmPlan:
    """Validate everything and lay out the work. Pure numpy — this is what the
    gateway runs before spawning, so a bad request fails fast without a GPU."""
    if model not in ISM_MODELS:
        raise IsmUnsupportedModel(
            f"ISM is not available for {model!r}; supported: {', '.join(ISM_MODELS)}",
            field="model",
            model=model,
        )

    seq_id, raw = req.item()
    try:
        seq, onehot = encode.prepare(
            raw, min_len=spec.min_seq_len, max_len=spec.seq_length, field="sequence"
        )
    except SequenceTooLong as exc:
        raise IsmSequenceTooLong(
            f"ISM is single-window: the sequence must fit the {spec.seq_length} bp "
            f"model input; got {exc.details.get('length')} bp",
            field="sequence",
            length=exc.details.get("length"),
            max_length=spec.seq_length,
        ) from None

    length = len(seq)
    if req.ism_window is None:
        ism_start, ism_end = 0, length
    else:
        ism_start, ism_end = req.ism_window.start, req.ism_window.end
        if ism_end > length:
            raise IsmWindowInvalid(
                f"ism_window [{ism_start}, {ism_end}) extends past the {length} bp sequence",
                field="ism_window",
                window=[ism_start, ism_end],
                length=length,
            )

    zero_cols = onehot[:, ism_start:ism_end].sum(axis=0) == 0
    if zero_cols.any():
        pos = ism_start + int(zero_cols.argmax())
        raise IsmWindowContainsN(
            f"position {pos} is N; every position in the mutated window needs a "
            "reference base",
            field="sequence",
            position=pos,
        )

    origin = place(length, spec)
    metric = resolve_metric(
        req.metric,
        MetricContext(spec=spec, catalog=catalog, origin=origin, seq_len=length),
    )

    return IsmPlan(
        id=seq_id,
        seq=seq,
        onehot=onehot,
        window=coords.tile_onehot(onehot, origin, spec),
        origin=origin,
        ism_start=ism_start,
        ism_end=ism_end,
        metric=metric,
        strand_average=req.metric.strand_average,
        batch_size=int(os.getenv("ISM_BATCH_SIZE") or ISM_BATCH.get(model, 32)),
    )


@dataclass
class IsmResult:
    u0: float
    raw: np.ndarray             # (4, W); reference entries equal u0
    centered: np.ndarray        # (4, W); raw minus its per-position mean
    projected: np.ndarray       # (W,); centered at the observed bases
    ref_bases: np.ndarray       # (W,) int codes into ACGT
    timings_ms: dict


def revcomp(batch: np.ndarray) -> np.ndarray:
    """Reverse-complement one-hot windows. With rows in ACGT order, complementing is
    reversing the channel axis (A<->T, C<->G), then positions reverse."""
    return np.ascontiguousarray(batch[:, ::-1, ::-1])


def _score_batch(batch: np.ndarray, plan: IsmPlan, adapter: ModelAdapter) -> np.ndarray:
    metric = plan.metric
    u = metric.score(adapter.forward(batch, metric.track_indices))
    if plan.strand_average:
        u_rc = metric.score_rc(adapter.forward(revcomp(batch), metric.track_indices))
        u = 0.5 * (u + u_rc)
    return u


def run_ism(plan: IsmPlan, adapter: ModelAdapter) -> IsmResult:
    spec = adapter.spec
    started = time.perf_counter()
    u0 = float(_score_batch(plan.window[None], plan, adapter)[0])
    ref_done = time.perf_counter()

    width = plan.ism_end - plan.ism_start
    # Sequence position p occupies window column p - (origin - crop_bp); the whole
    # sequence fits inside the window for every placement `place` produces.
    col0 = plan.ism_start - (plan.origin - spec.crop_bp)
    ref = plan.onehot[:, plan.ism_start : plan.ism_end].argmax(axis=0)

    raw = np.empty((4, width), dtype=np.float64)
    raw[ref, np.arange(width)] = u0

    edits = [(p, a) for p in range(width) for a in range(4) if a != ref[p]]
    for lo in range(0, len(edits), plan.batch_size):
        chunk = edits[lo : lo + plan.batch_size]
        batch = np.repeat(plan.window[None], len(chunk), axis=0)
        for n, (p, a) in enumerate(chunk):
            batch[n, :, col0 + p] = 0.0
            batch[n, a, col0 + p] = 1.0
        u = _score_batch(batch, plan, adapter)
        for n, (p, a) in enumerate(chunk):
            raw[a, p] = u[n]

    ism_done = time.perf_counter()
    centered = raw - raw.mean(axis=0, keepdims=True)
    projected = centered[ref, np.arange(width)]
    return IsmResult(
        u0=u0,
        raw=raw,
        centered=centered,
        projected=projected,
        ref_bases=ref,
        timings_ms={
            "ref": round((ref_done - started) * 1e3, 1),
            "ism": round((ism_done - ref_done) * 1e3, 1),
        },
    )
