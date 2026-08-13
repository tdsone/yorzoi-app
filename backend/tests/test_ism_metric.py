"""Metric resolution: interval->bin mapping, track selection, strand averaging."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import (
    IntervalOutsidePredictedSpan,
    StrandAverageUnsupported,
    UnknownMetric,
)
from app.core.spec import SHORKIE, YORZOI, ModelSpec
from app.core.tracks import TrackCatalog, load_catalog
from app.ism.metrics import MetricContext, resolve_interval_mean, resolve_metric
from app.schemas import IsmMetricSpec


def spec_params(**overrides) -> IsmMetricSpec:
    body = {"interval": {"start": 100, "end": 300}, "tracks": [0]}
    body.update(overrides)
    return IsmMetricSpec(**body)


def ctx_for(spec: ModelSpec, catalog, origin=0, seq_len=None) -> MetricContext:
    return MetricContext(
        spec=spec, catalog=catalog, origin=origin, seq_len=seq_len or spec.predicted_bp
    )


TINY = ModelSpec(
    name="tiny",
    seq_length=192,
    crop_bp=32,
    bin_size=16,
    n_tracks=4,
    stranded=True,
    value_space="coverage",
    depth_normalized=False,
    default_step=64,
    max_seq_len=1024,
)

TINY_CATALOG = TrackCatalog(
    model="tiny",
    n_tracks=4,
    groups={"plus": [0, 2], "minus": [2, 4]},
    tracks=[
        {"i": 0, "id": "p0", "strand_pair": 2},
        {"i": 1, "id": "p1", "strand_pair": 3},
        {"i": 2, "id": "m0", "strand_pair": 0},
        {"i": 3, "id": "m1", "strand_pair": 1},
    ],
)

UNPAIRED_CATALOG = TrackCatalog(
    model="tiny",
    n_tracks=4,
    groups={},
    tracks=[{"i": n, "id": f"t{n}"} for n in range(4)],
)


class TestIntervalToBins:
    def test_aligned_interval(self):
        m = resolve_interval_mean(
            spec_params(interval={"start": 32, "end": 64}), ctx_for(TINY, TINY_CATALOG)
        )
        assert (m.bin_lo, m.bin_hi) == (2, 4)
        assert m.bin_weights.tolist() == [16.0, 16.0]

    def test_partial_edge_bins_are_overlap_weighted(self):
        # [5, 27) at bin_size 16 touches bins 0 and 1 with 11 bp each.
        m = resolve_interval_mean(
            spec_params(interval={"start": 5, "end": 27}), ctx_for(TINY, TINY_CATALOG)
        )
        assert (m.bin_lo, m.bin_hi) == (0, 2)
        assert m.bin_weights.tolist() == [11.0, 11.0]
        assert m.describe["edge_weights"] == [11.0, 11.0]
        assert m.describe["interval_bp"] == 22.0

    def test_single_partial_bin(self):
        m = resolve_interval_mean(
            spec_params(interval={"start": 20, "end": 28}), ctx_for(TINY, TINY_CATALOG)
        )
        assert (m.bin_lo, m.bin_hi) == (1, 2)
        assert m.bin_weights.tolist() == [8.0]

    def test_negative_origin_shifts_the_grid(self):
        # A short sequence centred in the predicted span has origin < 0; bin j
        # covers [origin + j*bin, ...), so the same bp interval lands in later bins.
        m = resolve_interval_mean(
            spec_params(interval={"start": 0, "end": 16}),
            ctx_for(TINY, TINY_CATALOG, origin=-32, seq_len=64),
        )
        assert (m.bin_lo, m.bin_hi) == (2, 3)

    def test_interval_outside_predicted_span_is_rejected(self):
        with pytest.raises(IntervalOutsidePredictedSpan):
            resolve_interval_mean(
                spec_params(interval={"start": 100, "end": 200}),
                ctx_for(TINY, TINY_CATALOG, origin=0, seq_len=64),   # span [0, 64)
            )

    def test_interval_before_origin_is_rejected(self):
        with pytest.raises(IntervalOutsidePredictedSpan):
            resolve_interval_mean(
                spec_params(interval={"start": 0, "end": 16}),
                ctx_for(TINY, TINY_CATALOG, origin=16, seq_len=192),
            )


class TestScoring:
    def _metric(self, **overrides):
        return resolve_interval_mean(spec_params(**overrides), ctx_for(TINY, TINY_CATALOG))

    def test_mean_is_overlap_weighted(self):
        m = self._metric(interval={"start": 5, "end": 37})     # bins 0..3, weights 11,16,5
        y = np.zeros((1, TINY.n_bins, 1))
        y[0, 0, 0], y[0, 1, 0], y[0, 2, 0] = 1.0, 2.0, 4.0
        u = m.score(y)
        assert u[0] == pytest.approx((11 * 1 + 16 * 2 + 5 * 4) / 32)

    def test_sum_skips_normalization(self):
        m = self._metric(interval={"start": 5, "end": 37}, agg="sum")
        y = np.ones((1, TINY.n_bins, 1))
        assert m.score(y)[0] == pytest.approx(32.0)

    def test_tracks_are_averaged(self):
        m = self._metric(tracks=[0, 1], interval={"start": 0, "end": 16})
        y = np.zeros((1, TINY.n_bins, 2))
        y[0, 0, 0], y[0, 0, 1] = 2.0, 4.0
        assert m.score(y)[0] == pytest.approx(3.0)


class TestStrandAverage:
    def test_pair_closure_expands_the_selection(self):
        m = resolve_interval_mean(
            spec_params(tracks=[0], strand_average=True, interval={"start": 0, "end": 32}),
            ctx_for(TINY, TINY_CATALOG),
        )
        assert m.track_indices.tolist() == [0, 2]
        assert m.fwd_positions.tolist() == [0]
        assert m.rc_positions.tolist() == [1]
        assert m.describe["strand_pairs_added"] == 1

    def test_pair_closed_selection_adds_nothing(self):
        m = resolve_interval_mean(
            spec_params(tracks=[0, 2], strand_average=True, interval={"start": 0, "end": 32}),
            ctx_for(TINY, TINY_CATALOG),
        )
        assert m.track_indices.tolist() == [0, 2]
        assert m.fwd_positions.tolist() == [0, 1]
        assert m.rc_positions.tolist() == [0, 1]

    def test_unstranded_model_reads_the_same_channels(self):
        shorkie_catalog = load_catalog("shorkie")
        m = resolve_interval_mean(
            spec_params(strand_average=True),
            MetricContext(spec=SHORKIE, catalog=shorkie_catalog, origin=0, seq_len=SHORKIE.predicted_bp),
        )
        assert m.rc_positions is not None
        assert np.array_equal(m.rc_positions, m.fwd_positions)

    def test_missing_strand_pair_is_rejected(self):
        with pytest.raises(StrandAverageUnsupported):
            resolve_interval_mean(
                spec_params(strand_average=True),
                ctx_for(TINY, UNPAIRED_CATALOG),
            )

    def test_score_rc_mirrors_bins_and_swaps_channels(self):
        # Build y_rc explicitly as the prediction the RC window would produce for a
        # channel-symmetric model: rc bin k, pair channel <- fwd bin n-1-k, channel.
        m = resolve_interval_mean(
            spec_params(tracks=[0], strand_average=True, interval={"start": 5, "end": 37}),
            ctx_for(TINY, TINY_CATALOG),
        )
        rng = np.random.default_rng(1)
        y = rng.random((2, TINY.n_bins, 2))          # channels [0, 2] of the union
        y_rc = np.empty_like(y)
        y_rc[:, :, :] = y[:, ::-1, ::-1]             # mirror bins, swap 0 <-> 2
        assert np.allclose(m.score_rc(y_rc), m.score(y))

    def test_yorzoi_catalog_pairs_resolve(self):
        catalog = load_catalog("yorzoi")
        m = resolve_interval_mean(
            spec_params(tracks=[0, 5], strand_average=True),
            MetricContext(spec=YORZOI, catalog=catalog, origin=0, seq_len=YORZOI.predicted_bp),
        )
        pair0 = catalog.tracks[0]["strand_pair"]
        pair5 = catalog.tracks[5]["strand_pair"]
        assert m.track_indices.tolist() == sorted({0, 5, pair0, pair5})


class TestRegistry:
    def test_unknown_metric_is_rejected(self):
        # model_copy skips validation, standing in for a schema/worker version skew.
        params = spec_params().model_copy(update={"name": "does_not_exist"})
        with pytest.raises(UnknownMetric):
            resolve_metric(params, ctx_for(TINY, TINY_CATALOG))

    def test_default_selection_uses_the_model_preset(self):
        catalog = load_catalog("yorzoi")
        m = resolve_interval_mean(
            IsmMetricSpec(interval={"start": 100, "end": 300}),
            MetricContext(spec=YORZOI, catalog=catalog, origin=0, seq_len=YORZOI.predicted_bp),
        )
        # yorzoi's default preset (strand_profiles) spans all 162 tracks.
        assert m.describe["n_tracks_selected"] == 162
