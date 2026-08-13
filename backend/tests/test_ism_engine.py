"""ISM engine end-to-end against deterministic stub adapters.

The GC stub makes every expectation exact: mutating A/T -> G/C adds exactly one
count to exactly one bin, so the metric delta is a closed-form number and any
coordinate slip changes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core import coords
from app.core.spec import SHORKIE, YORZOI
from app.core.tracks import load_catalog
from app.ism.engine import place, plan_ism, run_ism
from app.models.stub import StubAdapter
from app.schemas import IsmRequest

SEQ = "ACGT" * 250          # 1000 bp


def make_request(**overrides):
    body = {
        "sequence": SEQ,
        "ism_window": {"start": 100, "end": 110},
        "metric": {"interval": {"start": 100, "end": 300}},
        "plots": {"include": False},
    }
    body.update(overrides)
    return IsmRequest(**body)


@pytest.fixture(scope="module")
def yorzoi_adapter():
    a = StubAdapter(YORZOI)
    a.load()
    return a


@pytest.fixture(scope="module")
def yorzoi_catalog():
    return load_catalog("yorzoi")


class TestPlacement:
    @pytest.mark.parametrize("spec", [SHORKIE, YORZOI], ids=lambda s: s.name)
    def test_matches_plan_origins_for_short_sequences(self, spec):
        for length in (spec.bin_size, 100, 999, 1000, spec.predicted_bp):
            origins, _ = coords.plan_origins(length, spec, "auto")
            assert [place(length, spec)] == origins, length

    @pytest.mark.parametrize("spec", [SHORKIE, YORZOI], ids=lambda s: s.name)
    def test_native_length_lands_on_crop(self, spec):
        assert place(spec.seq_length, spec) == spec.crop_bp

    @pytest.mark.parametrize("spec", [SHORKIE, YORZOI], ids=lambda s: s.name)
    def test_every_position_stays_inside_the_window(self, spec):
        # plan_origins would tile these lengths; place() must still keep the whole
        # sequence inside the single model input.
        for length in (spec.predicted_bp + 1, spec.seq_length - 1, spec.seq_length):
            origin = place(length, spec)
            window_lo = origin - spec.crop_bp
            assert window_lo <= 0
            assert window_lo + spec.seq_length >= length


class TestExactDeltas:
    def test_at_to_gc_mutation_raises_u_by_one_count(self, yorzoi_adapter, yorzoi_catalog):
        # Interval [100, 300) is 20 bins of weight 10; one extra G+C count in one
        # bin moves the overlap-weighted mean by 10/200 = 1/20.
        plan = plan_ism("yorzoi", make_request(), spec=YORZOI, catalog=yorzoi_catalog)
        res = run_ism(plan, yorzoi_adapter)
        # Position 100 is A (SEQ repeats ACGT). A->G and A->C gain a count; A->T is neutral.
        assert res.raw[2, 0] - res.u0 == pytest.approx(1 / 20)
        assert res.raw[1, 0] - res.u0 == pytest.approx(1 / 20)
        assert res.raw[3, 0] - res.u0 == pytest.approx(0.0, abs=1e-12)
        # Position 101 is C: losing it to A or T drops u; C->G is neutral.
        assert res.raw[0, 1] - res.u0 == pytest.approx(-1 / 20)
        assert res.raw[3, 1] - res.u0 == pytest.approx(-1 / 20)
        assert res.raw[2, 1] - res.u0 == pytest.approx(0.0, abs=1e-12)

    def test_sum_aggregation_scales_by_interval_bp(self, yorzoi_adapter, yorzoi_catalog):
        plan = plan_ism(
            "yorzoi",
            make_request(metric={"interval": {"start": 100, "end": 300}, "agg": "sum"}),
            spec=YORZOI,
            catalog=yorzoi_catalog,
        )
        res = run_ism(plan, yorzoi_adapter)
        assert res.raw[2, 0] - res.u0 == pytest.approx(10.0)   # one count in one 10 bp bin

    def test_mutations_outside_the_interval_do_not_move_u(self, yorzoi_adapter, yorzoi_catalog):
        plan = plan_ism(
            "yorzoi",
            make_request(
                ism_window={"start": 400, "end": 405},
                metric={"interval": {"start": 100, "end": 300}},
            ),
            spec=YORZOI,
            catalog=yorzoi_catalog,
        )
        res = run_ism(plan, yorzoi_adapter)
        assert np.allclose(res.raw, res.u0)
        assert np.allclose(res.projected, 0.0)

    def test_native_length_end_to_end(self, yorzoi_adapter, yorzoi_catalog):
        # Native placement puts bin 0 at crop_bp = 996 (NOT bin-aligned to zero:
        # 996 % 10 == 6), so the interval is chosen on the native grid.
        seq = "ACGT" * (YORZOI.seq_length // 4)
        plan = plan_ism(
            "yorzoi",
            make_request(
                sequence=seq,
                ism_window={"start": 996, "end": 1001},
                metric={"interval": {"start": 996, "end": 1196}},
            ),
            spec=YORZOI,
            catalog=yorzoi_catalog,
        )
        assert plan.origin == YORZOI.crop_bp
        res = run_ism(plan, yorzoi_adapter)
        assert res.raw[2, 0] - res.u0 == pytest.approx(1 / 20)      # A->G at 996 (A)

    def test_native_grid_weights_partial_first_bin(self, yorzoi_adapter, yorzoi_catalog):
        # Same setup with a zero-based interval: bin 0 covers [996, 1006), so the
        # overlap with [1000, 1200) is 6 bp and the A->G delta scales to 6/200.
        seq = "ACGT" * (YORZOI.seq_length // 4)
        plan = plan_ism(
            "yorzoi",
            make_request(
                sequence=seq,
                ism_window={"start": 1000, "end": 1001},
                metric={"interval": {"start": 1000, "end": 1200}},
            ),
            spec=YORZOI,
            catalog=yorzoi_catalog,
        )
        res = run_ism(plan, yorzoi_adapter)
        assert res.raw[2, 0] - res.u0 == pytest.approx(6 / 200)

    def test_mutations_in_the_crop_are_allowed_and_inert_for_the_stub(
        self, yorzoi_adapter, yorzoi_catalog
    ):
        # Native-length input: positions [0, crop_bp) are inside the model input but
        # outside the predicted span. A real model responds; the stub must not.
        seq = "ACGT" * (YORZOI.seq_length // 4)
        plan = plan_ism(
            "yorzoi",
            make_request(
                sequence=seq,
                ism_window={"start": 0, "end": 5},
                metric={"interval": {"start": 1000, "end": 1200}},
            ),
            spec=YORZOI,
            catalog=yorzoi_catalog,
        )
        res = run_ism(plan, yorzoi_adapter)
        assert np.allclose(res.raw, res.u0)


class ACountAdapter(StubAdapter):
    """Counts A per bin: RC-asymmetric, so it exposes the revcomp machinery."""

    def forward(self, batch, track_idx=None):
        spec = self.spec
        core = batch[:, :, spec.crop_bp : spec.seq_length - spec.crop_bp]
        a_count = core[:, 0, :]
        binned = a_count.reshape(batch.shape[0], spec.n_bins, spec.bin_size).sum(axis=2)
        n = spec.n_tracks if track_idx is None else len(np.asarray(track_idx))
        return np.repeat(binned[:, :, None], n, axis=2).astype(np.float32)


class TestStrandAverage:
    def test_gc_stub_is_rc_invariant(self, yorzoi_adapter, yorzoi_catalog):
        # G+C content is unchanged by reverse complement, so strand averaging must
        # reproduce the plain run apart from float association order.
        base = run_ism(
            plan_ism("yorzoi", make_request(), spec=YORZOI, catalog=yorzoi_catalog),
            yorzoi_adapter,
        )
        averaged = run_ism(
            plan_ism(
                "yorzoi",
                make_request(
                    metric={"interval": {"start": 100, "end": 300}, "strand_average": True}
                ),
                spec=YORZOI,
                catalog=yorzoi_catalog,
            ),
            yorzoi_adapter,
        )
        assert np.allclose(averaged.raw, base.raw, atol=1e-9)

    def test_a_counter_averages_to_at_content(self, yorzoi_catalog):
        # For an A-counting model, the RC pass counts A in the revcomp = T in the
        # forward sequence over the mirrored interval, so u = (A + T)/2 over the
        # interval. On the AT-only prefix this is interval_bp/2 per bin -> mean 5.0.
        adapter = ACountAdapter(YORZOI)
        adapter.load()
        seq = "AT" * 200 + "GC" * 300         # 1000 bp; [0, 400) is A/T only
        plan = plan_ism(
            "yorzoi",
            make_request(
                sequence=seq,
                ism_window={"start": 100, "end": 102},
                metric={"interval": {"start": 100, "end": 300}, "strand_average": True},
            ),
            spec=YORZOI,
            catalog=yorzoi_catalog,
        )
        res = run_ism(plan, adapter)
        # Reference: interval [100, 300) is alternating AT: per bin 5 A's forward,
        # 5 T's (= 5 A's under RC), so u0 = (5 + 5)/2 = 5.0 per 10 bp bin.
        assert res.u0 == pytest.approx(5.0)
        # Position 100 is A. A->T: forward loses one A (-1 in one bin of 20),
        # RC gains one (+1); the average is unchanged.
        assert res.raw[3, 0] - res.u0 == pytest.approx(0.0, abs=1e-9)
        # A->G: forward loses one A, RC unchanged -> -0.5 * 1/20.
        assert res.raw[2, 0] - res.u0 == pytest.approx(-0.5 / 20)

    def test_shorkie_supports_strand_average(self):
        adapter = StubAdapter(SHORKIE)
        adapter.load()
        plan = plan_ism(
            "shorkie",
            make_request(
                metric={
                    "interval": {"start": 100, "end": 300},
                    "strand_average": True,
                    "tracks": [0],
                }
            ),
            spec=SHORKIE,
            catalog=load_catalog("shorkie"),
        )
        res = run_ism(plan, adapter)
        assert np.isfinite(res.u0)
