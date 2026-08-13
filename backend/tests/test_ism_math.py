"""ISM core math: mutant generation, centering, projection, windowing, revcomp."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.spec import YORZOI
from app.core.tracks import load_catalog
from app.ism.engine import plan_ism, revcomp, run_ism
from app.models.stub import StubAdapter
from app.schemas import IsmRequest

SEQ = "ACGT" * 250        # 1000 bp


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
def adapter():
    a = StubAdapter(YORZOI)
    a.load()
    return a


@pytest.fixture(scope="module")
def catalog():
    return load_catalog("yorzoi")


class RecordingAdapter:
    """Wraps an adapter and keeps every batch it was asked to run."""

    def __init__(self, inner):
        self.inner = inner
        self.spec = inner.spec
        self.batches: list[np.ndarray] = []

    def forward(self, batch, track_idx=None):
        self.batches.append(batch.copy())
        return self.inner.forward(batch, track_idx)


class TestMutantGeneration:
    def test_each_mutant_differs_at_exactly_one_column(self, adapter, catalog):
        recorder = RecordingAdapter(adapter)
        plan = plan_ism("yorzoi", make_request(), spec=YORZOI, catalog=catalog)
        run_ism(plan, recorder)

        reference = plan.window
        # First batch is the lone reference forward; the rest are mutants.
        mutants = np.concatenate(recorder.batches[1:], axis=0)
        assert mutants.shape[0] == plan.n_mutants == 30
        for mutant in mutants:
            diff_cols = np.flatnonzero((mutant != reference).any(axis=0))
            assert diff_cols.size == 1
            column = mutant[:, diff_cols[0]]
            assert column.sum() == 1.0 and set(column) <= {0.0, 1.0}

    def test_mutated_columns_cover_the_window_three_times(self, adapter, catalog):
        recorder = RecordingAdapter(adapter)
        plan = plan_ism("yorzoi", make_request(), spec=YORZOI, catalog=catalog)
        run_ism(plan, recorder)

        reference = plan.window
        col0 = plan.ism_start - (plan.origin - YORZOI.crop_bp)
        touched = []
        for mutant in np.concatenate(recorder.batches[1:], axis=0):
            touched.append(int(np.flatnonzero((mutant != reference).any(axis=0))[0]))
        expected = [col0 + p for p in range(10) for _ in range(3)]
        assert sorted(touched) == sorted(expected)


class TestAttributionIdentities:
    def test_reference_entries_equal_u0(self, adapter, catalog):
        plan = plan_ism("yorzoi", make_request(), spec=YORZOI, catalog=catalog)
        res = run_ism(plan, adapter)
        width = plan.ism_end - plan.ism_start
        assert np.allclose(res.raw[res.ref_bases, np.arange(width)], res.u0)

    def test_centered_columns_sum_to_zero(self, adapter, catalog):
        plan = plan_ism("yorzoi", make_request(), spec=YORZOI, catalog=catalog)
        res = run_ism(plan, adapter)
        assert np.allclose(res.centered.sum(axis=0), 0.0, atol=1e-12)

    def test_projected_is_u0_minus_column_mean(self, adapter, catalog):
        plan = plan_ism("yorzoi", make_request(), spec=YORZOI, catalog=catalog)
        res = run_ism(plan, adapter)
        assert np.allclose(res.projected, res.u0 - res.raw.mean(axis=0))
        width = plan.ism_end - plan.ism_start
        assert np.allclose(res.projected, res.centered[res.ref_bases, np.arange(width)])

    def test_hand_computed_centering(self):
        # centered = raw - column mean; projected = centered at the observed base.
        raw = np.array([[1.0, 8.0], [5.0, 8.0], [1.0, 8.0], [1.0, 4.0]])
        ref = np.array([0, 3])
        centered = raw - raw.mean(axis=0, keepdims=True)
        assert centered[:, 0].tolist() == [-1.0, 3.0, -1.0, -1.0]
        assert centered[:, 1].tolist() == [1.0, 1.0, 1.0, -3.0]
        assert centered[ref, np.arange(2)].tolist() == [-1.0, -3.0]


class TestWindowedEqualsFull:
    def test_windowed_run_is_a_slice_of_the_full_run(self, adapter, catalog):
        # Edits are independent, so restricting the window must be bit-identical
        # to running everything and slicing.
        full = run_ism(
            plan_ism("yorzoi", make_request(ism_window=None), spec=YORZOI, catalog=catalog),
            adapter,
        )
        part = run_ism(
            plan_ism(
                "yorzoi",
                make_request(ism_window={"start": 40, "end": 60}),
                spec=YORZOI,
                catalog=catalog,
            ),
            adapter,
        )
        assert np.array_equal(part.raw, full.raw[:, 40:60])
        assert np.array_equal(part.ref_bases, full.ref_bases[40:60])


class TestRevcomp:
    def test_flips_channels_and_positions(self):
        batch = np.zeros((1, 4, 6), dtype=np.float32)
        batch[0, 0, 0] = 1.0        # A at position 0
        batch[0, 1, 2] = 1.0        # C at position 2
        rc = revcomp(batch)
        assert rc[0, 3, 5] == 1.0   # ...becomes T at the last position
        assert rc[0, 2, 3] == 1.0   # ...becomes G at the mirrored position

    def test_is_an_involution(self):
        rng = np.random.default_rng(0)
        batch = rng.random((3, 4, 8)).astype(np.float32)
        assert np.array_equal(revcomp(revcomp(batch)), batch)
