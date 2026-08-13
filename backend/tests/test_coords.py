import numpy as np
import pytest

from app.core import coords as C
from app.core.errors import BadRequest, NativeWindowUnavailable, TilingGap
from app.core.spec import BORZOI, SHORKIE, YORZOI

# FLASHZOI shares BORZOI's geometry dict, so sweeping it would test the same numbers.
SPECS = [SHORKIE, YORZOI, BORZOI]
STEPS = {
    SHORKIE.name: [16, 64, 7168, 8192, 14336],
    YORZOI.name: [10, 20, 500, 1000, 3000],
    # No tiny strides here: step=32 over a 524 kb input would materialise ~10k tiles
    # per merge just to prove alignment the 4096 stride already proves.
    BORZOI.name: [4096, 32_768, 98_304, 196_608],
}


def lengths_for(spec):
    out = list(range(1, 201))
    out += list(range(spec.predicted_bp - 64, spec.predicted_bp + 65))
    out += list(range(spec.seq_length - 64, spec.seq_length + 65))
    out += [300, 1000, 3000, 15_000, 20_000, 100_000, 131_072]
    return [n for n in out if 0 < n <= spec.max_seq_len]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
class TestExhaustiveTiling:
    """The property that matters: every base of the input lands in exactly the bins the
    response claims, for every length and every stride, on both models."""

    def test_full_coverage_no_gaps(self, spec):
        for length in lengths_for(spec):
            for step in STEPS[spec.name]:
                origins, mode = C.plan_origins(length, spec, "full", step)
                preds = [np.full((spec.n_bins, 2), i, np.float32) for i in range(len(origins))]
                values, start_bp = C.merge(preds, origins, spec, C.plan_clip(length, mode, spec))
                assert start_bp == 0, (spec.name, length, step)
                assert values.shape[0] == C.n_bins_for(length, spec), (spec.name, length, step)

    def test_origins_are_bin_aligned(self, spec):
        for length in lengths_for(spec):
            for step in STEPS[spec.name]:
                origins, _ = C.plan_origins(length, spec, "full", step)
                assert all(o % spec.bin_size == 0 for o in origins)

    def test_tile_windows_are_exactly_seq_length(self, spec):
        onehot = np.zeros((4, 20_000), np.float32)
        for length in [1, spec.bin_size, spec.predicted_bp, spec.seq_length, 20_000]:
            if length > spec.max_seq_len:
                continue
            origins, _ = C.plan_origins(length, spec, "full")
            for origin in origins:
                assert C.tile_onehot(onehot[:, :length], origin, spec).shape == (4, spec.seq_length)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
class TestWindowModes:
    def test_auto_picks_native_at_exact_window_length(self, spec):
        assert C.plan_origins(spec.seq_length, spec, "auto") == ([spec.crop_bp], "native")

    def test_auto_picks_full_otherwise(self, spec):
        _, mode = C.plan_origins(spec.seq_length - 1, spec, "auto")
        assert mode == "full"

    def test_full_at_window_length_produces_two_padded_tiles(self, spec):
        # Precisely why auto exists: every one of these tiles is padded on one side,
        # so their overlap averages N-contaminated predictions where one clean pass exists.
        origins, _ = C.plan_origins(spec.seq_length, spec, "full")
        assert len(origins) >= 2

    def test_native_rejects_other_lengths(self, spec):
        with pytest.raises(NativeWindowUnavailable):
            C.plan_origins(spec.seq_length - 1, spec, "native")

    def test_native_span_comes_from_origins_not_length(self, spec):
        # Deriving the span from the length would ask for ceil(seq_length/bin_size)
        # bins when a native pass only covers n_bins of them.
        origins, mode = C.plan_origins(spec.seq_length, spec, "native")
        preds = [np.ones((spec.n_bins, 2), np.float32)]
        values, start_bp = C.merge(preds, origins, spec, C.plan_clip(spec.seq_length, mode, spec))
        assert values.shape[0] == spec.n_bins
        # Reported in bp, so a crop that is not a whole number of bins (yorzoi: 996 at
        # 10 bp) survives instead of being truncated to a bin index.
        assert start_bp == spec.crop_bp


class TestShortSequences:
    def test_centred_in_the_predicted_span(self):
        origins, _ = C.plan_origins(1000, SHORKIE, "full")
        assert origins == [-6656]

    def test_pad_fraction_reported(self):
        origins, _ = C.plan_origins(1000, SHORKIE, "full")
        assert C.pad_fraction(1000, origins, SHORKIE) == pytest.approx(0.939, abs=1e-3)

    def test_sub_bin_sequence_gets_one_bin(self):
        origins, mode = C.plan_origins(5, SHORKIE, "full")
        values, start_bp = C.merge(
            [np.ones((SHORKIE.n_bins, 1), np.float32)], origins, SHORKIE, C.plan_clip(5, mode, SHORKIE)
        )
        assert values.shape[0] == 1
        assert start_bp == 0


class TestThresholdIsPredictedSpanNotWindow:
    """The previous implementation branched on the *window* length, so sequences
    between the predicted span and the window silently returned hard zeros."""

    def test_yorzoi_4000bp_is_tiled_not_zero_padded(self):
        origins, _ = C.plan_origins(4000, YORZOI, "full")
        assert len(origins) > 1

    @pytest.mark.parametrize("length", [3001, 4000, 4992])
    def test_yorzoi_gap_band_is_fully_covered(self, length):
        origins, mode = C.plan_origins(length, YORZOI, "full")
        preds = [np.ones((YORZOI.n_bins, 1), np.float32) for _ in origins]
        values, _ = C.merge(preds, origins, YORZOI, C.plan_clip(length, mode, YORZOI))
        assert values.shape[0] == C.n_bins_for(length, YORZOI)
        assert (values > 0).all()

    @pytest.mark.parametrize("length", [14337, 15000, 16384])
    def test_shorkie_gap_band_is_fully_covered(self, length):
        origins, mode = C.plan_origins(length, SHORKIE, "full")
        preds = [np.ones((SHORKIE.n_bins, 1), np.float32) for _ in origins]
        values, _ = C.merge(preds, origins, SHORKIE, C.plan_clip(length, mode, SHORKIE))
        assert (values > 0).all()


class TestMerge:
    def test_averages_overlaps_exactly(self):
        spec = SHORKIE
        origins = [0, spec.predicted_bp // 2]
        a = np.full((spec.n_bins, 1), 2.0, np.float32)
        b = np.full((spec.n_bins, 1), 4.0, np.float32)
        values, start_bp = C.merge([a, b], origins, spec)
        assert start_bp == 0
        overlap_start = origins[1] // spec.bin_size
        assert values[0] == pytest.approx(2.0)
        assert values[overlap_start] == pytest.approx(3.0)
        assert values[-1] == pytest.approx(4.0)

    def test_single_tile_is_passed_through_unchanged(self):
        pred = np.arange(SHORKIE.n_bins, dtype=np.float32).reshape(-1, 1)
        values, _ = C.merge([pred], [0], SHORKIE)
        assert np.array_equal(values, pred)

    def test_raises_rather_than_returning_silent_zeros(self):
        # A gap must be an error. Clamping the counter turns it into a block of
        # zeros indistinguishable from a real prediction.
        spec = SHORKIE
        origins = [0, spec.predicted_bp * 3]
        preds = [np.ones((spec.n_bins, 1), np.float32) for _ in origins]
        with pytest.raises(TilingGap):
            C.merge(preds, origins, spec)

    def test_rejects_clip_outside_covered_span(self):
        with pytest.raises(TilingGap):
            C.merge(
                [np.ones((SHORKIE.n_bins, 1), np.float32)],
                [0],
                SHORKIE,
                (0, (SHORKIE.n_bins + 10) * SHORKIE.bin_size),
            )

    def test_mismatched_tile_and_origin_counts(self):
        with pytest.raises(TilingGap):
            C.merge([np.ones((SHORKIE.n_bins, 1), np.float32)], [0, 16], SHORKIE)


class TestValidation:
    def test_rejects_unaligned_step(self):
        with pytest.raises(BadRequest):
            C.plan_origins(50_000, SHORKIE, "full", step=1000)  # not a multiple of 16

    def test_rejects_step_beyond_predicted_span(self):
        with pytest.raises(BadRequest):
            C.plan_origins(50_000, SHORKIE, "full", step=SHORKIE.predicted_bp + 16)

    def test_rejects_empty_sequence(self):
        with pytest.raises(BadRequest):
            C.plan_origins(0, SHORKIE)

    def test_rejects_bad_window_mode(self):
        with pytest.raises(BadRequest):
            C.plan_origins(1000, SHORKIE, "sideways")

    def test_tile_budget_is_checked_arithmetically(self):
        # The count must be rejected without materialising thousands of origins.
        with pytest.raises(BadRequest) as exc:
            C.plan_origins(131_072, SHORKIE, "full", step=16, max_tiles=64)
        assert exc.value.details["max_tiles"] == 64


class TestCoordinateMappingAgainstGroundTruth:
    """A sweep that fills each tile with a constant passes under *any* origin-to-bin
    mapping, including a wrong one. These tests define truth as a function of bp, so a
    phase error in the mapping produces a numerically detectable wrong answer."""

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
    @pytest.mark.parametrize("length", [5000, 40_000, 12_345])
    def test_merged_bins_carry_the_bp_they_claim(self, spec, length):
        if length > spec.max_seq_len:
            pytest.skip("beyond this model's limit")
        origins, mode = C.plan_origins(length, spec, "full")

        # Each tile reports the true bp coordinate of every bin it predicts.
        preds = [
            (origin + spec.bin_size * np.arange(spec.n_bins, dtype=np.float64)).reshape(-1, 1)
            for origin in origins
        ]
        values, start_bp = C.merge(preds, origins, spec, C.plan_clip(length, mode, spec))

        expected = start_bp + spec.bin_size * np.arange(values.shape[0])
        # Overlapping tiles average, and averaging agreeing coordinates is a no-op only
        # if every tile files its bins under the same grid.
        assert np.allclose(values[:, 0], expected), (
            f"{spec.name} L={length}: coordinate mapping is out of phase"
        )

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
    def test_tile_extraction_matches_the_declared_window(self, spec):
        rng = np.random.default_rng(0)
        length = spec.seq_length * 3
        onehot = np.eye(4, dtype=np.float32)[rng.integers(0, 4, length)].T

        for origin in C.plan_origins(length, spec, "full")[0]:
            tile = C.tile_onehot(onehot, origin, spec)
            lo = origin - spec.crop_bp
            overlap_start, overlap_end = max(0, lo), min(length, lo + spec.seq_length)
            assert np.array_equal(
                tile[:, overlap_start - lo : overlap_end - lo],
                onehot[:, overlap_start:overlap_end],
            )
            # Everything outside the sequence must be the all-zero N encoding.
            assert tile[:, : max(0, -lo)].sum() == 0
