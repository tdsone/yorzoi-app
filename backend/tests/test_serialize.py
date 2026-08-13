import io
import json

import numpy as np
import pytest

from app.core.serialize import build_ism_npz, build_npz, sig, sig_signed, to_json_values


class TestSignificantDigits:
    @pytest.mark.parametrize(
        "value,expected",
        [(0.0001234, 0.000123), (1.234567, 1.23), (122.14, 122.0), (0.0, 0.0)],
    )
    def test_known_values(self, value, expected):
        assert sig(np.array([value]))[0] == pytest.approx(expected, rel=1e-9)

    def test_relative_error_is_bounded(self):
        rng = np.random.default_rng(0)
        x = np.abs(rng.lognormal(-2, 1.5, 100_000)).astype(np.float32)
        rel = np.abs((sig(x) - x) / x)
        assert rel.max() < 5e-3

    def test_beats_decimal_rounding_on_small_values(self):
        # The reason the API rounds to significant digits: 3-decimal rounding sends
        # small positive coverage values to exactly zero.
        x = np.array([0.0004, 0.00012], dtype=np.float32)
        assert (np.round(x, 3) == 0).all()
        assert (sig(x) > 0).all()

    def test_never_produces_negative_or_nan(self):
        x = np.array([0.0, 1e-30, 1e-12, 1e30], dtype=np.float64)
        out = sig(x)
        assert np.isfinite(out).all()
        assert (out >= 0).all()

    def test_to_json_values_is_nested_lists(self):
        out = to_json_values(np.ones((2, 3), dtype=np.float32))
        assert out == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


class TestSignedSignificantDigits:
    def test_matches_sig_on_non_negative_input(self):
        x = np.array([0.0, 0.0001234, 1.234567, 122.14], dtype=np.float64)
        assert np.array_equal(sig_signed(x), sig(x))

    def test_preserves_sign(self):
        x = np.array([-0.0001234, -1.234567, 1.234567], dtype=np.float64)
        assert sig_signed(x).tolist() == pytest.approx([-0.000123, -1.23, 1.23], rel=1e-9)

    def test_attribution_shaped_input(self):
        # A mean-centered column sums to ~0 and mixes signs; rounding must not
        # collapse the negatives.
        col = np.array([0.6, -0.2, -0.2, -0.2])
        out = sig_signed(col)
        assert out.tolist() == [0.6, -0.2, -0.2, -0.2]


class TestIsmNpz:
    def test_round_trips_matrices_and_meta(self):
        result = {
            "meta": {"model": "yorzoi", "u0": 5.25},
            "matrices": {
                "alphabet": ["A", "C", "G", "T"],
                "start_bp": 100,
                "raw": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                "centered": [[-1.0, 0.5], [0.0, -0.5], [1.0, 0.0], [0.0, 0.0]],
                "projected": [0.5, -0.5],
            },
        }
        z = np.load(io.BytesIO(build_ism_npz(result)))
        assert sorted(z.files) == [
            "alphabet", "centered", "meta_json", "projected", "raw", "start_bp", "u0",
        ]
        assert z["raw"].shape == (4, 2)
        assert z["projected"].tolist() == [0.5, -0.5]
        assert z["start_bp"] == 100
        assert z["u0"] == 5.25
        assert json.loads(bytes(z["meta_json"]).decode())["model"] == "yorzoi"


class TestNpz:
    def _load(self, payload):
        return np.load(io.BytesIO(payload))

    def test_stacks_uniform_bin_counts(self):
        items = [
            {"id": "a", "bin_start_bp": 0, "values": np.ones((3, 10), np.float32)},
            {"id": "b", "bin_start_bp": 16, "values": np.full((3, 10), 2.0, np.float32)},
        ]
        z = self._load(build_npz(items, track_indices=[1, 2, 3], track_ids=list("xyz"), meta={}))
        assert z["predictions"].shape == (2, 3, 10)
        assert json.loads(bytes(z["meta_json"]).decode())["npz_layout"] == "stacked"
        assert z["bin_start_bp"].tolist() == [0, 16]
        assert z["sequence_ids"].tolist() == ["a", "b"]

    def test_splits_ragged_bin_counts(self):
        items = [
            {"id": "a", "bin_start_bp": 0, "values": np.ones((3, 10), np.float32)},
            {"id": "b", "bin_start_bp": 0, "values": np.ones((3, 7), np.float32)},
        ]
        z = self._load(build_npz(items, track_indices=[1, 2, 3], track_ids=list("xyz"), meta={}))
        assert "predictions" not in z.files
        assert z["pred_000"].shape == (3, 10)
        assert z["pred_001"].shape == (3, 7)
        assert json.loads(bytes(z["meta_json"]).decode())["npz_layout"] == "per_sequence"

    def test_values_survive_exactly(self):
        vals = np.array([[0.1234567, 122.14023, 6.5e-08]], dtype=np.float32)
        items = [{"id": "a", "bin_start_bp": 0, "values": vals}]
        z = self._load(build_npz(items, track_indices=[0], track_ids=["t"], meta={}))
        # Binary is the lossless path; unlike JSON there is no rounding here.
        assert np.array_equal(z["predictions"][0], vals)

    def test_meta_is_carried_inside_the_archive(self):
        items = [{"id": "a", "bin_start_bp": 0, "values": np.ones((1, 2), np.float32)}]
        z = self._load(
            build_npz(items, track_indices=[0], track_ids=["t"], meta={"model": "shorkie"})
        )
        assert json.loads(bytes(z["meta_json"]).decode())["model"] == "shorkie"
