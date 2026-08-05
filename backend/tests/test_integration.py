"""The real model through the real HTTP surface.

The stub-backed API tests prove routing and contracts; the equivalence tests prove the
numbers. This proves they compose — a real checkpoint, served over the real ASGI stack,
returning values that still match TensorFlow after JSON serialization.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from tests.test_equivalence import FIXTURE, checkpoint_paths, requires_weights

pytestmark = requires_weights


@pytest.fixture(scope="module")
def golden():
    return np.load(FIXTURE, allow_pickle=False)


@pytest.fixture(scope="module")
def client():
    from app.models.shorkie import ShorkieAdapter

    weights, params = checkpoint_paths()
    adapter = ShorkieAdapter(weights, params)
    adapter.load("cpu")
    return TestClient(create_app({"shorkie": adapter}))


def test_json_response_still_matches_tensorflow(client, golden):
    """Rounding to 3 significant digits must not cost more than ~0.5% relative."""
    seq = str(golden["sequences"][0])
    track = int(golden["track_indices"][0])
    r = client.post(
        "/v1/models/shorkie/predict",
        json={"sequences": [seq], "tracks": [track], "window": "native"},
    )
    assert r.status_code == 200
    body = r.json()

    got = np.array(body["predictions"][0]["values"][0])
    expected = golden["predictions"][0][:, 0]
    assert got.shape == expected.shape
    rel = np.abs(got - expected) / np.maximum(expected, 1e-9)
    assert rel.max() < 5e-3, f"max relative error after rounding: {rel.max():.2e}"


def test_npz_response_is_lossless(client, golden):
    seq = str(golden["sequences"][0])
    tracks = [int(i) for i in golden["track_indices"]]
    r = client.post(
        "/v1/models/shorkie/predict?format=npz",
        json={"sequences": [seq], "tracks": tracks, "window": "native"},
    )
    assert r.status_code == 200
    z = np.load(io.BytesIO(r.content))
    got = z["predictions"][0].T                       # (bins, tracks)
    assert np.allclose(got, golden["predictions"][0], rtol=1e-3, atol=1e-4)
    assert json.loads(bytes(z["meta_json"]).decode())["model"] == "shorkie"


def test_reported_geometry_is_honest(client, golden):
    seq = str(golden["sequences"][0])
    p = client.post(
        "/v1/models/shorkie/predict",
        json={"sequences": [seq], "tracks": [0]},
    ).json()["predictions"][0]
    # 16384 bp in: auto must take the clean single pass, and say where it starts.
    assert p["window"] == "native"
    assert p["bin_start_bp"] == 1024
    assert p["n_bins"] == 896
    assert p["pad_fraction"] == 0.0
    assert p["warnings"] == []


def test_aggregate_over_all_tracks_is_small_and_finite(client, golden):
    seq = str(golden["sequences"][0])
    r = client.post(
        "/v1/models/shorkie/predict",
        json={"sequences": [seq], "preset": "all", "mode": "aggregate"},
    )
    values = np.array(r.json()["predictions"][0]["values"])
    assert values.shape == (5215,)
    assert np.isfinite(values).all() and (values > 0).all()
    # The point of aggregate mode: 5215 numbers instead of 4.7 million. Measured at
    # 616 KB uncompressed and 62 KB gzipped, against ~90 MB for the full matrix.
    assert len(r.content) < 1_000_000


def test_short_sequence_is_warned_not_silently_padded(client):
    p = client.post(
        "/v1/models/shorkie/predict",
        json={"sequences": ["ACGT" * 100], "tracks": [0]},
    ).json()["predictions"][0]
    assert p["pad_fraction"] > 0.9
    assert "very_low_context" in p["warnings"]
    assert p["n_bins"] == 25
