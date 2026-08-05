import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import MAX_ROWS_JSON, create_app
from app.core.spec import SHORKIE, YORZOI
from app.models.stub import StubAdapter

SEQ = "ACGT" * 250          # 1000 bp


@pytest.fixture(scope="module")
def client():
    adapters = {}
    for spec in (SHORKIE, YORZOI):
        adapter = StubAdapter(spec)
        adapter.load()
        adapters[spec.name] = adapter
    return TestClient(create_app(adapters))


@pytest.fixture(scope="module")
def bare_client():
    """No adapters loaded — metadata endpoints must still work."""
    return TestClient(create_app({}))


def predict(client, model="shorkie", **body):
    body.setdefault("sequences", [SEQ])
    return client.post(f"/v1/models/{model}/predict", json=body)


class TestMetadataEndpoints:
    def test_health_does_not_require_a_model(self, bare_client):
        body = bare_client.get("/health").json()
        assert body["ok"] is True
        assert body["loaded"] == []
        assert set(body["models"]) == {"shorkie", "yorzoi"}

    def test_list_models(self, client):
        models = {m["name"]: m for m in client.get("/v1/models").json()["models"]}
        assert models["shorkie"]["n_bins_full"] == 896
        assert models["shorkie"]["bin_size_bp"] == 16
        assert models["yorzoi"]["n_bins_full"] == 300
        assert models["yorzoi"]["bin_size_bp"] == 10

    def test_describe_model_publishes_worked_limits(self, client):
        body = client.get("/v1/models/shorkie").json()
        assert body["loaded"] is True
        assert body["limits"]["max_rows_json"] == MAX_ROWS_JSON
        assert len(body["limits"]["examples"]) >= 2

    def test_unknown_model_is_404(self, client):
        r = client.get("/v1/models/nope")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "unknown_model"

    def test_tracks_are_cacheable_and_complete(self, client):
        r = client.get("/v1/models/shorkie/tracks")
        assert r.headers["cache-control"].startswith("public")
        body = r.json()
        assert body["n_tracks"] == 5215
        assert len(body["tracks"]) == 5215

    def test_yorzoi_tracks_carry_strand_pairing(self, client):
        tracks = client.get("/v1/models/yorzoi/tracks").json()["tracks"]
        assert all(tracks[t["strand_pair"]]["library_id"] == t["library_id"] for t in tracks)


class TestPredict:
    def test_default_selection_is_small(self, client):
        body = predict(client).json()
        assert len(body["tracks"]) == 4
        assert all(t["kind"] == "profile" for t in body["tracks"])
        assert all(t["units"] == "relative_to_window_mean" for t in body["tracks"])

    def test_geometry_is_reported_per_sequence(self, client):
        p = predict(client).json()["predictions"][0]
        assert p["length"] == 1000
        assert p["window"] == "full"
        assert p["bin_size_bp"] == 16
        assert p["bin_start_bp"] == 0
        assert p["n_bins"] == 63
        assert p["last_bin_partial"] is True
        assert p["last_bin_bp"] == 8
        assert p["pad_fraction"] > 0.9
        assert "very_low_context" in p["warnings"]

    def test_exact_window_length_uses_the_native_path(self, client):
        p = predict(client, sequences=["A" * SHORKIE.seq_length]).json()["predictions"][0]
        assert p["window"] == "native"
        assert p["tiles"] == 1
        assert p["bin_start_bp"] == 1024
        assert p["n_bins"] == 896
        assert p["pad_fraction"] == 0.0

    def test_ids_are_preserved_and_duplicates_do_not_collapse(self, client):
        body = predict(
            client,
            sequences=[{"id": "wt", "seq": SEQ}, {"id": "alt", "seq": SEQ}],
        ).json()
        assert [p["id"] for p in body["predictions"]] == ["wt", "alt"]

    def test_values_are_track_major(self, client):
        body = predict(client, tracks=[0, 1, 2]).json()
        values = body["predictions"][0]["values"]
        assert len(values) == 3
        assert len(values[0]) == body["predictions"][0]["n_bins"]

    def test_stub_values_track_sequence_content(self, client):
        # The stub encodes G+C per bin, so an all-AT sequence and an all-GC sequence
        # must differ. This is what makes the API tests sensitive to coordinate bugs.
        at = predict(client, sequences=["AT" * 500], tracks=[0]).json()
        gc = predict(client, sequences=["GC" * 500], tracks=[0]).json()
        assert at["predictions"][0]["values"][0][0] == 0.0
        assert gc["predictions"][0]["values"][0][0] == 16.0

    def test_aggregate_returns_one_scalar_per_row(self, client):
        body = predict(client, mode="aggregate", preset="all").json()
        p = body["predictions"][0]
        assert len(body["tracks"]) == 5215
        assert len(p["values"]) == 5215
        assert isinstance(p["values"][0], float)
        assert p["n_bins"] is None

    def test_bench_returns_timings_without_a_payload(self, client):
        body = client.post(
            "/v1/models/shorkie/predict?bench=true", json={"sequences": [SEQ]}
        ).json()
        assert body["bench"] is True
        assert "predictions" not in body
        assert body["meta"]["timings_ms"]["infer"] >= 0

    def test_yorzoi_predicts_on_its_own_geometry(self, client):
        p = predict(client, model="yorzoi", sequences=[SEQ]).json()["predictions"][0]
        assert p["bin_size_bp"] == 10
        assert p["n_bins"] == 100

    def test_meta_declares_units_are_not_comparable(self, client):
        meta = predict(client).json()["meta"]
        assert meta["depth_normalized"] is False
        assert meta["value_space"] == "binned_coverage_sum"


class TestBinaryFormat:
    def test_npz_round_trips(self, client):
        r = client.post(
            "/v1/models/shorkie/predict?format=npz",
            json={"sequences": [SEQ], "preset": "all"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        z = np.load(io.BytesIO(r.content))
        assert z["predictions"].shape == (1, 5215, 63)
        assert z["sequence_ids"].tolist() == ["0"]
        assert json.loads(bytes(z["meta_json"]).decode())["model"] == "shorkie"

    def test_npz_is_not_gzipped(self, client):
        # gzip on float32 costs real CPU for ~1.1x; the middleware must skip it.
        r = client.post(
            "/v1/models/shorkie/predict?format=npz", json={"sequences": [SEQ]}
        )
        assert r.headers.get("content-encoding") == "identity"

    def test_npz_lifts_the_json_row_cap(self, client):
        # 5215 rows is far past MAX_ROWS_JSON but legitimate in binary.
        r = client.post(
            "/v1/models/shorkie/predict?format=npz",
            json={"sequences": [SEQ], "preset": "all"},
        )
        assert r.status_code == 200


class TestErrorContract:
    def test_invalid_alphabet_is_422_with_position(self, client):
        r = predict(client, sequences=["ACGT" * 100 + "R" + "ACGT" * 100])
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "invalid_alphabet"
        assert err["character"] == "R"
        assert err["position"] == 400

    def test_too_short_is_reported_before_alphabet(self, client):
        # Length is the cheaper and more fundamental constraint, so it wins.
        assert predict(client, sequences=["ACG"]).json()["error"]["code"] == "sequence_too_short"

    def test_too_many_sequences(self, client):
        r = predict(client, sequences=[SEQ] * 9)
        assert r.status_code == 422

    def test_unknown_track_identifier(self, client):
        r = predict(client, tracks=["not_a_track"])
        assert r.json()["error"]["code"] == "unknown_track"

    def test_out_of_range_track_index(self, client):
        assert predict(client, tracks=[99999]).json()["error"]["code"] == "unknown_track"

    def test_json_row_cap_is_enforced(self, client):
        r = predict(client, preset="all")
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "response_budget_exceeded"
        assert err["requested_rows"] == 5215

    def test_explicit_native_on_wrong_length_is_rejected(self, client):
        # Silently honouring this is shorkie's wrong-answer path: the network accepts
        # nearby lengths and returns a confidently wrong prediction.
        r = predict(client, window="native")
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "native_window_unavailable"

    def test_unaligned_step_is_rejected(self, client):
        r = predict(client, sequences=["ACGT" * 8000], step=1000)
        assert r.status_code == 422

    def test_model_not_loaded_is_503(self, bare_client):
        r = predict(bare_client)
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unavailable"

    def test_errors_carry_a_request_id(self, client):
        body = predict(client, sequences=["ACGT" * 100 + "R"]).json()
        assert body["error"]["request_id"]

    def test_mixed_track_reference_types_rejected(self, client):
        assert predict(client, tracks=[0, "AAP1_S0"]).status_code == 422

    def test_empty_sequence_list_rejected(self, client):
        assert predict(client, sequences=[]).status_code == 422

    def test_never_returns_500_for_client_mistakes(self, client):
        for bad in ["", "   ", "N" * 5, ">only a header", "ACGU", "ACGT" * 100_000]:
            r = predict(client, sequences=[bad])
            assert r.status_code in (200, 422), f"{bad[:20]!r} -> {r.status_code}"


class TestCors:
    def test_frontend_origin_is_allowed(self, client):
        r = client.options(
            "/v1/models/shorkie/predict",
            headers={
                "Origin": "https://yorzoi.eu",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.headers["access-control-allow-origin"] == "https://yorzoi.eu"

    def test_localhost_dev_origin_is_allowed(self, client):
        r = client.options(
            "/v1/models/shorkie/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
