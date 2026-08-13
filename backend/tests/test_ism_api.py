"""ISM API routes against the in-process launcher and stub adapters."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.core.spec import SHORKIE, YORZOI
from app.ism.jobs import InProcessLauncher
from app.models.stub import StubAdapter

SEQ = "ACGT" * 250          # 1000 bp

BODY = {
    "sequence": SEQ,
    "ism_window": {"start": 100, "end": 110},
    "metric": {"interval": {"start": 100, "end": 300}},
    "plots": {"include": False},
}


def adapters():
    out = {}
    for spec in (SHORKIE, YORZOI):
        adapter = StubAdapter(spec)
        adapter.load()
        out[spec.name] = adapter
    return out


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(adapters(), ism_launcher=InProcessLauncher(adapters())))


@pytest.fixture(scope="module")
def polling_client():
    """Launcher that reports running exactly once before revealing the result."""
    return TestClient(
        create_app(adapters(), ism_launcher=InProcessLauncher(adapters(), defer=True))
    )


def submit(client, model="yorzoi", **overrides):
    body = {**BODY, **overrides}
    return client.post(f"/v1/models/{model}/ism", json=body)


def run_job(client, model="yorzoi", **overrides):
    r = submit(client, model, **overrides)
    assert r.status_code == 202, r.text
    return client.get(r.json()["poll"]).json()


class TestSubmit:
    def test_returns_202_with_job_accounting(self, client):
        r = submit(client)
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "submitted"
        assert body["model"] == "yorzoi"
        assert body["n_mutants"] == 30
        assert body["n_forwards"] == 31
        assert body["poll"] == f"/v1/ism/jobs/{body['job_id']}"

    def test_strand_average_doubles_forwards(self, client):
        r = submit(
            client, metric={"interval": {"start": 100, "end": 300}, "strand_average": True}
        )
        assert r.json()["n_forwards"] == 62

    def test_routes_absent_without_a_launcher(self):
        bare = TestClient(create_app(adapters()))
        assert bare.post("/v1/models/yorzoi/ism", json=BODY).status_code == 404
        assert bare.get("/v1/ism/jobs/x").status_code == 404

    def test_health_and_describe_advertise_ism(self, client):
        assert client.get("/health").json()["ism"] == ["shorkie", "yorzoi"]
        assert client.get("/v1/models/yorzoi").json()["ism_supported"] is True
        assert client.get("/v1/models/borzoi").json()["ism_supported"] is False


class TestPolling:
    def test_running_then_succeeded(self, polling_client):
        r = submit(polling_client)
        poll = r.json()["poll"]
        assert polling_client.get(poll).json() == {
            "job_id": r.json()["job_id"],
            "status": "running",
        }
        assert polling_client.get(poll).json()["status"] == "succeeded"

    def test_unknown_job_is_404(self, client):
        r = client.get("/v1/ism/jobs/no-such-job")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "job_not_found"


class TestResultPayload:
    def test_matrices_and_meta(self, client):
        body = run_job(client)
        assert body["status"] == "succeeded"
        result = body["result"]

        matrices = result["matrices"]
        assert matrices["alphabet"] == ["A", "C", "G", "T"]
        assert matrices["start_bp"] == 100
        assert len(matrices["raw"]) == 4 and len(matrices["raw"][0]) == 10
        assert len(matrices["centered"]) == 4
        assert len(matrices["projected"]) == 10

        meta = result["meta"]
        assert meta["ism_version"] == "1.0.0"
        assert meta["ism_window"] == [100, 110]
        assert meta["n_mutants"] == 30
        assert meta["metric"]["name"] == "interval_mean"
        assert meta["metric"]["interval"] == [100, 300]
        assert meta["u0"] > 0
        assert result["sequence"] == {"id": "0", "length": 1000}

    def test_matrix_identities_survive_serialization(self, client):
        result = run_job(client)["result"]
        raw = np.array(result["matrices"]["raw"])
        centered = np.array(result["matrices"]["centered"])
        # 3-significant-digit rounding: identities hold loosely, not exactly.
        assert np.allclose(centered.sum(axis=0), 0.0, atol=0.02 * np.abs(raw).max())

    def test_sequence_id_is_preserved(self, client):
        body = run_job(client, sequence={"id": "my-promoter", "seq": SEQ})
        assert body["result"]["sequence"]["id"] == "my-promoter"

    def test_logos_render_when_requested(self, client):
        body = run_job(
            client, plots={"include": True, "normalization": "both", "width_px": 600}
        )
        logos = body["result"]["logos"]
        assert [l["normalization"] for l in logos] == ["mean", "none"]
        for logo in logos:
            png = base64.b64decode(logo["png_base64"])
            assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_shorkie_end_to_end(self, client):
        body = run_job(
            client,
            model="shorkie",
            metric={"interval": {"start": 100, "end": 300}, "tracks": [0]},
        )
        assert body["status"] == "succeeded"
        assert body["result"]["meta"]["name"] == "shorkie"


class TestNpzFormat:
    def test_round_trips(self, client):
        r = submit(client)
        job = r.json()["job_id"]
        r = client.get(f"/v1/ism/jobs/{job}?format=npz")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert r.headers.get("content-encoding") == "identity"
        z = np.load(io.BytesIO(r.content))
        assert z["raw"].shape == (4, 10)
        assert z["centered"].shape == (4, 10)
        assert z["projected"].shape == (10,)
        meta = json.loads(bytes(z["meta_json"]).decode())
        assert meta["ism_window"] == [100, 110]

    def test_json_poll_still_works_after_npz(self, client):
        r = submit(client)
        job = r.json()["job_id"]
        client.get(f"/v1/ism/jobs/{job}?format=npz")
        assert client.get(f"/v1/ism/jobs/{job}").json()["status"] == "succeeded"


class TestErrorContract:
    def test_unsupported_model_is_422(self, client):
        for model in ("borzoi", "flashzoi"):
            r = submit(client, model)
            assert r.status_code == 422
            assert r.json()["error"]["code"] == "ism_unsupported_model"

    def test_unknown_model_is_404(self, client):
        assert submit(client, "nope").status_code == 404

    def test_model_missing_from_launcher_is_503(self):
        launcher = InProcessLauncher(adapters(), models=("shorkie",))
        c = TestClient(create_app(adapters(), ism_launcher=launcher))
        r = submit(c, "yorzoi")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unavailable"

    def test_sequence_too_long_for_one_window(self, client):
        r = submit(client, sequence="A" * (YORZOI.seq_length + 1))
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "ism_sequence_too_long"

    def test_window_past_sequence_end(self, client):
        r = submit(client, ism_window={"start": 0, "end": 2000})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "ism_window_invalid"

    def test_inverted_window_is_a_validation_error(self, client):
        assert submit(client, ism_window={"start": 50, "end": 50}).status_code == 422

    def test_n_inside_the_window(self, client):
        seq = SEQ[:105] + "N" + SEQ[106:]
        r = submit(client, sequence=seq)
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "ism_window_contains_n"
        assert err["position"] == 105

    def test_n_outside_the_window_is_fine(self, client):
        seq = SEQ[:500] + "N" + SEQ[501:]
        assert submit(client, sequence=seq).status_code == 202

    def test_interval_outside_predicted_span(self, client):
        # 1000 bp sequence: the predicted span clips to the sequence, so an
        # interval reaching past 1000 bp must be rejected.
        r = submit(client, metric={"interval": {"start": 900, "end": 1100}})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "interval_outside_predicted_span"

    def test_unknown_track_in_metric(self, client):
        r = submit(client, metric={"interval": {"start": 100, "end": 300}, "tracks": [999]})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "unknown_track"

    def test_errors_carry_the_request_id(self, client):
        r = submit(client, sequence="A" * (YORZOI.seq_length + 1))
        assert r.json()["error"]["request_id"]

    def test_worker_failure_comes_back_as_a_failed_job(self):
        # The gateway validates, so simulate version skew by corrupting the payload
        # at the launcher boundary.
        class SkewedLauncher(InProcessLauncher):
            def submit(self, model, payload):
                payload["request"]["metric"]["name"] = "not_a_metric"
                return super().submit(model, payload)

        c = TestClient(create_app(adapters(), ism_launcher=SkewedLauncher(adapters())))
        body = run_job(c)
        assert body["status"] == "failed"
        assert body["error"]["code"] in {"unknown_metric", "ism_job_failed"}
