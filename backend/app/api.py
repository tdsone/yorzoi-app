"""FastAPI application factory.

Deliberately free of Modal imports: the app is constructed and exercised in tests with
no cloud dependency, and `modal_app.py` is a thin deployment shim around it.
"""

from __future__ import annotations

import logging
import os
import time
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app import engine as EN
from app.core.errors import ApiError, ModelUnavailable
from app.core.serialize import build_npz, to_json_values
from app.core.spec import REGISTRY, get_spec
from app.core.tracks import DEFAULT_PRESET, load_catalog
from app.schemas import PredictRequest

API_VERSION = 1
ENGINE_VERSION = "1.0.0"

# Above this a JSON matrix stops being a sensible wire format; npz or aggregate mode
# carry larger selections.
MAX_ROWS_JSON = 64

DEFAULT_ORIGINS = [
    "https://yorzoi.eu",
    "https://www.yorzoi.eu",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

log = logging.getLogger("yorzoi.api")


def allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS")
    return [o.strip() for o in raw.split(",") if o.strip()] if raw else list(DEFAULT_ORIGINS)


def create_app(adapters: dict[str, object] | None = None) -> FastAPI:
    """`adapters` maps model name -> a loaded ModelAdapter. Models absent from the map
    are advertised by /v1/models but return 503 from /predict."""
    adapters = adapters or {}
    # No custom response class: current FastAPI serializes to JSON bytes directly and
    # deprecates ORJSONResponse.
    app = FastAPI(title="Yorzoi prediction API", version=str(API_VERSION))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Binary responses opt out by setting Content-Encoding themselves; Starlette skips
    # any response that already carries the header.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(status_code=exc.status, content=exc.to_dict(request_id))

    @app.get("/health")
    def health():
        """Liveness only. Deliberately does not touch a model, so an uptime monitor
        pointed here cannot keep an inference container alive."""
        return {
            "ok": True,
            "api_version": API_VERSION,
            "engine_version": ENGINE_VERSION,
            "models": sorted(REGISTRY),
            "loaded": sorted(n for n, a in adapters.items() if getattr(a, "loaded", False)),
        }

    @app.get("/v1/models")
    def list_models():
        return {"models": [spec.to_dict() for spec in REGISTRY.values()]}

    @app.get("/v1/models/{model}")
    def describe_model(model: str):
        spec = get_spec(model)
        limits = EN.Limits.for_spec(spec)
        adapter = adapters.get(model)
        return {
            **spec.to_dict(),
            "loaded": bool(adapter and getattr(adapter, "loaded", False)),
            "provenance": adapter.describe() if adapter else None,
            "default_preset": DEFAULT_PRESET.get(model),
            "limits": {
                "max_sequences": limits.max_sequences,
                "max_total_bp": limits.max_total_bp,
                "max_tiles": limits.max_tiles,
                "max_output_elements": limits.max_output_elements,
                "max_rows_json": MAX_ROWS_JSON,
                "examples": [
                    f"one {spec.seq_length} bp sequence, all {spec.n_tracks} tracks: "
                    f"{spec.n_bins * spec.n_tracks:,} values (npz only)",
                    f"eight {spec.seq_length} bp sequences, 64 tracks: "
                    f"{8 * spec.n_bins * 64:,} values",
                ],
            },
        }

    @app.get("/v1/models/{model}/tracks")
    def list_tracks(model: str, response: Response):
        get_spec(model)
        response.headers["Cache-Control"] = "public, max-age=86400, immutable"
        return load_catalog(model).to_dict()

    @app.post("/v1/models/{model}/predict")
    def predict(
        model: str,
        req: PredictRequest,
        request: Request,
        format: str = Query("json", pattern="^(json|npz)$"),
        bench: bool = False,
    ):
        request_id = str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()

        spec = get_spec(model)
        adapter = adapters.get(model)
        if adapter is None or not getattr(adapter, "loaded", False):
            raise ModelUnavailable(model)

        catalog = load_catalog(model)
        preset = req.preset
        if not (req.tracks or req.track_groups or preset):
            preset = DEFAULT_PRESET.get(model)

        selection = catalog.resolve(
            spec,
            tracks=req.tracks,
            track_groups=req.track_groups,
            preset=preset,
            max_rows=None if format == "npz" or req.mode == "aggregate" else MAX_ROWS_JSON,
        )

        aggregate = req.mode == "aggregate"
        limits = EN.Limits.for_spec(spec)
        plans = EN.plan(
            [EN.SequenceInput(i, s) for i, s in req.items()],
            spec,
            selection,
            window=req.window,
            step=req.step,
            aggregate=aggregate,
            limits=limits,
        )
        planned = time.perf_counter()
        result = EN.run(plans, adapter, selection, aggregate=aggregate, limits=limits)
        ran = time.perf_counter()

        meta = {
            "api_version": API_VERSION,
            "engine_version": ENGINE_VERSION,
            "request_id": request_id,
            **spec.to_dict(),
            **adapter.describe(),
            "mode": req.mode,
            "layout": "track_major",
            "n_tiles": result.n_tiles,
            "timings_ms": {
                "plan": round((planned - started) * 1e3, 2),
                "infer": round((ran - planned) * 1e3, 2),
            },
        }

        log.info(
            "predict model=%s id=%s sequences=%d bp=%d tiles=%d rows=%d mode=%s format=%s infer_ms=%.1f",
            model, request_id, len(plans), sum(len(p["seq"]) for p in plans),
            result.n_tiles, selection.n_rows, req.mode, format,
            meta["timings_ms"]["infer"],
        )

        if bench:
            meta["timings_ms"]["total"] = round((time.perf_counter() - started) * 1e3, 2)
            return {"meta": meta, "bench": True}

        tracks = [row.to_dict() for row in selection.rows]

        if format == "npz":
            payload = build_npz(
                [
                    {
                        "id": p.id,
                        "bin_start_bp": p.start_bp,
                        "values": p.values if not p.aggregated else p.values[:, None],
                    }
                    for p in result.predictions
                ],
                track_indices=[row.get("i", -1) for row in tracks],
                track_ids=[row["id"] for row in tracks],
                meta={**meta, "tracks": tracks},
            )
            return Response(
                content=payload,
                media_type="application/octet-stream",
                headers={
                    # gzip on float32 buys ~1.1x for real CPU; skip the middleware.
                    "Content-Encoding": "identity",
                    "Content-Length": str(len(payload)),
                    "Content-Disposition": f'attachment; filename="{model}_predictions.npz"',
                },
            )

        return {
            "meta": meta,
            "tracks": tracks,
            "predictions": [
                {
                    "id": p.id,
                    "length": p.length,
                    "window": p.window,
                    "tiles": p.tiles,
                    "bin_size_bp": p.bin_size_bp,
                    "bin_start_bp": p.start_bp,
                    "n_bins": None if p.aggregated else p.n_bins,
                    "last_bin_partial": None if p.aggregated else p.last_bin_partial,
                    "last_bin_bp": None if p.aggregated else p.last_bin_bp,
                    "pad_fraction": p.pad_fraction,
                    "n_fraction": p.n_fraction,
                    "warnings": p.warnings,
                    "values": to_json_values(p.values),
                }
                for p in result.predictions
            ],
        }

    return app

