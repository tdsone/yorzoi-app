"""The function that runs inside the GPU container.

Returns an envelope rather than raising: a typed ApiError must survive the Modal hop
as data, not as a pickled traceback. The gateway already validated the request, so
the re-validation here is defense in depth against payload drift between deploys.

Deliberately never imports `app.api` (fastapi is absent from the shorkie worker
image) — the two meta constants it would want are inlined.
"""

from __future__ import annotations

import time

from app.core.errors import ApiError
from app.core.serialize import sig_signed
from app.core.spec import get_spec
from app.core.tracks import load_catalog
from app.ism import ISM_VERSION
from app.ism.engine import plan_ism, run_ism

API_VERSION = 1


def execute(model: str, payload: dict, adapter) -> dict:
    """payload: {"request": <IsmRequest dump>, "request_id": str}."""
    request_id = payload.get("request_id")
    try:
        from app.schemas import IsmRequest

        req = IsmRequest(**payload["request"])
        spec = get_spec(model)
        plan = plan_ism(model, req, spec=spec, catalog=load_catalog(model))
        result = run_ism(plan, adapter)

        plots_started = time.perf_counter()
        logos = []
        if req.plots.include:
            from app.ism.logo import render_logo, to_base64

            kinds = {"mean": ["mean"], "none": ["none"], "both": ["mean", "none"]}
            for kind in kinds[req.plots.normalization]:
                if kind == "mean":
                    png = render_logo(
                        result.projected,
                        style="projected",
                        ref_bases=result.ref_bases,
                        start_bp=plan.ism_start,
                        width_px=req.plots.width_px,
                        dpi=req.plots.dpi,
                        title=f"{model} ISM — mean-centered attribution ({plan.id})",
                        ylabel="attribution",
                    )
                else:
                    png = render_logo(
                        result.raw - result.u0,
                        style="per_base",
                        start_bp=plan.ism_start,
                        width_px=req.plots.width_px,
                        dpi=req.plots.dpi,
                        title=f"{model} ISM — Δu vs reference ({plan.id})",
                        ylabel="Δu",
                    )
                logos.append(
                    {
                        "normalization": kind,
                        "png_base64": to_base64(png),
                        "width_px": req.plots.width_px,
                        "dpi": req.plots.dpi,
                    }
                )
        plots_ms = round((time.perf_counter() - plots_started) * 1e3, 1)

        meta = {
            "api_version": API_VERSION,
            "ism_version": ISM_VERSION,
            "request_id": request_id,
            **spec.to_dict(),
            **adapter.describe(),
            "u0": result.u0,
            "origin_bp": plan.origin,
            "predicted_span_bp": [
                max(0, plan.origin),
                min(len(plan.seq), plan.origin + spec.predicted_bp),
            ],
            "ism_window": [plan.ism_start, plan.ism_end],
            "n_mutants": plan.n_mutants,
            "n_forwards": plan.n_forwards,
            "batch_size": plan.batch_size,
            "metric": plan.metric.describe,
            "timings_ms": {**result.timings_ms, "plots": plots_ms},
        }
        return {
            "status": "succeeded",
            "result": {
                "meta": meta,
                "sequence": {"id": plan.id, "length": len(plan.seq)},
                "matrices": {
                    "alphabet": ["A", "C", "G", "T"],
                    "start_bp": plan.ism_start,
                    "raw": sig_signed(result.raw).tolist(),
                    "centered": sig_signed(result.centered).tolist(),
                    "projected": sig_signed(result.projected).tolist(),
                },
                "logos": logos,
            },
        }
    except ApiError as exc:
        return {"status": "failed", "error": exc.to_dict(request_id)["error"]}
    except Exception as exc:  # noqa: BLE001 — job failures must come back as data
        return {
            "status": "failed",
            "error": {
                "code": "ism_job_failed",
                "message": f"{type(exc).__name__}: {exc}"[:500],
                "request_id": request_id,
            },
        }
