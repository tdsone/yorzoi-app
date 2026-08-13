"""Cross-check the reimplemented ISM math against tangermeme itself.

tangermeme is deliberately NOT a dependency (it drags pandas/numba/pyBigWig into
environments that don't need them), so this suite runs only where someone has it
installed — e.g. `uv run --with tangermeme pytest tests/test_ism_tangermeme.py`.
Verified passing against tangermeme at implementation time; the numbers below match
to float32 precision.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tangermeme")
torch = pytest.importorskip("torch")

from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from app.core.spec import YORZOI
from app.core.tracks import load_catalog
from app.ism.engine import plan_ism, run_ism
from app.models.stub import StubAdapter
from app.schemas import IsmRequest

SPEC = YORZOI
INTERVAL = (996, 1196)
WINDOW = (996, 1016)


class UModel(torch.nn.Module):
    """The stub-adapter + interval_mean composition as one torch model:
    overlap-weighted mean over the interval of per-bin G+C counts (track 0)."""

    def __init__(self, origin: int, interval: tuple[int, int]) -> None:
        super().__init__()
        lo = (interval[0] - origin) // SPEC.bin_size
        hi = -(-(interval[1] - origin) // SPEC.bin_size)
        starts = origin + np.arange(lo, hi) * SPEC.bin_size
        w = np.minimum(starts + SPEC.bin_size, interval[1]) - np.maximum(starts, interval[0])
        per_bp = np.zeros(SPEC.seq_length)
        for j, wj in zip(range(lo, hi), w):
            per_bp[SPEC.crop_bp + j * SPEC.bin_size : SPEC.crop_bp + (j + 1) * SPEC.bin_size] = wj
        self.register_buffer("per_bp", torch.tensor(per_bp / w.sum(), dtype=torch.float32))

    def forward(self, X):
        gc = X[:, 1, :] + X[:, 2, :]
        return (gc * self.per_bp).sum(dim=1, keepdim=True)


@pytest.fixture(scope="module")
def ours_and_reference():
    req = IsmRequest(
        sequence="ACGT" * (SPEC.seq_length // 4),
        ism_window={"start": WINDOW[0], "end": WINDOW[1]},
        metric={"interval": {"start": INTERVAL[0], "end": INTERVAL[1]}, "tracks": [0]},
        plots={"include": False},
    )
    adapter = StubAdapter(SPEC)
    adapter.load()
    plan = plan_ism("yorzoi", req, spec=SPEC, catalog=load_catalog("yorzoi"))
    ours = run_ism(plan, adapter)

    X = torch.tensor(plan.window[None])
    model = UModel(plan.origin, INTERVAL)
    col0 = plan.ism_start - (plan.origin - SPEC.crop_bp)
    return ours, model, X, col0


def test_u0_matches_the_torch_mirror(ours_and_reference):
    ours, model, X, _ = ours_and_reference
    assert model(X)[0, 0].item() == pytest.approx(ours.u0, abs=1e-5)


def test_centered_equals_tangermeme_hypothetical(ours_and_reference):
    ours, model, X, col0 = ours_and_reference
    width = WINDOW[1] - WINDOW[0]
    attr = saturation_mutagenesis(
        model, X, start=col0, end=col0 + width, hypothetical=True, device="cpu"
    )
    assert np.allclose(attr[0].numpy(), ours.centered, atol=1e-5)


def test_projected_equals_tangermeme_channel_sum(ours_and_reference):
    ours, model, X, col0 = ours_and_reference
    width = WINDOW[1] - WINDOW[0]
    attr = saturation_mutagenesis(model, X, start=col0, end=col0 + width, device="cpu")
    assert np.allclose(attr[0].sum(dim=0).numpy(), ours.projected, atol=1e-5)
