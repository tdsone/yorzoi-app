"""Logo rendering: valid PNGs out of both styles, argument contracts enforced."""

from __future__ import annotations

import base64

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from app.ism.logo import render_logo, to_base64

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_projected_style_renders_png():
    values = np.array([0.5, -0.3, 0.0, 1.2, -0.9])
    png = render_logo(
        values,
        style="projected",
        ref_bases=np.array([0, 1, 2, 3, 0]),
        start_bp=100,
        width_px=400,
        dpi=72,
        title="test",
        ylabel="attribution",
    )
    assert png[:8] == PNG_MAGIC
    assert len(png) > 1000


def test_per_base_style_renders_png():
    deltas = np.array(
        [
            [0.0, 0.4, -0.2],
            [0.3, 0.0, -0.1],
            [-0.2, 0.1, 0.0],
            [0.1, -0.3, 0.2],
        ]
    )
    png = render_logo(deltas, style="per_base", width_px=400, dpi=72)
    assert png[:8] == PNG_MAGIC


def test_all_zero_values_still_render():
    png = render_logo(
        np.zeros(10),
        style="projected",
        ref_bases=np.zeros(10, dtype=int),
        width_px=300,
        dpi=72,
    )
    assert png[:8] == PNG_MAGIC


def test_projected_requires_ref_bases():
    with pytest.raises(ValueError, match="ref_bases"):
        render_logo(np.ones(4), style="projected")


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="style"):
        render_logo(np.ones(4), style="nope", ref_bases=np.zeros(4, dtype=int))


def test_base64_round_trip():
    png = render_logo(
        np.array([1.0, -1.0]),
        style="projected",
        ref_bases=np.array([0, 3]),
        width_px=300,
        dpi=72,
    )
    assert base64.b64decode(to_base64(png)) == png
