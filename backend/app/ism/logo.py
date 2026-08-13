"""Sequence-logo rendering (matplotlib Agg, hand-rolled glyphs).

TextPath glyphs instead of logomaker: the worker images stay pandas-free and the
two styles here are the whole requirement. matplotlib is imported inside the
functions so `import app.ism.logo` stays legal in images without it (gateway,
tests that skip plotting).

Styles:
- `projected` — one letter per position (the observed base), signed height from the
  mean-centered attribution; negatives hang below the axis.
- `per_base` — up to three letters per position from `raw - u0` (the reference
  base's delta is identically 0 and is omitted); positives stack up, negatives
  stack down, largest magnitude nearest the axis. This is the honest un-centered
  view: without centering the observed-base projection is identically zero, so a
  projected logo would be blank.
"""

from __future__ import annotations

import base64
import io

import numpy as np

ALPHABET = "ACGT"
COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}


def _glyphs():
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(family="DejaVu Sans", weight="bold")
    out = {}
    for char in ALPHABET:
        path = TextPath((0, 0), char, size=1.0, prop=prop)
        out[char] = (path, path.get_extents())
    return out


def _placed(path, bbox, x: float, y: float, height: float, affine_cls):
    """The glyph scaled into a 1 x height box whose bottom-left corner is (x, y)."""
    transform = (
        affine_cls()
        .translate(-bbox.x0, -bbox.y0)
        .scale(1.0 / bbox.width, height / bbox.height)
        .translate(x, y)
    )
    return path.transformed(transform)


def render_logo(
    values: np.ndarray,
    *,
    style: str,
    ref_bases: np.ndarray | None = None,
    start_bp: int = 0,
    width_px: int = 1600,
    dpi: int = 100,
    title: str = "",
    ylabel: str = "",
) -> bytes:
    """Render one logo to PNG bytes.

    `values` is (W,) for style="projected" (`ref_bases` required) or (4, W) deltas
    for style="per_base".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Affine2D

    glyphs = _glyphs()
    paths: list = []
    colors: list[str] = []
    width = int(values.shape[-1])

    if style == "projected":
        if ref_bases is None:
            raise ValueError("style='projected' requires ref_bases")
        v = np.asarray(values, dtype=np.float64)
        for p in range(width):
            if v[p] == 0.0:
                continue
            char = ALPHABET[int(ref_bases[p])]
            path, bbox = glyphs[char]
            paths.append(_placed(path, bbox, start_bp + p, min(0.0, v[p]), abs(v[p]), Affine2D))
            colors.append(COLORS[char])
        y_lo, y_hi = float(min(v.min(), 0.0)), float(max(v.max(), 0.0))
    elif style == "per_base":
        m = np.asarray(values, dtype=np.float64)
        for p in range(width):
            column = m[:, p]
            up = down = 0.0
            for a in np.argsort(-np.abs(column)):        # largest nearest the axis
                delta = column[a]
                if delta == 0.0:
                    continue
                char = ALPHABET[int(a)]
                path, bbox = glyphs[char]
                if delta > 0:
                    paths.append(_placed(path, bbox, start_bp + p, up, delta, Affine2D))
                    up += delta
                else:
                    down += delta
                    paths.append(_placed(path, bbox, start_bp + p, down, -delta, Affine2D))
                colors.append(COLORS[char])
        y_lo = float(min(np.clip(m, None, 0.0).sum(axis=0).min(), 0.0))
        y_hi = float(max(np.clip(m, 0.0, None).sum(axis=0).max(), 0.0))
    else:
        raise ValueError(f"unknown logo style {style!r}")

    height_px = min(max(width_px // 5, 240), 500)
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    try:
        if paths:
            ax.add_collection(PathCollection(paths, facecolors=colors, edgecolors="none"))
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_xlim(start_bp, start_bp + width)
        pad = 0.05 * max(y_hi - y_lo, 1e-9)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_xlabel("position (bp)")
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        return buf.getvalue()
    finally:
        plt.close(fig)


def to_base64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")
