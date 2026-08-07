"""Tiling and stitching on a shared global bin grid.

One concept carries the whole module: **origin** is the original-sequence bp coordinate
of the first *predicted* bp of a tile. That single number replaces the previous
implementation's left_pad / prediction_offset / start_pos_in_original triple and makes
the short, tiled and native paths the same code.

    window covers original bp   [origin - crop_bp, origin - crop_bp + seq_length)
    tile bin j                  [origin + bin_size*j, + bin_size)

What matters is that origins within one merge are congruent to **each other** modulo
`bin_size` — not that they are multiples of it. `crop_bp` is not always a whole number
of bins (yorzoi crops 996 bp at 10 bp resolution), so a native pass legitimately starts
at an origin that is not a multiple of `bin_size`. `merge` therefore positions tiles in
bp relative to the earliest origin and reports `start_bp`.

Do not "simplify" this back to an absolute `origin // bin_size` bin index. That
truncates, and the result is a silent sub-bin phase shift that averages misaligned
grids together — values no single window ever predicted.
"""

from __future__ import annotations

import numpy as np

from .errors import BadRequest, NativeWindowUnavailable, TilingGap
from .spec import ModelSpec

WINDOW_MODES = ("auto", "full", "native")


def _floor_to(value: int, unit: int) -> int:
    return (value // unit) * unit


def _ceil_to(value: int, unit: int) -> int:
    return -((-value) // unit) * unit


def n_bins_for(length: int, spec: ModelSpec) -> int:
    """Bins needed to cover a sequence of `length` bp on the global grid."""
    return -((-length) // spec.bin_size)


def plan_origins(
    length: int,
    spec: ModelSpec,
    window: str = "auto",
    step: int | None = None,
    max_tiles: int | None = None,
) -> tuple[list[int], str]:
    """Return (origins, resolved_window_mode).

    `native` is a single exact pass with no padding, covering the model's natural
    predicted span. `full` covers the whole input, padding and tiling as needed.
    `auto` picks native for an exactly-window-length sequence, else full — because
    tiling an exactly-window-length sequence produces two tiles that are each padded
    on one side, and averages two N-contaminated predictions where one clean pass
    exists.
    """
    if window not in WINDOW_MODES:
        raise BadRequest(f"window must be one of {', '.join(WINDOW_MODES)}", field="window")
    if length <= 0:
        raise BadRequest("sequence is empty", field="sequences")

    if window == "auto":
        window = "native" if length == spec.seq_length else "full"

    if window == "native":
        if length != spec.seq_length:
            raise NativeWindowUnavailable(
                f"window='native' requires exactly {spec.seq_length} bp; got {length} bp",
                field="window",
                required_length=spec.seq_length,
                length=length,
            )
        return [spec.crop_bp], "native"

    step = spec.default_step if step is None else step
    if step <= 0 or step % spec.bin_size or step > spec.predicted_bp:
        raise BadRequest(
            f"step must be a multiple of {spec.bin_size} in (0, {spec.predicted_bp}]",
            field="step",
        )

    if length <= spec.predicted_bp:
        # Centre the sequence in the predicted span; align so the first bin is whole.
        offset = _floor_to((spec.predicted_bp - length) // 2, spec.bin_size)
        return [-offset], "full"

    if max_tiles is not None:
        # Count arithmetically before materialising, so a pathological step cannot
        # allocate a huge list on the way to being rejected.
        projected = 1 + -(-(length - spec.predicted_bp) // step) + 1
        if projected > max_tiles:
            raise BadRequest(
                f"sequence needs ~{projected} tiles; the limit is {max_tiles}",
                field="sequences",
                tiles=projected,
                max_tiles=max_tiles,
            )

    origins = list(range(0, length - spec.predicted_bp + 1, step))
    # Round the tail UP to a bin boundary: rounding down would leave a sub-bin gap at
    # the 3' end that the coverage check would then reject.
    tail = _ceil_to(length - spec.predicted_bp, spec.bin_size)
    if origins[-1] != tail:
        origins.append(tail)
    return origins, "full"


def tile_onehot(onehot: np.ndarray, origin: int, spec: ModelSpec) -> np.ndarray:
    """Extract the (4, seq_length) model input for one tile.

    Positions outside the sequence stay all-zero, which is the encoding of N.
    """
    length = onehot.shape[1]
    win = np.zeros((4, spec.seq_length), dtype=np.float32)
    lo = origin - spec.crop_bp
    hi = lo + spec.seq_length
    a, b = max(0, lo), min(length, hi)
    if a < b:
        win[:, a - lo : b - lo] = onehot[:, a:b]
    return win


def pad_fraction(length: int, origins: list[int], spec: ModelSpec) -> float:
    """Share of the model input across all tiles that is padding rather than sequence."""
    real = 0
    for origin in origins:
        lo = origin - spec.crop_bp
        real += max(0, min(length, lo + spec.seq_length) - max(0, lo))
    return 1.0 - real / (len(origins) * spec.seq_length)


def merge(
    tile_preds: list[np.ndarray],
    origins: list[int],
    spec: ModelSpec,
    clip_bp: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int]:
    """Average overlapping tiles onto a shared bin grid.

    `tile_preds` are (n_bins, n_tracks) in tile order. Returns (values, start_bp) where
    start_bp is the original-sequence coordinate of row 0's first base.

    Positions are tracked in bp relative to the earliest origin rather than as absolute
    bin indices, because `crop_bp` is not always a whole number of bins — yorzoi crops
    996 bp at 10 bp resolution — so `origin // bin_size` would silently truncate and
    shift a native pass by a fraction of a bin.

    The output span comes from the origins, never from the sequence length: a native
    pass covers [crop_bp, seq_length - crop_bp), so deriving the span from the length
    would leave bins unwritten and trip the coverage check.
    """
    if not tile_preds:
        raise TilingGap("no tiles were produced")
    if len(tile_preds) != len(origins):
        raise TilingGap(f"{len(tile_preds)} tiles for {len(origins)} origins")

    ref = min(origins)
    offsets = []
    for origin in origins:
        delta = origin - ref
        if delta % spec.bin_size:
            raise TilingGap(
                f"tile origins {ref} and {origin} are {delta} bp apart, which is not a "
                f"whole number of {spec.bin_size} bp bins"
            )
        offsets.append(delta // spec.bin_size)

    span = max(offsets) + spec.n_bins
    n_tracks = tile_preds[0].shape[1]
    acc = np.zeros((span, n_tracks), dtype=np.float32)
    cnt = np.zeros((span, 1), dtype=np.float32)
    for pred, offset in zip(tile_preds, offsets):
        acc[offset : offset + spec.n_bins] += pred
        cnt[offset : offset + spec.n_bins] += 1.0

    start_bp = ref
    if clip_bp is not None:
        lo, hi = clip_bp
        if (lo - ref) % spec.bin_size or (hi - ref) % spec.bin_size:
            raise TilingGap(
                f"clip range [{lo}, {hi}) is not aligned to the {ref} bp tile grid"
            )
        i0 = (lo - ref) // spec.bin_size
        i1 = (hi - ref) // spec.bin_size
        if i0 < 0 or i1 > span:
            raise TilingGap(
                f"requested bp [{lo}, {hi}) are not covered by tiles spanning "
                f"[{ref}, {ref + span * spec.bin_size})"
            )
        acc, cnt = acc[i0:i1], cnt[i0:i1]
        start_bp = lo

    if not (cnt > 0).all():
        # Never clamp the counter to 1 to "fix" this. That is what the previous
        # implementation did, and it returns a block of zeros as if they were
        # predictions.
        raise TilingGap(f"{int((cnt == 0).sum())} bins were left uncovered by tiling")

    return acc / cnt, start_bp


def plan_clip(length: int, mode: str, spec: ModelSpec) -> tuple[int, int] | None:
    """bp range the response should report, or None to report the natural tile span."""
    if mode == "native":
        return None
    return (0, n_bins_for(length, spec) * spec.bin_size)
