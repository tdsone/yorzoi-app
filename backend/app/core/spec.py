"""Model geometry as data.

Yorzoi and Shorkie answer the same question — DNA in, per-position tracks out — and
differ only in constants. Describing that difference as a `ModelSpec` is what lets one
tiling/stitching/validation/serialization implementation serve both, instead of the two
divergent copies the previous design would have produced.

`bin_size` is the resolution of the values the adapter emits — the grid on which tiles
are stitched. Yorzoi's network predicts **300 bins of 10 bp**; the previous
implementation immediately expanded those to 3000 per-bp values, but that expansion is
a plain `repeat_interleave(10) / 10`, so it is 10x the bytes for zero extra
information and it destroys the alignment guarantees stitching depends on. Yorzoi is
therefore bin_size=10 over 300 bins, and per-bp expansion is a presentation option.

Note that `crop_bp` need not be a multiple of `bin_size`. Yorzoi crops 996 bp
(996 % 10 == 6) because 996 describes how its training *labels* were built, not a
property of the graph — the network's own positional unit is 8 bp and it drops 1296 bp
per edge. Only `predicted_bp % bin_size == 0` is a real invariant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    seq_length: int
    """Input window in bp. The ONLY length the network may ever be fed."""

    crop_bp: int
    """Discarded from each edge of the window; predictions do not cover it."""

    bin_size: int
    """bp per emitted bin, after any adapter-side unbinning. 1 means per-bp."""

    n_tracks: int
    stranded: bool
    value_space: str
    depth_normalized: bool
    """False means track magnitudes are NOT comparable to each other."""

    default_step: int
    """Tile stride in bp for sequences longer than `predicted_bp`."""

    max_seq_len: int
    description: str = ""

    def __post_init__(self) -> None:
        if self.predicted_bp <= 0:
            raise ValueError(f"{self.name}: crop_bp too large for seq_length")
        if self.predicted_bp % self.bin_size:
            raise ValueError(f"{self.name}: predicted_bp is not a whole number of bins")
        if self.default_step % self.bin_size:
            raise ValueError(f"{self.name}: default_step must be a multiple of bin_size")
        if not 0 < self.default_step <= self.predicted_bp:
            raise ValueError(f"{self.name}: default_step must be within (0, predicted_bp]")

    @property
    def predicted_bp(self) -> int:
        return self.seq_length - 2 * self.crop_bp

    @property
    def n_bins(self) -> int:
        return self.predicted_bp // self.bin_size

    @property
    def min_seq_len(self) -> int:
        return self.bin_size

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seq_length_bp": self.seq_length,
            "crop_bp": self.crop_bp,
            "predicted_bp": self.predicted_bp,
            "bin_size_bp": self.bin_size,
            "n_bins_full": self.n_bins,
            "n_tracks": self.n_tracks,
            "stranded": self.stranded,
            "value_space": self.value_space,
            "depth_normalized": self.depth_normalized,
            "max_seq_len_bp": self.max_seq_len,
            "min_seq_len_bp": self.min_seq_len,
            "description": self.description,
        }


SHORKIE = ModelSpec(
    name="shorkie",
    seq_length=16384,
    crop_bp=1024,          # Cropping1D(64) at 16 bp per bin
    bin_size=16,
    n_tracks=5215,
    stranded=False,        # the released targets sheet has no strand_pair column
    value_space="binned_coverage_sum",
    depth_normalized=False,
    default_step=7168,
    max_seq_len=131_072,
    description=(
        "Shorkie (Chao et al.), PyTorch port. 16 bp bins; outputs are per-track "
        "coverage sums in each track's native units and are not depth-normalized."
    ),
)

YORZOI = ModelSpec(
    name="yorzoi",
    seq_length=4992,
    crop_bp=996,           # label-space convention: (4992 - 3000) / 2, not a graph property
    bin_size=10,           # the network emits 300 bins; per-bp is an expansion, not data
    n_tracks=162,
    stranded=True,         # 81 plus-strand tracks then their 81 minus-strand partners
    value_space="coverage",
    depth_normalized=False,
    default_step=1000,
    max_seq_len=131_072,
    description=(
        "Yorzoi RNA-seq coverage prediction. 10 bp bins over 81 stranded library pairs."
    ),
)

REGISTRY: dict[str, ModelSpec] = {s.name: s for s in (SHORKIE, YORZOI)}


def get_spec(name: str) -> ModelSpec:
    from .errors import UnknownModel

    try:
        return REGISTRY[name]
    except KeyError:
        raise UnknownModel(
            f"unknown model {name!r}; available: {', '.join(sorted(REGISTRY))}",
            field="model",
        ) from None
