"""Track catalog, selection, and presets.

A selection is not just a list of channel indices. Presets return *derived rows* —
an averaged profile over an assay group — so the resolved object carries both the raw
indices the model must produce and the rows the response should contain.

Why profiles are mean-normalized: these tracks are not depth-normalized, so every one
carries its own arbitrary sequencing-depth factor. A plain mean over 3053 such tracks
is dominated by whichever tracks happen to have the largest factor. Dividing each track
by its own mean over the returned window first makes the profile a shape, which is what
a plot wants, and the response labels the units accordingly so nobody mistakes it for
coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .errors import BadRequest, ResponseBudgetExceeded, UnknownTrack
from .spec import ModelSpec

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

RAW = "raw"
PROFILE = "profile"

# preset -> ordered list of (row_label, group_name). "all" is handled separately.
PRESETS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "shorkie": {
        "assay_profiles": [
            ("ChIP-exo", "chip_exo"),
            ("ChIP-MNase", "chip_mnase"),
            ("RNA-seq (IDEA)", "rnaseq_idea"),
            ("RNA-seq (strains)", "rnaseq_strains"),
        ],
        "rnaseq_profile": [("RNA-seq (IDEA)", "rnaseq_idea")],
    },
    "yorzoi": {
        "strand_profiles": [("Plus strand", "plus"), ("Minus strand", "minus")],
    },
}

DEFAULT_PRESET = {"shorkie": "assay_profiles", "yorzoi": "strand_profiles"}


@dataclass(frozen=True)
class Row:
    """One row of the response's `values`."""

    kind: str                      # RAW or PROFILE
    label: str
    members: tuple[int, ...]       # positions into TrackSelection.indices
    index: int | None = None       # channel index, for RAW rows
    assay: str | None = None
    description: str | None = None

    def to_dict(self) -> dict:
        out: dict = {"kind": self.kind, "id": self.label}
        if self.index is not None:
            out["i"] = self.index
        if self.assay:
            out["assay"] = self.assay
        if self.description:
            out["description"] = self.description
        if self.kind == PROFILE:
            out["n_tracks"] = len(self.members)
            out["units"] = "relative_to_window_mean"
        return out


@dataclass(frozen=True)
class TrackSelection:
    indices: np.ndarray            # channels the model must produce, ascending
    rows: tuple[Row, ...]
    normalize_profiles: bool = True

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def reduce(self, values: np.ndarray) -> np.ndarray:
        """(n_selected_channels, n_bins) -> (n_rows, n_bins)."""
        out = np.empty((len(self.rows), values.shape[1]), dtype=np.float32)
        for r, row in enumerate(self.rows):
            block = values[list(row.members)]
            if row.kind == RAW:
                out[r] = block[0]
            elif self.normalize_profiles:
                scale = block.mean(axis=1, keepdims=True)
                np.maximum(scale, 1e-9, out=scale)
                out[r] = (block / scale).mean(axis=0)
            else:
                out[r] = block.mean(axis=0)
        return out


@dataclass
class TrackCatalog:
    model: str
    n_tracks: int
    groups: dict[str, list[int]]
    tracks: list[dict]
    _by_id: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {t["id"]: t["i"] for t in self.tracks}

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "n_tracks": self.n_tracks,
            "groups": self.groups,
            "presets": sorted(PRESETS.get(self.model, {})) + ["all"],
            "tracks": self.tracks,
        }

    def _resolve_one(self, item: int | str) -> int:
        if isinstance(item, bool):
            raise UnknownTrack(f"invalid track reference {item!r}", field="tracks")
        if isinstance(item, int):
            if not 0 <= item < self.n_tracks:
                raise UnknownTrack(
                    f"track index {item} is out of range 0..{self.n_tracks - 1}",
                    field="tracks",
                    index=item,
                )
            return item
        try:
            return self._by_id[item]
        except KeyError:
            raise UnknownTrack(
                f"unknown track identifier {item!r}", field="tracks", identifier=item
            ) from None

    def resolve(
        self,
        spec: ModelSpec,
        *,
        tracks: list[int | str] | None = None,
        track_groups: list[str] | None = None,
        preset: str | None = None,
        max_rows: int | None = None,
        normalize_profiles: bool = True,
    ) -> TrackSelection:
        explicit: list[int] = [self._resolve_one(t) for t in (tracks or [])]

        for group in track_groups or []:
            if group not in self.groups:
                raise UnknownTrack(
                    f"unknown track group {group!r}; available: {', '.join(sorted(self.groups))}",
                    field="track_groups",
                    group=group,
                )
            lo, hi = self.groups[group]
            explicit.extend(range(lo, hi))

        presets = PRESETS.get(self.model, {})
        profile_specs: list[tuple[str, str]] = []
        if preset == "all":
            explicit.extend(range(self.n_tracks))
        elif preset is not None:
            if preset not in presets:
                raise BadRequest(
                    f"unknown preset {preset!r}; available: "
                    f"{', '.join(sorted(presets) + ['all'])}",
                    field="preset",
                )
            profile_specs = presets[preset]

        if not explicit and not profile_specs:
            raise BadRequest(
                "no tracks selected; pass tracks, track_groups or preset", field="tracks"
            )

        needed: list[int] = sorted(set(explicit))
        for _, group in profile_specs:
            lo, hi = self.groups[group]
            needed.extend(range(lo, hi))
        indices = np.array(sorted(set(needed)), dtype=np.int64)
        position = {int(v): p for p, v in enumerate(indices)}

        rows: list[Row] = []
        for label, group in profile_specs:
            lo, hi = self.groups[group]
            rows.append(
                Row(
                    kind=PROFILE,
                    label=label,
                    members=tuple(position[i] for i in range(lo, hi)),
                    assay=group,
                )
            )
        for i in sorted(set(explicit)):
            meta = self.tracks[i]
            rows.append(
                Row(
                    kind=RAW,
                    label=meta["id"],
                    members=(position[i],),
                    index=i,
                    assay=meta.get("assay"),
                    description=meta.get("description"),
                )
            )

        if max_rows is not None and len(rows) > max_rows:
            raise ResponseBudgetExceeded(
                f"selection resolves to {len(rows)} rows; the limit for this format "
                f"is {max_rows}. Use format=npz, a smaller selection, or mode=aggregate.",
                field="tracks",
                requested_rows=len(rows),
                limit=max_rows,
            )

        return TrackSelection(
            indices=indices, rows=tuple(rows), normalize_profiles=normalize_profiles
        )


@lru_cache(maxsize=8)
def load_catalog(model: str) -> TrackCatalog:
    path = DATA_DIR / f"{model}_tracks.json"
    if not path.exists():
        raise FileNotFoundError(f"missing track catalog: {path}")
    raw = json.loads(path.read_text())
    return TrackCatalog(
        model=raw["model"],
        n_tracks=raw["n_tracks"],
        groups={k: list(v) for k, v in raw["groups"].items()},
        tracks=raw["tracks"],
    )
