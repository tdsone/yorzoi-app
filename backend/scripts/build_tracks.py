"""Build the track catalogs the service ships with.

    uv run python scripts/build_tracks.py

Shorkie: parses the released targets sheet (data/shorkie_targets.txt, downloaded from
https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/targets.txt).
Yorzoi:  reworks track_annotation.json into the same schema.

Assay is assigned by index range and cross-checked against the description text. The
sheet's own `group` column is not usable: it labels all 1128 ChIP-exo rows "Other".
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Contiguous blocks in the released sheet, verified against the file.
SHORKIE_BLOCKS = [
    ("chip_exo", 0, 1128, r"chip[-_ ]?exo"),
    ("chip_mnase", 1128, 1148, r"mnase"),
    ("rnaseq_idea", 1148, 4201, r"rna[-_ ]?seq"),
    ("rnaseq_strains", 4201, 5215, r"rna[-_ ]?seq"),
]


def build_shorkie() -> dict:
    src = DATA / "shorkie_targets.txt"
    with src.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    if len(rows) != 5215:
        raise SystemExit(f"expected 5215 rows, got {len(rows)}")

    assay_of: dict[int, str] = {}
    for assay, lo, hi, pattern in SHORKIE_BLOCKS:
        rx = re.compile(pattern, re.I)
        mismatched = 0
        for i in range(lo, hi):
            assay_of[i] = assay
            if not rx.search(rows[i]["description"]):
                mismatched += 1
        # The block ranges are the source of truth; the regex is a tripwire that the
        # released sheet has not been reordered underneath us.
        if mismatched > 0.02 * (hi - lo):
            raise SystemExit(
                f"{assay}: {mismatched}/{hi - lo} descriptions do not match /{pattern}/ "
                "— the sheet layout may have changed; re-derive the blocks before shipping"
            )

    tracks = []
    for i, row in enumerate(rows):
        if int(row["index"]) != i:
            raise SystemExit(f"index column is not contiguous at row {i}")
        tracks.append(
            {
                "i": i,
                "id": row["identifier"],
                "assay": assay_of[i],
                "description": row["description"],
            }
        )

    ids = {t["id"] for t in tracks}
    if len(ids) != len(tracks):
        raise SystemExit("duplicate identifiers in the sheet")

    return {
        "model": "shorkie",
        "n_tracks": len(tracks),
        "source": {
            "url": "https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/targets.txt",
            "md5": "58fc7ccf62ae5e9f29c444e85fbf9771",
            "bytes": 781531,
        },
        "groups": {a: [lo, hi] for a, lo, hi, _ in SHORKIE_BLOCKS},
        "tracks": tracks,
    }


def _library_id(filename: str) -> str:
    return re.sub(r"\.(plus|minus)\.bw$", "", filename)


def build_yorzoi() -> dict:
    """Build from data/yorzoi_track_annotation.json — the copy shipped with the model.

    Channel order is the 81 plus-strand tracks followed by their 81 minus-strand
    partners, so track i pairs with track i + 81.

    Do NOT build this from the repo-root track_annotation.json. That file holds the
    same two sets of names in a different order: plus[i] and minus[i] name the same
    library for only 12 of 81 positions, so pairing by position — which is exactly what
    the strand_pair field and the frontend's `coverage[trackIdx + 81]` both do — would
    be wrong for 69 libraries. The assertion below is what keeps that from recurring.
    """
    src = DATA / "yorzoi_track_annotation.json"
    ann = json.loads(src.read_text())
    plus, minus = ann["+"], ann["-"]
    if len(plus) != len(minus):
        raise SystemExit(f"strand lists differ in length: {len(plus)} vs {len(minus)}")

    mismatched = [
        i for i, (p, m) in enumerate(zip(plus, minus)) if _library_id(p) != _library_id(m)
    ]
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)}/{len(plus)} positions pair different libraries "
            f"(first at index {mismatched[0]}: {plus[mismatched[0]]} vs "
            f"{minus[mismatched[0]]}). This annotation file is shuffled and must not "
            "be used to build a paired catalog."
        )

    tracks = []
    for strand, names in (("+", plus), ("-", minus)):
        for name in names:
            i = len(tracks)
            tracks.append(
                {
                    "i": i,
                    "id": f"{_library_id(name)}:{strand}",
                    "assay": "rnaseq",
                    "strand": strand,
                    "library_id": _library_id(name),
                    "description": name,
                    "strand_pair": (i + len(plus)) if strand == "+" else (i - len(plus)),
                }
            )

    return {
        "model": "yorzoi",
        "n_tracks": len(tracks),
        "source": {"file": "data/yorzoi_track_annotation.json (shipped with the model)"},
        "groups": {"plus": [0, len(plus)], "minus": [len(plus), len(tracks)]},
        "tracks": tracks,
    }


def main() -> int:
    for name, build in (("shorkie", build_shorkie), ("yorzoi", build_yorzoi)):
        catalog = build()
        out = DATA / f"{name}_tracks.json"
        out.write_text(json.dumps(catalog, separators=(",", ":")))
        print(f"{out.name}: {catalog['n_tracks']} tracks, {out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
