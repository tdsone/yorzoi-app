"""Structural guarantees that are easy to break and expensive to notice."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_app_does_not_import_modal():
    """`app/` must stay deployable-agnostic and testable without modal installed."""
    result = subprocess.run(
        [sys.executable, "-c", "import app.api, app.engine; import sys; print('modal' in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_core_does_not_import_torch():
    """A 1 s torch import must not sit on the path of /health or the metadata routes."""
    result = subprocess.run(
        [sys.executable, "-c", "import app.api; import sys; print('torch' in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_vendored_model_is_in_sync():
    """Fails when the vendored network definition drifts from its upstream checkout."""
    script = ROOT / "scripts" / "vendor_shorkie.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"], cwd=ROOT, capture_output=True, text=True
    )
    if "no shorkie.py under" in result.stderr:
        pytest.skip("upstream checkout not present on this machine")
    assert result.returncode == 0, result.stdout + result.stderr


def test_vendored_model_records_its_provenance():
    header = (ROOT / "app" / "models" / "_shorkie_net.py").read_text()[:600]
    assert "shorkie-pytorch @" in header
    assert "Do not edit here" in header


class TestCatalogsAreShipped:
    @pytest.mark.parametrize("model,count", [("shorkie", 5215), ("yorzoi", 162)])
    def test_catalog_is_present_and_complete(self, model, count):
        catalog = json.loads((ROOT / "data" / f"{model}_tracks.json").read_text())
        assert catalog["n_tracks"] == count
        assert len(catalog["tracks"]) == count
        assert [t["i"] for t in catalog["tracks"]] == list(range(count))

    def test_shorkie_catalog_records_its_source(self):
        catalog = json.loads((ROOT / "data" / "shorkie_tracks.json").read_text())
        assert catalog["source"]["md5"] == "58fc7ccf62ae5e9f29c444e85fbf9771"

    def test_yorzoi_catalog_pairs_strands_correctly(self):
        # The repo-root track_annotation.json is a shuffled permutation that pairs
        # correctly for only 12 of 81 libraries; building from it would mislabel the
        # partner of 69 tracks.
        tracks = json.loads((ROOT / "data" / "yorzoi_tracks.json").read_text())["tracks"]
        for track in tracks:
            assert tracks[track["strand_pair"]]["library_id"] == track["library_id"]
            assert tracks[track["strand_pair"]]["strand"] != track["strand"]


def test_deployment_image_covers_the_vendored_module_imports():
    """The vendored network imports h5py at module scope even when weights come from
    safetensors. Omitting it from the image is a crash-loop at container start, which
    only shows up after a deploy — so check it here instead."""
    import ast

    net = ROOT / "app" / "models" / "_shorkie_net.py"
    tree = ast.parse(net.read_text())
    third_party = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                third_party.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            third_party.add(node.module.split(".")[0])
    third_party -= set(sys.stdlib_module_names)

    spec = (ROOT / "modal_app.py").read_text()
    # The CPU predict image is everything up to the first GPU image base.
    marker = "_yorzoi_base = ("
    assert marker in spec, "modal_app.py image layout changed; update this test's marker"
    cpu_image = spec.split(marker)[0]
    missing = [name for name in sorted(third_party) if f'"{name}' not in cpu_image]
    assert not missing, f"deployment image is missing: {missing}"
