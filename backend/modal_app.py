"""Modal deployment shim.

Deploy from this directory:

    cd backend && modal deploy modal_app.py

`add_local_file` and `add_local_python_source` resolve against the deploy working
directory, so deploying from the repo root will not find `data/` or `app/`.

Nothing here is imported by `app/`, and `app/` never imports modal — that keeps the
whole service testable with plain pytest and keeps a 1 s `import torch` off the path of
`/health` and the metadata endpoints.

This does not touch `index.py`. The existing deployment keeps serving
`https://tom-ellis-lab--yorzoi-app-fastapi-app.modal.run/generate` until the new
service is verified in a preview deployment and the frontend is switched over.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).parent

CUDA_TAG = "12.6.0-devel-ubuntu22.04"
MINUTES = 60

# Pin this. Modal caches a run_function layer on the source and referenced globals, so
# an unpinned revision means a rebuild can silently bake different weights, and there
# is nothing for a response or a cache key to bind to.
YORZOI_REVISION: str | None = None      # set to a commit SHA before deploying yorzoi

# huggingface.co/johahi/{borzoi,flashzoi}-replicate-0, main as of 2026-08-10.
BORZOI_REVISION = "521db103372769581b8d1d616e6a3b496da490c5"
FLASHZOI_REVISION = "72040fb525d4b9abab21ead1c53b76b6bd8b6d67"

# Prebuilt for exactly (cu12, torch 2.5, cp312, cxx11abi=FALSE) — the ABI of PyPI
# torch 2.5 wheels. Changing the torch pin below means changing this URL too;
# installing flash-attn from sdist instead is a ~30 min nvcc build.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

app = modal.App("yorzoi-api")

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    # torch comes from the CPU index in its own call: pointing index_url at
    # download.pytorch.org for the whole install would send fastapi and pydantic there
    # too, and that index does not host them.
    .pip_install("torch==2.5.*", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "fastapi[standard]==0.115.*",
        "numpy>=1.26",
        "pydantic>=2.9",
        "safetensors>=0.4",
        # Required even though weights load from safetensors: the vendored network
        # imports h5py at module scope for its TF-checkpoint loader. Dropping it makes
        # the container crash-loop on `import app.models._shorkie_net`.
        # tests/test_packaging.py enforces this against the module's actual imports.
        "h5py>=3.12",
    )
    # copy=True is a build step, so it has to precede any non-copy add_local_*.
    .add_local_dir(HERE / "data", "/root/data", copy=True)
    # Modal 1.0 removed automounting: without this the image builds, the deploy
    # succeeds, and the container dies on `import app`. Mounted at startup, so it
    # stays last and edits to app/ do not rebuild the image.
    .add_local_python_source("app")
)

gpu_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .pip_install("yorzoi==0.2.1", "fastapi[standard]==0.115.*", "numpy>=1.26", "pydantic>=2.9")
    .add_local_dir(HERE / "data", "/root/data", copy=True)
    .add_local_python_source("app")
)


def _bake_borzoi_weights():
    # Runs at image build with HF_HOME=/weights, so containers do no network I/O.
    # pytorch_model.bin is excluded: from_pretrained prefers the safetensors file and
    # the .bin duplicate is ~750 MB of dead image weight.
    from huggingface_hub import snapshot_download

    snapshot_download(
        "johahi/borzoi-replicate-0",
        revision=BORZOI_REVISION,
        allow_patterns=["*.json", "*.safetensors"],
    )


def _bake_flashzoi_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "johahi/flashzoi-replicate-0",
        revision=FLASHZOI_REVISION,
        allow_patterns=["*.json", "*.safetensors"],
    )


def _borzoi_family_image(bake) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        # Default PyPI torch bundles the CUDA 12 runtime; the flash-attn wheel above
        # is compiled against exactly this torch line.
        .pip_install("torch==2.5.*")
        .pip_install(
            "borzoi-pytorch==0.5.1",
            "fastapi[standard]==0.115.*",
            "numpy>=1.26",
            "pydantic>=2.9",
        )
        .env({"HF_HOME": "/weights"})
        .run_function(bake)
        # Containers must never fall back to the network for weights: offline mode
        # turns a cache miss into a loud boot failure instead of an unpinned download.
        .env({"HF_HUB_OFFLINE": "1"})
    )


def _finalize(image: modal.Image) -> modal.Image:
    # Local layers stay last so edits to app/ or data/ do not rebuild the image.
    return image.add_local_dir(HERE / "data", "/root/data", copy=True).add_local_python_source("app")


borzoi_image = _finalize(_borzoi_family_image(_bake_borzoi_weights))
flashzoi_image = _finalize(
    _borzoi_family_image(_bake_flashzoi_weights).pip_install(FLASH_ATTN_WHEEL)
)


@app.cls(
    image=base_image,
    cpu=4.0,
    memory=(4096, 16384),   # (request, limit) — a bare int is only a request
    timeout=150,            # Modal's web-endpoint ceiling; past it a 303 turns POST into GET
    scaledown_window=5 * MINUTES,
    max_containers=10,
)
@modal.concurrent(max_inputs=8, target_inputs=4)
class ShorkieService:
    """Shorkie runs on CPU.

    A 16384 bp forward pass measures ~0.10 s on four CPU threads and the model is only
    14.2 M parameters, so a full 8-sequence request is well under a second. Serving it
    from CPU removes the GPU cold start, the cross-container hop, and most of the
    150-second risk, at a fraction of the cost.
    """

    @modal.enter()
    def start(self):
        # Imported inside the container, never at module scope.
        from app.api import create_app
        from app.models.shorkie import ShorkieAdapter

        adapter = ShorkieAdapter(
            weights="/root/data/shorkie_f0.safetensors",
            params="/root/data/shorkie_params.json",
        )
        adapter.load("cpu")
        self.asgi = create_app({"shorkie": adapter})

    @modal.asgi_app()
    def web(self):
        return self.asgi


@app.cls(
    image=gpu_image,
    gpu="A100",
    memory=(8192, 24576),
    timeout=150,
    scaledown_window=10 * MINUTES,
    max_containers=4,
)
@modal.concurrent(max_inputs=4, target_inputs=2)
class YorzoiService:
    @modal.enter()
    def start(self):
        from app.api import create_app
        from app.models.yorzoi import YorzoiAdapter

        adapter = YorzoiAdapter(revision=YORZOI_REVISION)
        adapter.load("cuda:0")
        self.asgi = create_app({"yorzoi": adapter})

    @modal.asgi_app()
    def web(self):
        return self.asgi


@app.cls(
    image=borzoi_image,
    gpu="A100",
    memory=(8192, 24576),
    timeout=150,
    scaledown_window=10 * MINUTES,
    max_containers=4,
)
# One 524 kb window is ~16k attention tokens over a conv tower spanning half a million
# positions; two interleaved fp32 forwards is where a 40 GB A100 starts to be a
# gamble, so concurrency is lower than the small-window services.
@modal.concurrent(max_inputs=2, target_inputs=1)
class BorzoiService:
    @modal.enter()
    def start(self):
        from app.api import create_app
        from app.models.borzoi import BorzoiAdapter

        adapter = BorzoiAdapter(variant="borzoi", revision=BORZOI_REVISION)
        adapter.load("cuda:0")
        self.asgi = create_app({"borzoi": adapter})

    @modal.asgi_app()
    def web(self):
        return self.asgi


@app.cls(
    image=flashzoi_image,
    gpu="A100",
    memory=(8192, 24576),
    timeout=150,
    scaledown_window=10 * MINUTES,
    max_containers=4,
)
@modal.concurrent(max_inputs=2, target_inputs=1)
class FlashzoiService:
    @modal.enter()
    def start(self):
        from app.api import create_app
        from app.models.borzoi import BorzoiAdapter

        adapter = BorzoiAdapter(variant="flashzoi", revision=FLASHZOI_REVISION)
        adapter.load("cuda:0")
        self.asgi = create_app({"flashzoi": adapter})

    @modal.asgi_app()
    def web(self):
        return self.asgi
