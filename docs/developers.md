# Developer guide

This guide covers local development, tests, model assets, and deployment. For the
website and API request/response formats, see the [user guide](users.md). The
[root README](../README.md) gives a shorter overview.

Jump to [architecture](#two-backend-paths),
[installation](#prerequisites-and-installation),
[local API development](#local-api-development-without-weights),
[tests](#tests-and-validation),
[model assets](#weights-catalogs-and-fixtures),
[adding models](#adding-or-changing-a-model), or
[deployment](#modal-deployment).

## Two backend paths

The frontend currently calls the legacy Yorzoi `/generate` endpoint implemented in
[`backend/index.py`](../backend/index.py). The newer service in
[`backend/app/`](../backend/app/) implements the `/v1` API for Yorzoi, Shorkie,
Borzoi, and Flashzoi, plus asynchronous in-silico saturation mutagenesis (ISM) for
Yorzoi and Shorkie. Changing or deploying the newer service does not switch the
frontend to it.

| Area | Entry points and responsibilities |
| --- | --- |
| Browser app | [`frontend/src/App.jsx`](../frontend/src/App.jsx): React UI, sequence changes, legacy API requests, and coverage panels. Redux state lives in [`store.js`](../frontend/src/store.js). |
| Editor integration | [`frontend/patches/`](../frontend/patches/) and [`rebuild-ove.cjs`](../frontend/scripts/rebuild-ove.cjs): patch and rebuild Open Vector Editor. |
| Legacy backend | [`backend/index.py`](../backend/index.py): Modal app `yorzoi-app`, Yorzoi inference, and `/generate`. |
| v1 API | [`app/api.py`](../backend/app/api.py): `create_app(...)`, metadata routes, prediction routes, optional ISM routes, CORS, and serialization. |
| Prediction pipeline | [`app/engine.py`](../backend/app/engine.py): validate, encode, tile, batch, merge, and reduce. |
| Shared geometry | [`app/core/`](../backend/app/core/): model specifications, coordinates, encoding, track selection, errors, and serialization. |
| Model implementations | [`app/models/`](../backend/app/models/): adapter protocol, real adapters, deterministic test adapter, and vendored Shorkie network. |
| ISM | [`app/ism/`](../backend/app/ism/): validation, mutant batches, metrics, logos, worker execution, and the job-launcher interface. |
| Cloud deployment | [`backend/modal_app.py`](../backend/modal_app.py): Modal app `yorzoi-api`, images, per-model services, ISM gateway, and workers. |
| Data and checks | [`backend/data/`](../backend/data/), [`backend/scripts/`](../backend/scripts/), and [`backend/tests/`](../backend/tests/). |

The `app/` package does not import Modal. Importing the API or prediction engine
does not import PyTorch; real adapters import it when loading or running a model.
This allows the HTTP contract and coordinate handling to be tested without cloud
credentials, model downloads, or a GPU. Packaging tests protect this boundary.

## Prerequisites and installation

Use Git, Node.js with npm, Python 3.12 or newer, and `uv`. The frontend does not pin
a Node version; use a Node installation compatible with its locked dependencies.
The deployment images explicitly use Python 3.12. A Modal account is needed only
to deploy services, not to call the hosted API or run the local tests below.

Run frontend commands from `frontend/`:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`. Installation
runs `patch-package` and rebuilds `@teselagen/ove` from source. `npm run dev`
repeats the editor rebuild and starts Vite with a fresh dependency prebundle;
`npm start` starts Vite without repeating that rebuild. The production build is:

```bash
# From frontend/
npm run build
```

Output is written to `frontend/dist/`. Keep the patch and rebuild steps when
upgrading the editor: this app depends on behavior provided by the local patch.
Do not install with lifecycle scripts disabled unless you subsequently run the
`postinstall` script yourself. There is currently no `npm test` or lint script;
the older `App.test.js` file is not a configured frontend test suite.

In a separate terminal, install the backend environment from the repository root:

```bash
cd backend
uv sync --group dev
uv run pytest
```

[`pyproject.toml`](../backend/pyproject.toml) and
[`uv.lock`](../backend/uv.lock) describe the local environment. The development
group includes pytest, HTTPX, PyTorch, HDF5 support, safetensors, and matplotlib.
It does not include the real Yorzoi or Borzoi packages; Modal images install those
separately. Local dependency installation can be large because of PyTorch.

## Local API development without weights

`create_app()` is a factory, not a module-level ASGI `app`. With no adapters it
serves health and model metadata, but prediction requests return `503` because no
model is loaded. The model catalog lists all four registered models even when a
particular server only loads one; `/health` reports the `loaded` list.

This complete smoke test uses the deterministic Shorkie stub. Run it from
`backend/` after installing the development dependencies:

```bash
uv run python - <<'PY'
from fastapi.testclient import TestClient
from app.api import create_app
from app.core.spec import SHORKIE
from app.models.stub import StubAdapter

adapter = StubAdapter(SHORKIE)
adapter.load()
with TestClient(create_app({"shorkie": adapter})) as client:
    assert client.get("/health").json()["loaded"] == ["shorkie"]
    response = client.post(
        "/v1/models/shorkie/predict",
        json={"sequences": [{"id": "smoke", "seq": "ACGT" * 250}], "tracks": [0]},
    )
    response.raise_for_status()
    prediction = response.json()["predictions"][0]
    assert prediction["id"] == "smoke"
    assert prediction["n_bins"] == 63
    print({"id": prediction["id"], "n_bins": prediction["n_bins"]})
PY
```

Stub values are synthetic and intended for integration work, not biological
interpretation. For an HTTP server usable from another terminal or client, the
following starts the same stub on port 8000. `--with uvicorn` adds the server to
this command's environment because it is not a declared local dependency:

```bash
# From backend/
uv run --with uvicorn python - <<'PY'
import uvicorn
from app.api import create_app
from app.core.spec import SHORKIE
from app.models.stub import StubAdapter

adapter = StubAdapter(SHORKIE)
adapter.load()
uvicorn.run(create_app({"shorkie": adapter}), host="127.0.0.1", port=8000)
PY
```

In another terminal, check `curl http://127.0.0.1:8000/health` or open
`http://127.0.0.1:8000/docs` for the generated API documentation. Stop the server
with Ctrl-C. The frontend will still send requests to its hardcoded legacy URL;
starting this server does not redirect browser requests.

For local ISM development, inject
[`InProcessLauncher`](../backend/app/ism/jobs.py) as well as adapters:

```python
from app.ism.jobs import InProcessLauncher

adapters = {"shorkie": adapter}
app = create_app(adapters, ism_launcher=InProcessLauncher(adapters, defer=True))
```

This launcher executes the job synchronously during submission. `defer=True`
simulates one `running` poll before returning the result; it is a test aid, not a
background queue. See [`test_ism_api.py`](../backend/tests/test_ism_api.py) for
small requests and polling examples. Without a launcher, the ISM routes are not
registered.

## Tests and validation

Run commands in this section from `backend/`:

```bash
uv run pytest
uv run pytest tests/test_api.py tests/test_coords.py tests/test_encode.py
uv run pytest tests/test_ism_api.py tests/test_ism_engine.py tests/test_ism_metric.py
```

The suite covers sequence validation, bin coordinates, tiling and stitching,
track selection, API errors, JSON/NPZ serialization, packaging, ISM math, and logo
generation. Tests using the deterministic adapter detect coordinate and channel
errors through values derived from the input bases.

Real Shorkie equivalence and integration tests compare against the committed
TensorFlow fixture. They skip when the checkpoint is absent. The default lookup
expects a sibling `shorkie-pytorch/data/` directory beside this repository; use
explicit paths on other layouts:

```bash
# Replace the checkpoint path with your fold-0 HDF5 checkpoint.
SHORKIE_CHECKPOINT=/absolute/path/to/checkpoint_f0.h5 \
SHORKIE_PARAMS="$PWD/data/shorkie_params.json" \
uv run pytest tests/test_equivalence.py tests/test_integration.py
```

The optional comparison with tangermeme skips unless tangermeme is installed:

```bash
uv run --with tangermeme pytest tests/test_ism_tangermeme.py
```

This extra dependency is unnecessary for the normal suite. The `slow` marker is
declared in pytest configuration, but checkpoint availability is what controls
the current equivalence/integration skips; do not rely on `-m 'not slow'` to
exclude those tests.

For frontend changes, run `npm run build` from `frontend/` and exercise sequence
editing/import, automatic prediction, strand panels, loading state, and errors in
the browser. Backend tests do not validate the frontend's editor integration.

## Weights, catalogs, and fixtures

### Shorkie

The network implementation is vendored in
[`_shorkie_net.py`](../backend/app/models/_shorkie_net.py); the adapter supports
both the original HDF5 checkpoint and converted safetensors. Deployment expects
`backend/data/shorkie_f0.safetensors`, which is intentionally gitignored.

With a fold-0 checkpoint and its matching parameter file available, run from
`backend/`:

```bash
uv run python scripts/convert_weights.py \
  --checkpoint /absolute/path/to/checkpoint_f0.h5 \
  --params data/shorkie_params.json
```

The converter writes `data/shorkie_f0.safetensors` and
`data/shorkie_f0_provenance.json`, copies the supplied parameters into
`data/shorkie_params.json`, and checks that every state tensor survives the
safetensors round trip exactly. The provenance records source and output SHA-256
hashes. Review generated tracked-file changes before committing them. Checkpoint
source URLs and fold naming are documented in
[`convert_weights.py`](../backend/scripts/convert_weights.py).

To serve real Shorkie locally, replace the stub construction in the local-server
example with:

```python
from app.models.shorkie import ShorkieAdapter

adapter = ShorkieAdapter("data/shorkie_f0.safetensors", "data/shorkie_params.json")
adapter.load("cpu")
```

Use an explicit upstream checkout when checking or refreshing the vendored code;
the script's default is a maintainer-specific absolute path:

```bash
# From backend/; replace the source directory with your local checkout.
uv run python scripts/vendor_shorkie.py --src /absolute/path/to/shorkie-pytorch --check
uv run python scripts/vendor_shorkie.py --src /absolute/path/to/shorkie-pytorch
```

The second command rewrites the vendored file and its source-commit header.
Review the upstream changes and rerun numerical tests when updating it.

### Yorzoi, Borzoi, and Flashzoi

[`modal_app.py`](../backend/modal_app.py) pins Hugging Face revisions for all
three models. Yorzoi uses `yorzoi==0.2.1` and loads its pinned weights during
container startup. Borzoi and Flashzoi use `borzoi-pytorch==0.5.1`; their image
builds download the pinned configuration and safetensors, then set
`HF_HUB_OFFLINE=1` for runtime. A missing baked cache should fail startup rather
than silently fetch a different revision.

Flashzoi requires CUDA and runs with bf16 autocast. Its prebuilt flash-attn wheel
is tied to the Python, PyTorch, CUDA, and ABI choices in `modal_app.py`; update
those together. The local `uv` environment does not reproduce these GPU images.

### Track catalogs and the golden fixture

The four catalog JSON files are generated from the committed target sheets and
Yorzoi annotation file. Rebuild them from `backend/`:

```bash
uv run python scripts/build_tracks.py
```

The builder verifies expected track counts, assay boundaries, and strand pairing.
For Yorzoi, its source is `data/yorzoi_track_annotation.json`, not the legacy
`backend/track_annotation.json`. Catalog order must match model output channels.

The small Shorkie golden fixture is already committed and is sufficient to run
the reference comparisons once weights are available. Regenerating it requires
the original `predictions_f0_chunk009.h5` reference data:

```bash
# From backend/; --src is the directory containing the original HDF5 predictions.
uv run python scripts/build_fixture.py --src /absolute/path/to/shorkie-pytorch/data
```

This rewrites the NPZ and JSON files under `tests/fixtures/`. It extracts four
genomic inputs and 64 tracks spanning the output head; it does not generate new
TensorFlow predictions or download the source data.

## Adding or changing a model

1. Add a `ModelSpec` to [`core/spec.py`](../backend/app/core/spec.py) and register
   it in `REGISTRY`. Specify the exact network input length, crop, bin size, output
   tracks, strand semantics, units, maximum length, and tile stride. The predicted
   span must contain whole bins, and the stride must be a positive bin multiple
   no larger than that span.
2. Implement [`ModelAdapter`](../backend/app/models/base.py). Inputs are float32
   arrays shaped `(batch, 4, seq_length)` in A/C/G/T order; outputs are float32
   `(batch, n_bins, selected_tracks)` arrays. Validate shapes, load in evaluation
   mode, use inference mode, and select channels on the compute device before
   copying results back to the CPU. Record weights/revision and any unresolved
   assumptions in `describe()`.
3. Generate a matching `data/<model>_tracks.json` catalog. Add presets and a default
   selection in [`core/tracks.py`](../backend/app/core/tracks.py), preserving stable
   track identifiers and correct strand pairs.
4. Add geometry, encoding, API, and catalog tests, plus numerical reference tests
   when a real checkpoint is available. A plausible output shape alone does not
   prove the alignment or prediction values are correct.
5. Add the deployment image and service in
   [`modal_app.py`](../backend/modal_app.py), pin dependency/weight revisions, and
   choose resource limits and batch sizes for the actual input geometry.
6. Update the [user guide](users.md) and metadata examples. ISM support is separate:
   it also requires an entry in `ISM_MODELS`, worker/launcher wiring, and tests for
   its metrics and reverse-complement strand handling.

Keep model-specific transforms in the adapter, so the shared prediction engine
can continue to operate on one layout. Yorzoi inverts its training transform in
float32 before tile averaging; Borzoi and Flashzoi retain their scaled, squashed
training value space. Changing these semantics is an API-visible change.

## Modal deployment

The deployed Modal application is named `yorzoi-api`. Maintainers can manage it
through the [Modal dashboard](https://modal.com/apps/tom-ellis-lab/main/deployed/yorzoi-api).
Users can call the [public service URLs](users.md#connecting-to-the-versioned-api)
directly using the [Python quickstart](users.md#python-quickstart); no Modal
account or deployment is required to use the hosted API.

Its topology is defined in [`modal_app.py`](../backend/modal_app.py):

| Component | Resources | Role |
| --- | --- | --- |
| `ShorkieService` | 4 CPU cores | Loads Shorkie and serves prediction requests. |
| `YorzoiService` | A100 GPU | Loads Yorzoi and serves prediction requests. |
| `BorzoiService`, `FlashzoiService` | A100 GPU each | Separate large-window prediction services with lower concurrency. |
| `IsmGateway` | CPU | Validates ISM submissions, spawns workers, and serves polling requests. |
| `YorzoiIsmWorker`, `ShorkieIsmWorker` | A100 GPU each | Run ISM jobs, including metric calculation and logo rendering. |

Each prediction service loads only its own model. Its metadata routes still list
all registered models, but requests for other models return `503`. ISM endpoints
exist on the gateway; prediction endpoints on that gateway have no loaded
adapters. Use the separate origins in the
[public endpoint table](users.md#connecting-to-the-versioned-api). When deploying
to another workspace, record the URLs printed by Modal for that deployment.

The code configures 150-second timeouts for web services and 30-minute timeouts
for ISM workers. The gateway uses Modal FunctionCall IDs as job IDs and retrieves
results without maintaining its own database. Expired outputs map to HTTP `410`;
clients should save completed results they need to retain.

Before deploying, prepare `data/shorkie_f0.safetensors`, run the tests, and check
the pinned dependency and model revisions. With Modal credentials configured for
the intended workspace, run from the repository root:

```bash
cd backend
uv run modal deploy modal_app.py
```

Run from `backend/`: the deployment needs the local `app` package and data assets.
This command builds images and updates the `yorzoi-api` application in the
configured Modal environment. It can download model weights and start billable
cloud resources. It is not a local test command.

### Configuration

| Setting | Where it applies | Behavior |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | API process | Comma-separated CORS origins. Defaults include the two Yorzoi website origins and localhost/127.0.0.1 on port 5173. |
| `ISM_BATCH_SIZE` | ISM planning/execution processes | Overrides the default batch size: 64 for Yorzoi and 32 for Shorkie. Use a positive integer sized for the available memory. |
| `SHORKIE_ALLOW_TF32` | Shorkie adapter | Only the value `1` enables TF32; it is disabled by default to preserve reference fidelity. |
| `SHORKIE_CHECKPOINT`, `SHORKIE_PARAMS` | Local equivalence/integration tests | Override the default checkpoint and parameter paths. These do not configure deployed service paths. |
| `HF_HOME`, `HF_HUB_OFFLINE` | Borzoi-family Modal images | Set in the image definitions for the baked weight cache and offline runtime. |

The application does not load `.env` files itself. Export variables for a local
process, or configure the relevant Modal image/container environment explicitly;
a local shell export does not by itself configure remote workers. Apply ISM batch
configuration consistently to the gateway and worker. Model revisions, resource
limits, and checkpoint locations are code settings in `modal_app.py` rather than
general environment overrides.

### Frontend cutover

The old and new deployments use different Modal application names. Verify the
new services in the intended environment before changing
[`fetchPredictedCoverage`](../frontend/src/App.jsx). Cutover requires a response
adapter as well as a URL change: legacy output is keyed by sequence and contains
per-base tracks, while v1 returns prediction records, selected tracks, and bins
with explicit coordinates. Use `bin_start_bp`, `bin_size_bp`, and the track
metadata when mapping predictions into editor panels. For v1 Yorzoi, expanding a
bin to per-base display repeats its per-base-average value across the bin.

Check representative existing sequences, exact native windows, shorter padded
inputs, longer tiled inputs, and forward/reverse track pairing. Capture a Yorzoi
reference fixture before relying on an assumed equivalence with the old UI.
Updating `modal_app.py` alone leaves the existing frontend on `/generate`.

## Current limitations to preserve in reviews

- Shorkie's species index is hardcoded to `109` and reported as unverified. The
  TensorFlow fixture uses that same index, so numerical equivalence cannot validate
  the biological choice.
- Yorzoi's 4,992 bp input and 996 bp crop are inherited serving conventions, with
  `geometry_verified: false`. There is no Yorzoi golden fixture in this repository
  to resolve the potential offset from the training-data geometry.
- All model outputs have `depth_normalized: false`. Default profile presets
  normalize tracks to their own window means for display; raw track magnitudes
  should not be compared across assays. Padding and ambiguous bases affect the
  predictions, and the API exposes warnings and fractions for that reason.
- ISM is limited to Yorzoi and Shorkie, one sequence fitting one native input
  window. Its in-process launcher is for tests; the Modal launcher provides the
  deployed asynchronous behavior.
- The browser app remains a legacy Yorzoi client. It does not expose the four-model
  v1 API or ISM workflow, and it currently logs submitted sequences and prediction
  results to the browser console.
- CORS restricts browser origins but is not authentication. The API code does not
  implement user authentication or request rate limiting; account for this when
  operating publicly reachable compute endpoints.
