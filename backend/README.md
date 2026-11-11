# Prediction API

One model-agnostic service for **shorkie**, **yorzoi**, **borzoi** and **flashzoi**.
DNA in, per-track coverage out.

Use our hosted deployment with the [Python quickstart](../docs/users.md#python-quickstart).
The [public service URLs](../docs/users.md#connecting-to-the-versioned-api) cover
all four prediction models and the ISM gateway; no Modal account or local model
installation is needed to call them.

The models differ only in constants — window size, crop, bin size, track count — so
they share one implementation of validation, tiling, stitching, track selection and
serialization. Adding a model is a `ModelSpec` plus an adapter; flashzoi is borzoi's
geometry under different weights and precision, and one adapter serves both.

| | window | crop/edge | bin | bins | tracks |
|---|---|---|---|---|---|
| shorkie | 16384 bp | 1024 bp | 16 bp | 896 | 5215 (unstranded) |
| yorzoi | 4992 bp | 996 bp | 10 bp | 300 | 162 (81 stranded pairs) |
| borzoi / flashzoi | 524288 bp | 163840 bp | 32 bp | 6144 | 7611 (CAGE/RNA-seq stranded) |

## Layout

```
app/
  core/     spec, errors, encode, coords, tracks, serialize   (numpy only, no torch)
  models/   adapter protocol + shorkie / yorzoi / borzoi(+flashzoi) / stub,
            and the vendored shorkie network
  engine.py encode -> tile -> batch -> merge -> reduce
  api.py    FastAPI factory
modal_app.py   deployment shim; app/ never imports modal
```

Importing `app.api` or `app.engine` pulls in neither `modal` nor `torch` (enforced by
`tests/test_packaging.py`) — torch lives behind lazy imports in the adapters — so
`/health` and the metadata routes stay off a ~1 s torch import and the whole service
is testable with plain pytest.

## Endpoints

```
GET  /health                        liveness; never touches a model
GET  /v1/models                     geometry of every registered model
GET  /v1/models/{model}             one model + limits with worked examples
GET  /v1/models/{model}/tracks      full track catalog (cacheable)
POST /v1/models/{model}/predict     ?format=json|npz&bench=true
POST /v1/models/{model}/ism         submit a saturation-mutagenesis job (202 + job id)
GET  /v1/ism/jobs/{job_id}          ?format=json|npz — poll status / fetch the result
```

The ISM routes are served by the dedicated gateway deployment (see the ISM section);
the per-model predict services do not register them.

```bash
# Default: assay profiles over a 1 kb promoter, ~1 KB of JSON.
curl -sS -X POST localhost:8000/v1/models/shorkie/predict \
  -H 'content-type: application/json' -d '{"sequences":["ACGT..."]}' \
  | jq '.predictions[0] | {n_bins, bin_start_bp, pad_fraction, warnings}'

# Variant effect: all 5215 tracks reduced to one number each.
# Measured 62 KB gzipped, against ~90 MB for the same request as a full matrix.
curl -sS -X POST localhost:8000/v1/models/shorkie/predict \
  -H 'content-type: application/json' \
  -d '{"sequences":[{"id":"wt","seq":"…"},{"id":"alt","seq":"…"}],
       "mode":"aggregate","preset":"all"}'

# Everything, as binary, for numpy users.
curl -sS -X POST 'localhost:8000/v1/models/shorkie/predict?format=npz' \
  -H 'content-type: application/json' -d '{"sequences":["…16384bp…"],"preset":"all"}' -o out.npz
python -c "import numpy as np; print(np.load('out.npz')['predictions'].shape)"
```

Matrix responses are capped at 25 M values — enough for shorkie's full 5215-track
matrix, not for borzoi/flashzoi's 7611 x 6144 window (46.8 M). There, use
`mode: "aggregate"` or a narrower selection.

Requests take `sequences` (1–8, plain strings or `{id, seq}`), a selection
(`tracks` | `track_groups` | `preset`), `mode` (`matrix` | `aggregate`) and `window`
(`auto` | `full` | `native`). Responses are index-aligned arrays keyed by your `id` —
never by the sequence string, which silently collapses duplicates and echoes kilobytes
back at you.

## In-silico saturation mutagenesis (ISM)

Available for **yorzoi** and **shorkie**. Every position in a window is mutated to
every other base, each mutant is scored with a scalar metric u, and the result is the
data behind a sequence plot: for every position, u for all four bases, plus
tangermeme-equivalent attribution matrices and server-rendered logos.

ISM is an **async job** — a full-window scan is minutes of GPU, past Modal's 150 s
web ceiling — served by a dedicated CPU gateway that validates, spawns an A100
worker, and polls. Polling never keeps a GPU warm; the GPU bills only for the job.

```
POST /v1/models/{model}/ism         -> 202 {job_id, n_mutants, n_forwards, poll}
GET  /v1/ism/jobs/{job_id}          -> {status: running | failed | succeeded, ...}
GET  /v1/ism/jobs/{job_id}?format=npz   binary matrices, PNGs excluded
```

```bash
# Submit: mutate a 300 bp window, score u = mean coverage over a gene body,
# strand-averaged (u = (u_fwd + u_revcomp)/2), logos for both normalizations.
curl -sS -X POST "$GATEWAY/v1/models/yorzoi/ism" \
  -H 'content-type: application/json' -d '{
    "sequence": {"id": "my-promoter", "seq": "ACGT..."},
    "ism_window": {"start": 200, "end": 500},
    "metric": {"interval": {"start": 500, "end": 900},
               "strand_average": true, "agg": "mean"},
    "plots": {"normalization": "both"}
  }'
```

```python
import time, requests
job = requests.post(f"{GATEWAY}/v1/models/yorzoi/ism", json=body).json()
while (r := requests.get(f"{GATEWAY}{job['poll']}").json())["status"] == "running":
    time.sleep(5)
result = r["result"]                      # matrices + base64 PNG logos
raw = result["matrices"]["raw"]           # [4][W]: u per base per position
```

Request semantics:

- **One sequence, one window**: `len(seq) <= seq_length` (4992 yorzoi / 16384
  shorkie), no tiling. Mutating positions outside the predicted span is allowed —
  they affect predictions inside it — but `metric.interval` must lie within the
  predicted span. N is rejected inside `ism_window` (no reference base), legal
  elsewhere.
- **`metric`** (prototype `interval_mean`, registry-extensible): per-bin values
  averaged over `interval` (partial edge bins overlap-weighted; `agg: "sum"` skips
  the normalization), averaged across the selected `tracks`/`track_groups`/`preset`
  (raw values — no per-window mean normalization, which would hide uniform effects).
  `strand_average: true` is reverse-complement TTA: every input also runs
  reverse-complemented (yorzoi: paired-strand tracks swapped, coordinates mirrored)
  and u is the average — at twice the forward passes.
- **`plots.normalization`** picks logos only; the matrices always include everything:
  `raw` (4xW, u per base, reference entries = u0), `centered` (raw minus its
  per-position mean — tangermeme's hypothetical view; the observed-base projection of
  this is the standard ISM logo), `projected` (W, `centered` at the observed base,
  = u0 − mean). Not z-scored, matching tangermeme; verified equal to
  `tangermeme.saturation_mutagenesis` to float precision in
  `tests/test_ism_tangermeme.py`.

Costs, roughly: a full-window yorzoi scan is ~15k forwards (~tens of seconds of
A100); full-window shorkie ~49k (~minutes); `strand_average` doubles both. The
host-transfer per mutant scales with the track selection — an all-track shorkie
metric moves ~19 MB per forward, so select the tracks you mean. JSON results are
~0.3–1 MB plus ~50–200 KB per logo. Job results live in Modal's FunctionCall store
(~7-day retention); after that, polls return **410** and you re-submit.

## Reading the output

`values` is `[track][bin]`. Bin *k* covers
`[bin_start_bp + k*bin_size_bp, +bin_size_bp)` of **your** sequence.

- **`bin_start_bp` is not always 0.** In `native` mode the model predicts only the
  centre of its window: shorkie reports 1024, yorzoi 996, borzoi/flashzoi 163840.
  Never assume the first bin is the first base.
- **`window: auto`** picks `native` when your sequence is exactly the model's window
  length, and `full` otherwise. The two differ: for shorkie at exactly 16384 bp,
  `full` produces two tiles that are each N-padded on one side and averages them,
  which measures ~2.6% away from the single clean pass. Every JSON response states
  which mode ran.
- **Magnitudes are not comparable across tracks.** `meta.depth_normalized` is `false`
  for every model: each track carries its own sequencing-depth factor, and
  borzoi/flashzoi additionally emit their training space — per-track scaling and
  soft-clip squashing are not inverted. Compare positions and sequences, not tracks.
  Preset rows are normalized to their own window mean and labelled
  `units: relative_to_window_mean` — shapes, not coverage.
- **Short sequences are padded with N and `pad_fraction` says how much.** Padding
  measurably distorts the prediction over the real sequence, not just the flanks, and
  the first and last `crop_bp` of any `full`-mode answer are predicted partly from
  padding and run systematically high. Comparisons between sequences at the same
  padding are sound; absolute magnitudes for short inputs are not.

## Development

```bash
cd backend
uv sync --group dev
uv run pytest                       # 150 tests
uv run python scripts/build_tracks.py
```

The equivalence tests need the fold-0 checkpoint; they skip without it. Point
`SHORKIE_CHECKPOINT` at it, or keep a `shorkie-pytorch` checkout beside this repo.

`uv sync` pulls a CPU torch build for tests. On Linux the resolver will also fetch
CUDA wheels for it — torch is a test-only dependency here and the deployment images
install their own, so pin a CPU index if that ever matters.

### Weights

**Borzoi / flashzoi** load from Hugging Face (`johahi/borzoi-replicate-0`,
`johahi/flashzoi-replicate-0`) via `borzoi-pytorch`. The Modal images bake the
revisions pinned in `modal_app.py` at build and run with `HF_HUB_OFFLINE=1`, so
containers never touch the network for weights.

**Shorkie** weights are converted once:

```bash
uv run python scripts/convert_weights.py \
  --checkpoint ../../shorkie-pytorch/data/checkpoint_f0.h5 \
  --params     ../../shorkie-pytorch/data/shorkie_params.json
```

Writes `data/shorkie_f0.safetensors` (gitignored) and a provenance file with both
SHA-256s, and asserts the round trip is exact. Upstream checkpoints are at
`https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f{0..7}/model_best.h5`
— note `shorkie_models/`; the paths in the shorkie-pytorch README point at a location
that now returns `NoSuchKey`.

The network definition is **vendored** at `app/models/_shorkie_net.py` because
`shorkie-pytorch` is not currently pip-installable: its pyproject has no
`[build-system]` and no `py-modules`, so the built wheel contains only metadata —
`pip install` succeeds and `import shorkie` then fails. Re-sync with
`uv run python scripts/vendor_shorkie.py`; `--check` fails on drift.

## Testing

| tier | what it proves | cost |
|---|---|---|
| unit | tiling covers every base at every length and stride for every model, with truth defined as a function of bp so a phase error is detectable | <1 s |
| API | every endpoint and error path against a deterministic stub whose output depends on the bases each bin covers | <1 s |
| equivalence | the **whole pipeline** reproduces the original TensorFlow shorkie | ~2 s |
| integration | the real checkpoint over the real ASGI stack, JSON and npz | ~2 s |

Measured response sizes for one 16384 bp shorkie sequence (gzipped where applicable):
4 default profiles 6.9 KB · 64-track matrix 93 KB · all-5215 aggregate 62 KB ·
all-5215 npz 19.9 MB uncompressed.

The equivalence fixture (`tests/fixtures/shorkie_golden.npz`, 880 KB) holds 4 genomic
sequences with the exact one-hot tensor TensorFlow was fed, plus its outputs for 64
tracks sampled across the full 0–5214 range. Measured: **Pearson 1.0000000000, max
absolute deviation 9.9e-05**.

## Deploying

```bash
cd backend && modal deploy modal_app.py
```

Paths resolve against the deploy directory, so run it from `backend/`. Four predict
services plus the ISM tier:

- **shorkie** serves from **CPU** — a 16384 bp forward is ~0.10 s on four threads for
  a 14.2 M-parameter model, so a GPU buys nothing and costs a cold start.
- **yorzoi**, **borzoi** and **flashzoi** each keep an A100. Flashzoi is CUDA-only
  (flash_attn) and runs under bf16 autocast — a ~3x faster borzoi distillation with
  the same head. Its image installs a flash-attn wheel prebuilt for the pinned torch;
  changing the torch pin means changing the wheel URL in `modal_app.py`.
- **IsmGateway** (CPU) hosts the ISM routes for both models and spawns
  **YorzoiIsmWorker** / **ShorkieIsmWorker** (A100, 30 min timeout, 2 min
  scaledown). Shorkie ISM runs on GPU even though its predict service is CPU: a scan
  is tens of thousands of forwards, not one. Batch sizes are `ISM_BATCH` in
  `app/ism/engine.py`, overridable per deploy with `ISM_BATCH_SIZE`.

This does not touch `index.py`; the current deployment keeps serving until the new
service is verified and the frontend is switched.

## Known-unverified

Both are surfaced in `meta`, not buried here.

- **shorkie `species_index: 109`** (`species_index_verified: false`). Shorkie's input
  carries a one-hot species channel. The equivalence result cannot validate the index,
  because the reference TensorFlow predictions were generated with the same hardcoded
  value. The paper's own LM inference script uses 9 for the same assembly and the same
  165-species list. Predicting with the wrong index yields output that is still
  structured and still plausible (109 vs 9: Pearson 0.357).
- **yorzoi `seq_length=4992`, `crop_bp=996`** (`geometry_verified: false`). Inherited
  from the deployed service. Yorzoi's own dataset code describes a nominal 5000 bp
  window with 1000 bp of context, but 5000 cannot pass through the network, so the
  served geometry is not the nominal one and the crop may carry a 4 bp offset. There is
  no golden data for yorzoi here; capturing production output as a fixture before any
  cutover is the way to pin it.
