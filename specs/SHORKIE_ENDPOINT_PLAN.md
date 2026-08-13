# Shorkie inference endpoint — implementation plan

**Target:** `/Users/tds122/Documents/yorzoi-app` (branch `main`)
**Model source:** `github.com/tdsone/shorkie-pytorch` @ `c9a15f3`

Claims marked **[VERIFIED]** were checked against live artifacts (files executed, URLs fetched, wheels
built). Claims marked **[EST]** are estimates. **UNKNOWN** means exactly that, and every UNKNOWN below
names the command that resolves it.

---

## 1. Summary

Add a **second Modal ASGI function** — `shorkie_app` — to the existing `modal.App("yorzoi-app")`, with its
own image, GPU, and concurrency budget. It serves `POST /predict`, `GET /tracks`, `GET /model`, `GET /health`
at `https://tom-ellis-lab--yorzoi-app-shorkie-app.modal.run`. The existing `fastapi_app` function is not
renamed and not modified, so `frontend/src/App.jsx:31` keeps working unchanged.

Three things drive the design:

**(a) Track selection is the product.** One prediction is 896 bins × 5215 tracks = 4.67 M floats. Serialized
the way `/generate` does it that is 93 MB of JSON, 41.5 MB gzipped, and **10 s of gzip CPU per sequence**
(measured on real fold-0 output). The endpoint therefore defaults to a 4-row assay-profile preset (~1–4 KB),
caps JSON at 64 tracks, and offers `mode="aggregate"` (all 5215 tracks reduced to one scalar each, 28 KB
gzipped) and `format=npz` (all 5215 profiles as fp32 binary) as the escape hatches.

**(b) fp32, TF32 explicitly disabled, no autocast.** The entire value of `shorkie-pytorch` is
`Pearson R = 1.00000000` against the TF reference. bf16 costs ~1.1 % median relative error
(**[VERIFIED]** — reproduced independently: R = 0.99990285 both times) on a 19.7 GFLOP model whose actual
bottleneck is response serialization. `torch.autocast(device_type="cuda")` also defaults to **fp16**, not
bf16 — note that `backend/index.py:164` is therefore running yorzoi in fp16 today, probably unintentionally.

**(c) Ship against a real correctness oracle.** `shorkie-pytorch/data/predictions_f0_chunk*.h5` holds 1000
golden TF-model outputs, so the deployment can be validated end-to-end against the original TensorFlow
model rather than "it returns numbers." Two caveats that change how the test must be written: those files
are **gitignored and local-only** (16.8 GB, `git ls-files data/` returns only `shorkie_params.json`
**[VERIFIED]**), and **chunks 000–004 are synthetic random DNA** — only chunks 005–009 are genomic
(`modal_extract_weights.py:42-43,63,78`, `N_RANDOM=500` emitted first). Real genomic activations run ~12×
larger in mean and reach max 122 vs 18 for random, which is exactly the regime where a precision bug shows
up. Test against chunk009, not chunk000.

**Two open questions must close before launch,** both detailed in §6. The species-index one is a risk to
your `shorkie-pytorch` repo generally, not just to this endpoint.

---

## 2. Key design decisions

1. **Second `@app.function` + `@modal.asgi_app()` in the same `modal.App`, not a new route on `fastapi_app`.**
   *Rejected:* adding `POST /shorkie/predict` to the existing FastAPI app.
   *Why:* `image` and `gpu` are per-function in Modal, so a second function buys full image/hardware/
   autoscaling isolation from one `modal deploy`. Sharing would put both models in every container's GPU
   and let a Shorkie OOM take the live `/generate` down.
   *Caveat the isolation does not cover:* one `modal.App` means **one deployment unit** — `modal deploy`
   redeploys both functions and drains yorzoi's warm containers. That is a cold start for the next
   yorzoi.eu visitor after every Shorkie iteration, and a Shorkie rollback is an app-level rollback. Use
   `modal serve` for iteration and batch your `modal deploy`s.

2. **Do not rename `fastapi_app` or the app.** The deployed URL is `<workspace>--<app>-<function>.modal.run`
   and `frontend/src/App.jsx:31` hardcodes it. Renaming is a silent production break for zero benefit.

3. **fp32, `allow_tf32 = False`, no autocast.** TF32 defaults to `True` for cuDNN convs (**[VERIFIED]**),
   giving ~10 mantissa bits across 21 `Conv1d` layers — enough to push the deployment off the validated
   fp32 reference at ~1e-3. Keep `SHORKIE_ALLOW_TF32=1` as an escape hatch once §7.2 has measured the cost.
   (Note: `torch.backends.cuda.matmul.allow_tf32` already defaults to `False`, so that half of the line is
   a harmless no-op kept for explicitness.)

4. **Track selection with a small, meaningful default — not required, not "all".**
   *Rejected:* mandatory `tracks` (breaks `curl -d '{"sequences":[...]}'` on first contact) and defaulting
   to all 5215 (a 41 MB / 10 s footgun).
   *Why:* a 4-row assay preset returns in ~1–4 KB and maps onto the existing frontend panel layout.

5. **`mode="aggregate"` is first-class and is what to steer people toward.** All 5215 tracks reduced over
   the predicted window is **28 KB gzipped** (measured) vs 41.5 MB for the matrix — 1483× smaller — and it
   is literally what "sequence to expression" means. Ref-vs-alt variant effect becomes one request.

6. **Return 16 bp bins, never per-bp.** *Rejected:* expanding to bp like yorzoi's
   `untransform_then_unbin(resolution=10)`. Shorkie has no unbinning function and no sub-16 bp resolution;
   expansion is 16× the bytes for zero information. The client does
   `values[t][Math.floor((i - bin_start_bp) / 16)]`.

7. **Response is an index-aligned `predictions[]` array with a caller-supplied `id`, never a dict keyed by
   the sequence string.** *Why:* yorzoi's `predictions_dict[original_seq]` silently collapses duplicate
   sequences (a WT-vs-WT control vanishes and the client cannot detect it), has no defined order, and
   echoing a 16 kb key costs ~5 KB gzipped — roughly a quarter of a typical response — to return data the
   client just sent.

8. **`values` is track-major `[n_tracks][n_bins]`,** transposed from the model's native `(bins, tracks)`.
   The position axis is 4× smoother than the track axis (measured), so track-major gzips 3.6–16 % better,
   and it matches both `App.jsx`'s `coverage[trackIdx]` indexing and yorzoi's convention.

9. **Fix `shorkie-pytorch/pyproject.toml` and install the package; do not vendor `shorkie.py`.**
   The current file fails twice. `requires-python = ">=3.13"` blocks the 3.12 image, and with no
   `[build-system]` and no `py-modules`, setuptools' flat-layout discovery finds `data/` and produces a
   wheel containing **only `dist-info/`**, with `top_level.txt = "data"` — so `pip install` *succeeds* and
   `import shorkie` *fails*. **[VERIFIED — `uv build --wheel` reproduced, namelist is dist-info only.]**
   The 4-line fix is required regardless and makes the standalone repo strictly better.

10. **Weights: convert all 8 folds to safetensors once, publish to a public HF repo, bake into the image at
    build, pinned to a commit SHA.** *Rejected:* `wget` from GCS at build (the bucket layout **already moved
    once** — `seqnn-share/shorkie/f0/model_best.h5` returns **404**, the live path is
    `seqnn-share/shorkie_models/shorkie/f0/model_best.h5` **[VERIFIED]**, and your README still points at the
    dead one); runtime download; Modal Volume.

11. **Track metadata is a committed repo file** added with `add_local_file(..., copy=True)`. It is a 781 KB
    immutable constant frozen by the 5215-unit head. A boot-time assert
    `len(tracks) == 5215 == model.head.out_features` turns the worst failure mode — silent off-by-N track
    mislabeling — into a loud boot failure.

12. **`async def` route + `asyncio.to_thread` for the GPU block; `@modal.concurrent(max_inputs=4, target_inputs=2)`
    with a `Semaphore(1)` around the GPU section.** *Rejected:* a sync `def` handler (Modal terminates the
    **entire container** on a single input cancellation for synchronous functions — one user hitting Escape
    would 5xx everyone else on that container) and copying `max_inputs=100` from `fastapi_app` (at
    ~0.3–0.5 GB host RAM per in-flight request, 100 concurrent is a host OOM before the GPU notices).

13. **Single fold f0; no RC, no shift, no ensembling in v1.** Each multiplies latency and none is validated
    on this port. Bake only f0 (457 MB for all eight is dead weight when nothing uses them).

14. **N-padding for short sequences, with the endpoint framed as *differential*.** All-zero is the correct
    in-distribution encoding of N (matches `baskerville.dna.dna_1hot(n_uniform=False)`; `_expand_species_encoding`
    at `shorkie.py:577-590` copies channels 0–3 verbatim and leaves the BERT channel 4 at zero **[VERIFIED]**).
    But N-padding measurably corrupts the prediction *over the user's real sequence*: a 1 kb core N-padded
    to 16384 vs the same core in random ACGT gives near-zero correlation (two independent measurements:
    R = −0.071 and R = +0.106 on different sequences — same conclusion, sequence-dependent magnitude) with
    3–5× inflated absolute values. So: warn with `pad_fraction`, and document that A-vs-B at identical
    padding is valid while absolute magnitudes for short inputs are not.

15. **`window="auto"`: a 16384 bp input takes the exact single-pass path.** The tiled path applied to
    exactly 16384 bp produces origins `[0, 2048]` — *neither* tile is clean (one has 1024 bp of N on the
    left, the other 1024 bp on the right) and their overlap averages two N-contaminated predictions, when a
    single exact pass exists and reproduces the validated numbers. A user pasting exactly 16384 bp is doing
    the most obvious thing possible and must not silently get the degraded path.

---

## 3. The API contract

Base URL: `https://tom-ellis-lab--yorzoi-app-shorkie-app.modal.run`. Paths are unprefixed (`/predict`, not
`/shorkie/predict`) — it is a dedicated subdomain.

### 3.1 Constants

```python
SEQ_LENGTH   = 16384   # model input window; nothing else is valid (§6.3)
N_BINS       = 896     # 1024 positions - 2 * Cropping1D(64)
BIN_SIZE     = 16
CROP_BP      = 1024    # 64 bins * 16 bp, cropped from EACH edge
PREDICTED_BP = 14336   # 896 * 16
N_TRACKS     = 5215
TILE_STEP    = 7168    # fixed in v1

MAX_SEQUENCES     = 8          # matches train.batch_size
MIN_SEQ_LEN       = 16         # one bin
MAX_SEQ_LEN       = 131_072
MAX_TOTAL_BP      = 262_144
MAX_TRACKS_JSON   = 64         # 64 * 896 * 4 B = 229 KB/seq raw
MAX_VALUES_JSON   = 1_000_000
MAX_VALUES_BINARY = 25_000_000 # ~100 MB fp32 body; see the 150 s note in §3.6
```

### 3.2 `POST /predict`

Query params: `?format=json|npz` (default `json`), `?bench=true` (runs encode + forward + postprocess,
returns `meta` only — same semantics as `/generate?bench=true`).

```jsonc
{
  "sequences": ["ACGT..."],        // or [{"id":"wt","seq":"ACGT..."}]; 1..8 items, 16..131072 bp each,
                                   // 262144 bp total. Uppercased, whitespace stripped, FASTA '>' lines
                                   // dropped. Alphabet ACGTN after uppercasing; N -> all-zero column.

  "tracks":       [1148, 1149],    // int indices 0..5214 OR identifier strings (not mixed)
  "track_groups": ["rnaseq_idea"], // chip_exo | chip_mnase | rnaseq_idea | rnaseq_strains
  "preset":       "assay_profiles",// assay_profiles (default) | rnaseq_profile | all
                                   // union of the three selectors, deduped, ascending index
                                   // "all" is rejected in JSON matrix mode

  "mode":   "matrix",              // matrix (default) | aggregate  (aggregate = sum over predicted bins)
  "window": "auto"                 // auto (default) | full | native
}
```

`window`: `auto` uses `native` when `len(seq) == 16384`, else `full`. `native` is one unpadded pass
returning the 896 bins over `[1024, 15360)` — the paper-reproduction / equivalence-test path — and requires
exactly 16384 bp. `full` always tiles and pads to cover the whole input.

**Response — `mode="matrix"`, `format=json`:**

```jsonc
{
  "meta": {
    "api_version": 1,
    "model": "shorkie", "fold": "f0",
    "weights_revision": "<HF commit SHA>", "checkpoint_sha256": "<computed at boot>",
    "seq_length_bp": 16384, "predicted_bp": 14336, "crop_bp": 1024,
    "bin_size_bp": 16, "n_bins_full": 896, "n_tracks_total": 5215,
    "layout": "track_major",                 // values[t][b]
    "value_space": "binned_coverage_sum",    // 16 bp sum in each track's native units
    "untransformed": false,                  // no transform exists to invert; see §6.2
    "depth_normalized": false,               // magnitudes are NOT comparable across tracks
    "species_index": 109,                    // see §6.1 — must be resolved before launch
    "precision": "sig3",
    "timing_ms": {"encode": 4, "forward": 31, "postprocess": 2, "serialize": 3}
  },
  "tracks": [
    {"i": 1148, "id": "ARG80_T0_S757", "assay": "rnaseq_idea",
     "description": "ARG80-T0 RNAseq, coverage"}
  ],
  "predictions": [
    {
      "id": "wt", "length": 1000,
      "window": "full", "tiles": 1,
      "bin_start_bp": 0, "n_bins": 63,      // bin k covers [bin_start_bp+16k, +16)
      "last_bin_partial": true, "last_bin_bp": 8,
      "pad_fraction": 0.939, "warnings": ["very_low_context"],
      "values": [[0.128, 0.131, "…63 floats…"]]
    }
  ]
}
```

JSON floats are rounded to **3 significant digits**, not 3 decimals. Rounding to decimals is a bug, not a
knob: at 3 decimals the max relative error is **1.0** — 0.1 % of values round to exactly 0.0 and the bottom
decile is destroyed. Three significant digits is both smaller on the wire (7.88 vs 10.73 MB gzipped for a
full matrix) and 200× more accurate.

```python
def sig(x, n=3):                       # x is strictly positive (softplus head)
    x = np.maximum(x, 1e-12)
    m = 10.0 ** (n - 1 - np.floor(np.log10(x)))
    return np.round(x * m) / m
```

**Response — `mode="aggregate"`:** `values`/`n_bins`/`bin_start_bp` are replaced by one scalar per selected
track (`"agg": "sum"`, summed over all predicted bins), plus `n_bins_summed`.

**Response — `format=npz`:** `Content-Type: application/octet-stream`, `Content-Encoding: identity`. Gzip
on float bytes is 1.10–1.16× for 0.2–2.2 s of CPU — never do it. Members:

```
predictions    float32  (n_seq, n_tracks, n_bins)   track-major
track_indices  int32    (n_tracks,)
track_ids      <U…      (n_tracks,)
bin_start_bp   int32    (n_seq,)
sequence_ids   <U…      (n_seq,)
meta_json      bytes    # the same meta object as the JSON response
```

`meta_json` lives **inside** the archive, not in a response header — a browser cannot read a custom header
without `expose_headers`, and there is no reason to add that surface.

### 3.3 Auxiliary endpoints

| Endpoint | Purpose |
|---|---|
| `GET /tracks` | Full 5215-entry catalog, ~357 KB raw / **51 KB gzipped** (measured). `Cache-Control: public, max-age=86400, immutable`. |
| `GET /model` | Constants, `weights_revision`, `checkpoint_sha256`, `species_index`, and `limits` — including **worked examples** of the real envelope, not just raw constants (a 131 kb sequence is legal alone but two of them exceed `MAX_VALUES_JSON`). |
| `GET /health` | `{"ok", "model_loaded", "cuda", "eval_mode"}`; 503 if `model.training`. |

⚠️ **`/health` and `/tracks` boot a GPU container.** Do not wire `/health` to an uptime monitor — it will
keep an L40S warm 24/7 (~$47/day). Ship `shorkie_tracks.json` as a **frontend build asset** so the browser
never cold-starts a GPU just to populate a track picker.

### 3.4 Error contract

Every error is `{"error": {"code", "message", "field", "request_id"}}`. Structural validation lives in
pydantic `field_validator`s so FastAPI emits **422 automatically** — note that the current `/generate`
raises a bare `ValueError` *inside the handler body* (`backend/index.py:84`, called from `:110`), which
becomes an **HTTP 500 with a stack trace containing the user's sequence** for what is plainly a client error.

Ship these six codes; let pydantic's default 422 body carry the rest: `invalid_alphabet` (with the offending
character and its 0-based position), `too_many_sequences`, `sequence_too_long`, `unknown_track`,
`response_budget_exceeded` (with `requested_values` and `limit`), `model_loading` (503 + `Retry-After: 10`).

Budget overrun is **422, not 413** — the request body is small and well-formed; it is the *result* that is
unsatisfiable. Never silently truncate.

**No bare `assert` in the request path.** `assert` raises an unhandled 500 and `python -O` deletes it
entirely — which is exactly the failure mode this section criticises `/generate` for. Raise a typed internal
error with a `code` and log it. Boot-time asserts (§4.4) are fine: they run once, before serving.

**Logging contract** (Modal logs are the only debugging surface): one line per request with `request_id`,
sequence count, total bp, tile count, track count, `mode`, `format`, and the `timing_ms` breakdown.

### 3.5 Examples

```bash
# Default: 4 assay profiles over a 1 kb promoter. ~1 KB response.
curl -sS -X POST https://tom-ellis-lab--yorzoi-app-shorkie-app.modal.run/predict \
  -H 'content-type: application/json' \
  -d '{"sequences":["ATGACCCAGT...TTTGA"]}' \
  | jq '.predictions[0] | {id, n_bins, pad_fraction, warnings}'
```

```bash
# Variant effect: all 5215 tracks summed over the window, WT vs mutant. ~28 KB each.
curl -sS -X POST .../predict -H 'content-type: application/json' -d '{
  "sequences":[{"id":"wt","seq":"…16384bp…"},{"id":"alt","seq":"…16384bp…"}],
  "mode":"aggregate","preset":"all"
}' > ve.json
```

```python
# …and turning that into a ranked variant effect (this is the flagship use case):
import json, numpy as np
d = json.load(open("ve.json"))
wt, alt = (np.array(p["values"]) for p in d["predictions"])   # (5215,) each
lfc = np.log2((alt + 1e-3) / (wt + 1e-3))
for i in np.argsort(-np.abs(lfc))[:10]:
    print(f"{lfc[i]:+.2f}  {d['tracks'][i]['id']}  {d['tracks'][i]['description']}")
```

```bash
# All 5215 profiles as binary, for numpy users.
curl -sS -X POST '.../predict?format=npz' -H 'content-type: application/json' \
  -d '{"sequences":["…16384bp…"],"preset":"all","window":"native"}' -o out.npz
python -c "import numpy as np; print(np.load('out.npz')['predictions'].shape)"   # (1, 5215, 896)
```

### 3.6 The real SLA: Modal's 150-second request ceiling

Modal enforces a **150 s HTTP request timeout on web endpoints**, independent of the function `timeout`.
Past 150 s Modal issues a **303 redirect** to resume the stream — and a 303 converts POST→GET in browsers
and most HTTP clients, so a slow request does not simply take longer, it fails in a way the handler cannot
route around. (This also means `fastapi_app`'s `timeout=20*MINUTES` has never been reachable.)

Set `timeout=150` so a runaway request fails honestly as a timeout, and size `MAX_VALUES_BINARY` so the p99
body transfers well inside 150 s on a home connection. This is why the primary equivalence test in §7.2
runs over `modal run` rather than HTTP.

---

## 4. Packaging & deployment

**Canonical deploy CWD is `backend/`.** Every relative path below assumes it; `add_local_file` resolves
against the deploy-time CWD, so deploying from the repo root would break the build.

### 4.1 How the code reaches the container

`pip install` of the **fixed** package from a pinned GitHub tarball:

```python
.pip_install("shorkie-pytorch @ https://github.com/tdsone/shorkie-pytorch/archive/<POST-FIX-SHA>.tar.gz")
```

PyPI is **not** the v1 path: `pypi.org/pypi/shorkie-pytorch/json` → **404 [VERIFIED]**, the name is unclaimed,
and publishing needs an account plus token setup that buys nothing here. Do **not** `pip install git+https://…`
against current `main` — it fails on `requires-python` and then fails *silently* on the empty-wheel bug.
The upstream fix (§4.5) is a hard prerequisite, not a parallel task.

Your two new local modules must be added explicitly — Modal 1.0 removed automounting and auto-includes only
the module defining the App:

```python
.add_local_python_source("shorkie_core", "shorkie_tracks_api")
```

Without this, the image builds, `modal deploy` succeeds, and the container dies at start with
`ModuleNotFoundError`.

### 4.2 How the weights reach the container

One-time local conversion → **public** HF repo `tom-ellis-lab/shorkie` → `snapshot_download` at **image
build**, snapshotted into the layer. Runtime does zero network I/O.

The repo does not exist yet: `huggingface.co/api/models/tom-ellis-lab/shorkie` → **401 [VERIFIED]** (vs
`tom-ellis-lab/yorzoi` → 200). Create it public and verify anonymously **before** the image build ever runs,
or the build dies with a 401 that reads like an auth problem rather than a missing repo. If it must be
private, `run_function(..., secrets=[modal.Secret.from_name("huggingface-secret")])` is required.

**Pin `SHORKIE_REV` to a commit SHA from day one.** Modal hashes only the function source, its kwargs, and
referenced globals to decide whether to rebuild a `run_function` layer. With `SHORKIE_REV = "main"`,
re-uploading corrected weights changes nothing Modal hashes — the cached layer is reused indefinitely and
every container silently serves the old weights. The boot-time shape assert cannot catch this; the shapes
still match. Compute `checkpoint_sha256` at boot from the actual loaded file and surface it in `/model`.

### 4.3 The image chain

```python
SHORKIE_REPO = "tom-ellis-lab/shorkie"
SHORKIE_REV  = "<pinned commit SHA>"
WEIGHTS_DIR  = "/weights"

def _bake_shorkie_weights():
    from huggingface_hub import snapshot_download
    snapshot_download(SHORKIE_REPO, revision=SHORKIE_REV, local_dir=WEIGHTS_DIR)

shorkie_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")   # same tag as the yorzoi image
    .pip_install(
        "shorkie-pytorch @ https://github.com/tdsone/shorkie-pytorch/archive/<SHA>.tar.gz",
        "torch>=2.5", "numpy>=1.26", "h5py>=3.10", "safetensors>=0.4",
        "huggingface-hub>=0.24",            # build-time only
        "fastapi[standard]==0.115.*", "starlette==0.41.*", "orjson==3.10.*",
    )
    .run_function(_bake_shorkie_weights)     # no gpu= — CPU-only, don't pay for one
    .add_local_python_source("shorkie_core", "shorkie_tracks_api")
    .add_local_file(Path(__file__).parent / "shorkie_tracks.json", "/root/shorkie_tracks.json", copy=True)
    .env({"SHORKIE_WEIGHTS": WEIGHTS_DIR})
)
```

**`h5py` stays a hard runtime dependency.** `shorkie.py:3` is an unconditional module-level `import h5py`,
so `from shorkie import Shorkie` needs it whether or not you load safetensors. Either keep it (this plan) or
move the import inside `_load_tf_weights` upstream — but do not let the docs and the code disagree, because
the failure is an `ImportError` at container boot with a confusing traceback.

Pin FastAPI/Starlette/orjson. Modal caches a `pip_install` layer by its definition string, so unpinned deps
hold *until* something invalidates the layer (a builder-version bump, a `force_build`, an edit to the shared
`image` variable) — at which point a fresh Starlette resolves straight into production with no review.

### 4.4 The function

```python
@app.function(
    image=shorkie_image,
    gpu="L40S",                  # benchmark A10 (§8)
    cpu=2.0,
    memory=(4096, 16384),        # (request, limit) — a bare int is a REQUEST and caps nothing
    timeout=150,                 # matches Modal's web-endpoint ceiling (§3.6)
    scaledown_window=120,
    max_containers=10,           # hard spend ceiling
)
@modal.concurrent(max_inputs=4, target_inputs=2)
@modal.asgi_app()
def shorkie_app():
    ...
```

Verified against modal 1.4.2 **[VERIFIED]**: `App.function(gpu, cpu, memory, timeout, scaledown_window,
image, min_containers, max_containers, buffer_containers, …)`, `modal.concurrent(max_inputs, target_inputs)`,
`modal.asgi_app(label, custom_domains, requires_proxy_auth)`, `Image.run_function/env/add_local_file(copy=)/
add_local_python_source`. Leave `min_containers` at 0 — an always-warm L40S is ~$47/day.

Inside `shorkie_app()`, **copy the middleware stack** — a second FastAPI app inherits none of `fastapi_app`'s
(`backend/index.py:57-65`). Without it the frontend change is dead on arrival from `yorzoi.eu` and every
gzipped size in §8 is off by 2–7× because nothing compresses:

```python
app = FastAPI(title="Shorkie API", default_response_class=ORJSONResponse)
app.add_middleware(CORSMiddleware, allow_origins=[
    "https://yorzoi.eu", "https://www.yorzoi.eu",
    "http://localhost:5173", "http://127.0.0.1:5173",
], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)   # npz responses set Content-Encoding: identity
```

Container start:

```python
torch.backends.cuda.matmul.allow_tf32 = os.getenv("SHORKIE_ALLOW_TF32") == "1"
torch.backends.cudnn.allow_tf32       = os.getenv("SHORKIE_ALLOW_TF32") == "1"

cfg   = json.load(open(f"{WEIGHTS_DIR}/params.json"))["model"]
model = Shorkie(cfg)
model.load_state_dict(load_file(f"{WEIGHTS_DIR}/shorkie_f0.safetensors"))
model.eval()                       # MANDATORY — see §6.5
model.to("cuda:0")
assert not model.training
assert model.head.out_features == N_TRACKS == len(TRACKS)
```

**Resulting URLs.** New: `…--yorzoi-app-shorkie-app.modal.run`. Unchanged:
`…--yorzoi-app-fastapi-app.modal.run`.

### 4.5 Upstream `shorkie-pytorch` checklist — one PR, merge before touching the backend

- [ ] **`pyproject.toml`**: add `[build-system] requires = ["setuptools>=68"]`,
      `build-backend = "setuptools.build_meta"`; add `[tool.setuptools] py-modules = ["shorkie"]`
      *(the line that fixes the silent empty-wheel bug)*; `requires-python = ">=3.10"`; runtime deps →
      `["h5py>=3.10", "numpy>=1.24", "torch>=2.2"]`; move `tensorflow`/`matplotlib`/`ipykernel`/`pytest`/
      `modal` to `[project.optional-dependencies]`; bump to `0.2.0`. Then `uv lock` — the committed
      `uv.lock` was resolved under `.python-version = 3.13` and will not satisfy the new floor.
- [ ] **Verify the install for real**, not just the build:
      `uv venv --python 3.12 /tmp/v && /tmp/v/bin/pip install "shorkie-pytorch @ https://github.com/tdsone/shorkie-pytorch/archive/<newSHA>.tar.gz" && /tmp/v/bin/python -c "import shorkie; print(shorkie.Shorkie)"`
- [ ] **`from_tf_checkpoint` must return an eval-mode model.** It currently returns `model.training == True`.
      One train-mode forward gives R ≈ −0.03 against the correct output *and* mutates all 40 BatchNorm
      `running_mean`/`running_var` buffers **[VERIFIED, reproduced twice]**. The persistent damage from a
      single event is R = 0.983 (momentum 0.1), not total garbage — but it is silent and cumulative.
- [ ] **Add `Shorkie.from_safetensors(config, path)`** (5 lines) so the backend doesn't hand-roll loading.
- [ ] **Export a vectorised `dna_1hot(seq) -> (4, L) float32` LUT encoder.** `demo.ipynb`'s version raises
      `KeyError` on `N` and is a 16384-iteration Python loop per sequence.
- [ ] **Fix the three dead GCS URLs in `README.md`** (and `modal_extract_weights.py:35-37`,
      `key_prefix="shorkie/"`): `seqnn-share/shorkie/f{n}/` → `seqnn-share/shorkie_models/shorkie/f{n}/`.
      Old path 404, new path 206 **[VERIFIED]**.
- [ ] **Publish an equivalence oracle subset** — 8 sequences from a *genomic* chunk (005–009) as fp32,
      ~152 MB, to the HF weights repo. Today the entire oracle is gitignored (`data/*.h5`) and exists only
      on your laptop, so §7.2 cannot run on any other machine.
- [ ] **`git pull` the `shorkie-paper` clone.** Local is `ed28a96` **[VERIFIED]**; upstream `c6003ce` now
      ships `data/manifest.json`, `scripts/01_data_build/`, and `minimal_example/` — the authoritative spec
      for §6.2 and possibly the resolution for §6.1.

---

## 5. Implementation steps

Steps marked **∥** are independent.

**Step −1 — Harden the live `/generate` first (own commits, before any Shorkie work).**
These are pre-existing and independent; doing them first means a later regression is unambiguously Shorkie's.

  a. **`GenerateRequest.sequences` is unbounded.** No count cap, no length cap. A single 1 Mb sequence
     produces ~996 chunks stacked into **one** batch tensor (`index.py:160-162`) ≈ 1.9 GB on the GPU, then
     `.tolist()` materializes ~484 M Python floats ≈ **15 GB of host RAM**. One curl. Add
     `conlist(str, max_length=8)` and a per-sequence length cap. *This is the highest-value change to the
     currently-deployed service in this document.*
  b. **Silent zeros for `3000 < L ≤ 4992`** (`index.py:177-188`). The branch tests `original_len > 4992`
     (the *window*) but the model predicts only the centre 3000 bp; full coverage needs `left_pad ≥ 996`,
     i.e. `L ≤ 3000`. At L=4000 that is **1000 bp (25 %) of hard zeros returned as predictions**; at
     L=4992, **39.9 %** **[VERIFIED, arithmetic reproduced]**. Branch on `prediction_length`, not
     `window_size`.
  c. Add `torch.inference_mode()` at `index.py:164` (~3× memory for nothing today) and pin
     `fastapi[standard]` / `orjson`.

**Step 0 ∥ — Track metadata.** *(15 min)*
```bash
curl -L -o backend/shorkie_targets.txt \
  https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/targets.txt
```
781,531 bytes, md5 `58fc7ccf62ae5e9f29c444e85fbf9771` **[VERIFIED — downloaded and parsed independently by
two agents]**. Columns: `index, identifier, file, sum_stat, clip_soft, description, group`. Write
`backend/build_shorkie_tracks.py` (stdlib `csv`, no pandas) emitting `backend/shorkie_tracks.json` with
`index`, `identifier`, `description`, `assay`. **Assign `assay` by index range and assert the description
regex agrees** — do not derive it from the `group` column, which mislabels all 1128 ChIP-exo rows as
`Other`. Commit both files plus a `SOURCE.md` recording URL/size/md5/date.

| assay | index range | n |
|---|---|---|
| `chip_exo` | 0–1127 | 1128 |
| `chip_mnase` | 1128–1147 | 20 |
| `rnaseq_idea` | 1148–4200 | 3053 |
| `rnaseq_strains` | 4201–5214 | 1014 |

All four blocks contiguous, identifiers unique, and **no row anywhere mentions strand/plus/minus/fwd/rev** —
**[VERIFIED twice, independently]**.

**Step 1 — Upstream PR.** Execute §4.5. Blocks Steps 2 and 5.

**Step 2 ∥ — Weight conversion + public HF repo.** `shorkie-pytorch/scripts/publish_to_hf.py`: for each
fold, download from `…/shorkie_models/shorkie/f{n}/model_best.h5`, `from_tf_checkpoint`, `save_file` as
safetensors. Upload f0 + `params.json` + the oracle subset to `tom-ellis-lab/shorkie`, public.
*Verify:* load the safetensors into a fresh `Shorkie(cfg)` and assert `max|Δ|` vs `from_tf_checkpoint` is
exactly `0.0`; assert `sum(p.numel() for p in m.parameters()) == 14_243_775` **[VERIFIED — the model is
14.2 M params, 57 MB fp32, not the ~100 M originally assumed]**; then
`curl -o /dev/null -w '%{http_code}' https://huggingface.co/tom-ellis-lab/shorkie/resolve/main/shorkie_f0.safetensors`
returns 200 **anonymously**.

**Step 3 ∥ — Pure-function core (numpy only, no torch, no Modal).** `backend/shorkie_core.py`. Everything
here is unit-testable on a laptop in <1 s — which requires adding dev deps to `backend/pyproject.toml`,
which today declares exactly one dependency (`modal>=1.4.2`):

```toml
[dependency-groups]
dev = ["pytest>=8", "numpy>=1.26", "h5py>=3.10", "requests>=2.32"]
```

*Encoder:*
```python
_LUT = np.zeros((256, 4), dtype=np.float32)
for k, b in enumerate(b"ACGT"): _LUT[b, k] = 1.0

def dna_1hot(seq: str) -> np.ndarray:      # (4, L); anything not ACGT -> all-zero column (= N)
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)].T
```
**Uppercase and validate before this call.** The LUT maps lowercase to all-zero, i.e. silently to N — and
soft-masked repeats from SGD/UCSC arrive lowercase and are real bases. Silent coercion here is the exact
class of bug §6.4 is about.

*Coordinate math.* One concept: **`origin` = the original-sequence bp coordinate of the first *predicted* bp
of a tile**, always `≡ 0 (mod 16)`. This replaces yorzoi's `left_pad`/`prediction_offset`/
`start_pos_in_original` triple and makes the short, tiled, and native paths one code path.

```
window covers original bp   [origin - 1024, origin + 15360)     (exactly 16384 bp)
tile bin j (0<=j<896)       [origin + 16j, origin + 16j + 16)
global bin index            G = origin // 16 + j
```

```python
def floor16(x): return (x // 16) * 16
def ceil16(x):  return -((-x) // 16) * 16

def plan_origins(L, window="auto", step=TILE_STEP):
    if window == "native" or (window == "auto" and L == SEQ_LENGTH):
        if L != SEQ_LENGTH: raise BadRequest("native_requires_16384")
        return [CROP_BP], "native"                        # one exact, unpadded pass
    if L <= PREDICTED_BP:                                 # threshold is 14336, NOT 16384
        return [-floor16((PREDICTED_BP - L) // 2)], "full"
    origins = list(range(0, L - PREDICTED_BP + 1, step))
    tail = ceil16(L - PREDICTED_BP)                       # round UP, else a <=15 bp silent 3' gap
    if origins[-1] != tail: origins.append(tail)
    return origins, "full"

def merge(tile_preds, origins, clip_to=None):
    """tile_preds: list of (896, n_tracks). Returns (values, g_first) on the global 16 bp bin grid."""
    g = [o // 16 for o in origins]
    g_first, g_last = min(g), max(g) + N_BINS
    acc = np.zeros((g_last - g_first, tile_preds[0].shape[1]), np.float32)
    cnt = np.zeros((g_last - g_first, 1), np.float32)
    for pred, gi in zip(tile_preds, g):
        s = gi - g_first
        acc[s:s + N_BINS] += pred; cnt[s:s + N_BINS] += 1
    if not (cnt > 0).all(): raise InternalError("tiling_gap")   # never a bare assert (§3.4)
    out = acc / cnt
    if clip_to is not None:                                # (g_lo, g_hi) for window="full"
        g_lo, g_hi = clip_to
        out, g_first = out[g_lo - g_first : g_hi - g_first], g_lo
    return out, g_first
```

**`merge` must derive its bin span from the origins, not from `L`.** Deriving `n_bins = ceil(L/16)` is
correct for `full` and **wrong for `native`**: a native pass at `origin=1024` covers global bins 64…959, so
128 of the 1024 bins implied by `L` are never written and the contiguity check fires. `native` returns 896
bins at `bin_start_bp = 1024`; `full` passes `clip_to = (0, ceil(L/16))`. *(This defect was caught by
executing the draft's own pseudocode — it would have blocked the load-bearing equivalence test.)*

Two deliberate departures from the yorzoi reassembler: a raised error instead of
`overlap_counter[overlap_counter == 0] = 1` (`index.py:227`) — that clamp turns a coverage gap into a silent
block of zeros in the response — and fp32 accumulation regardless of forward dtype.

*Verify Step 3:* `uv run pytest` from `backend/`. Exhaustive coverage sweep over
`L ∈ [16,200] ∪ [14300,14400] ∪ [16300,16500] ∪ {300, 1000, 3000, 15000, 20000, 100000, 131072}` ×
`step ∈ {16, 64, 7168, 8192, 14336}`: zero gaps, all origins `≡ 0 mod 16`. **This sweep has already been run
against the pseudocode above and passes with 0 gaps and 0 assertion trips [VERIFIED]**, including the
general proof that with `step` a multiple of 16 the tail append can never open a gap. Pin `L = 16, 15,
14336, 14337, 16384, 16385`; assert `plan_origins(16384, "full") == [0, 2048]` and
`plan_origins(16384, "auto") == ([1024], "native")`.

**Step 4 ∥ (needs 0, 3) — Track resolution + presets.** `backend/shorkie_tracks_api.py`: load the JSON once,
build `identifier → index`, resolve the union of `tracks`/`track_groups`/`preset`, enforce caps, return the
`tracks[]` block. *Verify:* unit tests for unknown identifier, mixed int/str → 422, `preset="all"` + JSON
matrix → 422, cap overrun → 422 with `requested_values`.

**Step 5 — the Modal function, in three verifiable slices.** Do not do this as one commit; nothing in it is
testable until all of it works otherwise.

  - **5a — CPU-only stub.** `shorkie_app` with the image chain, middleware, and `/health` returning a
    constant. No model. *Verifies:* image build, the tarball install, `add_local_python_source`,
    `add_local_file` path resolution, CORS from `localhost:5173`, and the URL. Most of §4's failure modes
    live here and this is where they should surface.
  - **5b — model load.** Add the container-start block and the boot asserts; `/health` reports real state.
  - **5c — `/predict`.** Pydantic models, validators, budget checks (**before** any allocation), the GPU
    section, `merge`, `meta`.

```python
GPU_LOCK = threading.Semaphore(1)

@app.post("/predict")
async def predict(req: ShorkieRequest, format: str = "json", bench: bool = False):
    plans = [(item, *plan_origins(len(item.seq), req.window)) for item in req.sequences]
    check_budgets(plans, sel)                       # raises 422 before anything is allocated
    return await asyncio.to_thread(_run, plans, sel, format, bench)

def _run(plans, sel, format, bench):
    with GPU_LOCK:
        for chunk in batched(tiles, 8):
            x = torch.from_numpy(np.stack([...])).to("cuda:0", non_blocking=True)
            assert x.shape[1:] == (4, SEQ_LENGTH)   # §6.3 — the model has no shape validation
            with torch.inference_mode():            # no autocast
                y = model(x)                        # (b, 896, 5215)
            y = y[:, :, sel_idx].float().contiguous()   # slice tracks ON GPU, before any copy
```

The **GPU-side track slice** is the difference between a viable and a non-viable endpoint: it turns an
18.7 MB/tile device→host copy into 0.23 MB for a 64-track selection. Preset rows are likewise reduced on
the GPU, not after the copy.

*Verify:* `modal serve backend/index.py` → `/health`, then a 1 kb `/predict`. Then `modal deploy` and
confirm `…-fastapi-app.modal.run/generate` still returns the stored golden response.

**Step 6 (needs 2, 5b) — GPU equivalence via `modal run`.** §7.2. This is the first CUDA validation this
port has ever had, and it must happen before the endpoint is announced.

**Step 7 ∥ (needs 5c) — `format=npz`.** ~20 lines: `np.savez` into a `BytesIO`,
`Content-Encoding: identity`, explicit `Content-Length`.

**Step 8 ∥ (needs 6) — Biological validation.** §7.3, including the species-index sweep. Run before
announcing.

**Step 9 ∥ — README.** `README.md` is wrong three ways today and the plan is incomplete without fixing it:
the documented URL `…-yorzoi-app-model-generate.modal.run` returns **404** while the live one is
`…-yorzoi-app-fastapi-app.modal.run/generate` (**200 [VERIFIED]**); "each DNA seq needs to be 4992 bp long!"
has been false since padding/tiling shipped; and the documented response `{prediction: [[float]]}` is not
the dict-keyed-by-sequence shape actually returned. Fix all three, document the Shorkie endpoint with the
§3.5 examples, and state plainly what the numbers mean (§6.2).

**Step 10 ∥ (needs 5c) — Frontend.** Separate PR. Mostly deletion: `Object.keys(data)[0]` →
`data.predictions[0]`; the hardcoded `track2idx` and the `coverage[trackIdx + 81]` strand offset both go
away (Shorkie has no strand axis, §6.4). Bin→base becomes a lookup inside the existing `coverageToChrom`:
`values[t][Math.floor((i - bin_start_bp) / 16)] ?? 0`. `bin_start_bp` must come from the server — it is the
one field the client cannot compute without reimplementing the padding and cropping arithmetic. Also:
surface `error.code` instead of `catch → null` (`App.jsx:56-59`), add `MIN_SEQ_LEN`/`MAX_SEQ_LEN` guards
next to the `/^[acgtn]+$/i` test at `App.jsx:104`, and move both Modal base URLs into one `VITE_*` config
module.

*Product decision this forces:* `App.jsx` currently renders yorzoi. Decide whether Shorkie replaces it, gets
a second route, or sits behind a model switcher — and what happens to `backend/track_annotation.json`. The
backend ships and is curl-usable regardless; this is not a blocker.

---

## 6. Correctness risks & open questions

Ordered by how badly each would produce **plausible but wrong** numbers.

### 6.1 Species index 109 — **UNKNOWN, and a launch blocker**

`shorkie.py:17` hardcodes `R64_SPECIES_INDEX = 109` → input channel 114. **The `R = 1.00000000` equivalence
result cannot validate this**, because `modal_extract_weights.py:96`, which generates the TF golden
predictions, hardcodes *the same 109*. Both sides make the identical assumption; the result proves the port,
not the index.

Meanwhile the paper's own LM inference script uses a different value for the same assembly **[VERIFIED by
direct read]**:

```
shorkie-paper/analysis/shorkie_lm/lm_SMT3_viz/3_inference_smt3_unmasked.py:16
    SPECIES_INDEX = 9  # GCA_000146045_2 index (0-based) from sorted list
:17 NUM_SPECIES = 165
```

`GCA_000146045.2` **is** S. cerevisiae S288C, and this is the same 165-species list and the same 16384 bp
window as the LM the supervised model was fine-tuned from. (A second paper script,
`umap_cluster_promoter/1_predict_seqs_LM.py:130-131`, sets **no** species channel at all — so the paper's
own scripts are not mutually consistent, which is why the empirical sweep below matters more than another
round of code reading.)

The sensitivity is severe, and reproduced independently by two agents to three decimal places:

| comparison | Pearson R | mean abs Δ |
|---|---|---|
| idx 109 vs idx 9 | **0.357** | 1.183 |
| idx 109 vs no species channel | **0.341** | 1.051 |

(baseline output mean 0.204.) A wrong constant channel yields output that is **still structured and still
plausible** — there is no eyeball test for this.

**Resolve before launch, in this order:**

1. **Lookup (~20 min).** `git pull` the `shorkie-paper` clone from `ed28a96` to `c6003ce`; grep the new
   `scripts/01_data_build/` and `minimal_example/` trees for the ordered 165-species list used by the
   *supervised* model (not the 1505-entry unfiltered `species_EnsemblFungi.txt`). If S. cerevisiae sorts to
   109 there, you are done. If it sorts to 9, `shorkie.py:17` is wrong and your equivalence result is
   measuring the wrong thing correctly.
2. **Experiment (~2 h, free data only).** Download S288C R64 FASTA + GFF from SGD. Build a 16384 bp window
   centred on a highly expressed gene (`TDH3`, `PGK1`, `ACT1`) and a matched silent intergenic window.
   Sweep `R64_SPECIES_INDEX = k` for `k ∈ [0, 165)` and compute mean RNA-seq-block (1148–4200) signal over
   the annotated ORF ÷ mean over intergenic. **The correct index should be a clear outlier** — only the
   right species channel gives the model a reason to place a coverage peak precisely on an annotated
   S. cerevisiae ORF. This one test simultaneously validates the bin coordinate mapping (§6.3), the track
   block offsets, and the deployment overall.

*Do not ship a `"species_index_verified": false` flag as a permanent hedge.* A day of work beats a
disclaimer in every response saying you don't know whether the model is pointed at the right species.

### 6.2 Output units — **RESOLVED: there is nothing to invert, but nothing is comparable either**

| pipeline step | value for these 5215 tracks | effect |
|---|---|---|
| crop | `-c 1024` (`--crop`, not clip) | geometry only; matches `Cropping1D(64) × 16 bp` |
| scale | column **absent** → 1 | identity |
| pool | `sum_stat = "sum"` for **all 5215 rows [VERIFIED twice]** | plain 16 bp sum; the `sum_sqrt` branch is not taken |
| clip_soft | `100000` for **all 5215 rows [VERIFIED twice]** | never fires in practice |
| clip | column **absent** | no hard clip |

So the absence of an `untransform_then_unbin` analogue in `shorkie.py` is **correct, not an omission**.
`y[b, t]` predicts the sum of that track's bigWig coverage over that 16 bp bin, in the track's native units;
the `poisson_mn` loss with a `softplus` head emits a Poisson rate directly in target space.

**The real hazard is cross-track comparison.** The experiment is literally named
`..._rna_seq_no_norm_5215_tracks`, there is no `scale` column, and in the paper's `coverage.sh` the RPM
scaling is computed and then commented out. Every track carries its own arbitrary sequencing-depth factor.
Hence `"depth_normalized": false` as a machine-readable field, so downstream code can assert on it — and
hence the preset question in §10.

*Genuinely UNKNOWN, believed negligible:* the global per-track `clip_pct` percentile trim applied at TFRecord
write time, which is unknowable without the TFRecords — they live in the **requester-pays** `gs://shorkie-paper`
bucket **[VERIFIED]**. *Fallback:* serve raw and document `value_space`, which is what this plan does anyway.
Do **not** speculatively apply `(y+1)² − 1`; with `sum_stat="sum"` that would badly distort every track.

### 6.3 Non-16384 input and bin coordinates

The model has **no shape validation anywhere** (`shorkie.py` validates the checkpoint at `:703-707`, never
the input). Measured failure taxonomy: most lengths crash; `L = 2048` returns a **silent empty tensor**;
`L = 16380…16383` returns the **same 896 bins** with a silently wrong prediction (`MaxPool1d(ceil_mode=True)`
absorbs the last 4 bp), so a bin-count check would pass; and `MultiheadAttention` computes its
relative-position basis from the *runtime* token count via `pow_rate = exp(log(T+1)/16)` (`shorkie.py:172,
180-187, 271-273`), so any other length silently rescales the entire learned positional basis to a regime
the model never saw. It will not error. It will just be wrong.

**Mitigation:** every path produces exactly `(B, 4, 16384)`; assert the tensor shape immediately before
`model(x)` — a bin-count check is not sufficient.

The bin mapping, cross-confirmed three ways in the paper repo and by forward pass **[VERIFIED]**:

> Output bin `i` covers input bp `[16i + 1024, 16i + 1040)` relative to the 16384 bp window.
> Predicted span `[1024, 15360)` = 14336 bp.

Because `origin ≡ 0 (mod 16)` and `CROP_BP = 1024 ≡ 0 (mod 16)`, the **left** boundary is never partial. The
only partial bin is the last, when `L % 16 ≠ 0`. **Do not renormalise it** — the prediction there is
conditioned on the padding actually being present; it is not a per-bp density you can rescale. Report
`last_bin_partial` / `last_bin_bp` instead.

### 6.4 Reverse-complement and strand — **the landmine does not exist here**

RC-averaging without a `strand_pair` map would normally average sense with antisense coverage, destroying
the strand specificity that makes an RNA-seq prediction interesting. **These tracks are unstranded:**
`targets.txt` has 7 columns and no `strand_pair`; a grep of all 5215 rows across `identifier` + `file` +
`description` for `strand|plus|minus|_fwd|_rev` returns **0 matches** **[VERIFIED twice, independently]**;
baskerville's loader takes the `strand_transform = None` branch when the column is absent. So yorzoi's
`{"+": [...], "-": [...]}` shape does not transfer at all.

RC therefore reduces to "is it worth 2× latency?", which is unmeasured. **v1: not exposed.** When added, the
flip is `p_rc.flip(0)` — axis 0, because Shorkie is channels-last, the opposite of yorzoi — with partner
`j' = 895 − j`.

*UNKNOWN, with an exact test:* whether some RNA-seq subset is nonetheless strand-specific. Predict a
strongly unidirectionally transcribed gene (`TDH3`/`YGR192C`) forward and reverse-complemented and compare
`pred_rc.flip(0)` against `pred_fwd` over bins 1148–4200. High R ⇒ unstranded. 10 minutes, no external data.

### 6.5 `.eval()` — the sharpest edge in the system

`from_tf_checkpoint` returns `model.training == True`. A single train-mode forward gives R ≈ −0.03 against
the correct output *and* mutates all 40 BatchNorm `running_mean`/`running_var` buffers at momentum 0.1
**[VERIFIED, reproduced twice]**. The persistent damage from one such event is R = 0.983 — not garbage, but
silently degraded and cumulative, which is worse for diagnosis. Mitigations: `.eval()` immediately after
load, `assert not model.training` at boot, `/health` re-checks it, and the upstream fix in §4.5.

### 6.6 Precision

bf16 vs fp32 on a full forward: **median rel 1.1 %, p99 5.6–5.9 %, R = 0.99990285** — reproduced to the
digit by two independent agents. That is four orders of magnitude worse than the port's fp32-vs-TF fidelity
(mean abs 1.64e-06). The win is also illusory: `ResTowerBlock` multiplies by an fp32 `res_scale` parameter
(`shorkie.py:107,112`), which **promotes the memory-dominant skip tensors back to fp32** — confirmed by
hooking the block under bf16 autocast, both outputs return float32 **[VERIFIED]**. Run fp32.

*(The draft additionally argued that fp16's exponent range endangers the BatchNorms. That justification does
not hold — loaded `running_var` spans [0.0096, 193.7] and max `|running_mean|` is 39.5, three orders of
magnitude from fp16's 65504 limit. The fp32 decision stands on the fidelity argument alone.)*

**Determinism:** B=1 vs B=4 of the same sequence differ by exactly 0.0 on CPU. On CUDA expect ~1e-6
batch-size-dependent drift from cuBLAS/cuDNN tile selection (**[EST]**). Do not promise bit-identical
results across batch sizes.

---

## 7. Testing plan

### 7.1 Unit tests — no GPU, no network, <1 s

`uv run pytest` from `backend/`, after adding the `[dependency-groups] dev` block from Step 3. Covers the
exhaustive tiling sweep; `plan_origins(16384, "auto")` → native, 1 tile at origin 1024; `merge` in both
native and full modes; `dna_1hot` N/lowercase/invalid handling; track resolution and every 422 in §3.4;
`sig(x, 3)` relative error bounded at 5e-3; track-major transpose orientation.

There is **no CI in either repo** (`.github/workflows` does not exist **[VERIFIED]**). Either add one
workflow running this tier — it is the only GPU-free, network-free one — or state in the README that tests
are run manually. Do not describe any tier as "runs in CI" until one exists.

### 7.2 GPU equivalence against the original TF model — the load-bearing test

`shorkie-pytorch/data/predictions_f0_chunk*.h5` is the oracle: `sequences (100, 16384, 4) float32` one-hot
plus `predictions (100, 896, 5215) float32` TF outputs. Three things the naive version of this test gets
wrong:

- **Use chunk 005–009, not 000.** Chunks 000–004 are synthetic random DNA
  (`modal_extract_weights.py:42-43,63,78` — `N_RANDOM = 500` emitted first, `CHUNK_SIZE = 100`). Measured:
  chunk000 pred mean 0.259 / max 18.0 vs chunk009 mean 3.18 / max **122.1**. The genomic regime is where a
  precision bug actually shows.
- **Use a relative tolerance.** The repo reports max abs error **1.95e-03** on CPU/MPS in fp32 and its own
  test gates at `max_abs < 1e-2` (`tests/test_1k_equivalence.py:134`) precisely because large predictions
  carry large absolute diffs at perfect relative fidelity. A `d.max() < 1e-3` gate would fail a *correct*
  fp32 CUDA deployment. Gate on per-sequence Pearson > 0.999999 plus `np.allclose(rtol=1e-3, atol=1e-4)`,
  and record the actual numbers.
- **Guard `OSError`.** `predictions_f0_chunk000.h5` genuinely fails on a full read (`filter returned
  failure during read`); both existing repo scripts `try/except OSError: continue` for this reason. The
  oracle is 996 usable sequences, not 1000.

**Run it over `modal run`, not HTTP.** A dedicated non-web `@app.function(gpu="L40S", image=shorkie_image)`
entrypoint reads the oracle from a `modal.Volume` and does a pure tensor comparison — no 150 s ceiling, no
npz round-trip, no response caps, and no need for a public URL to exist before the first correctness signal
arrives. Run it twice, once with `SHORKIE_ALLOW_TF32=1`, and record both deltas plus the latency difference
in the README; that is the data that decides whether the TF32-off default is worth keeping.

Then a **smaller HTTP smoke test** (n=2, `preset="all"`, `format=npz`, `window="native"`) proves the
serialization path end to end. Note that n=8 × 5215 × 896 = 37.4 M values would exceed
`MAX_VALUES_BINARY = 25 M` and be rejected by the endpoint's own budget check — the HTTP ceiling is n=5.

A one-bin coordinate shift, a transposed layout, a wrong track offset, or a missing `.eval()` each fail this
test loudly. A wrong species index does **not** — that is what §6.1 is for.

### 7.3 Biological validation — "are these numbers about yeast?"

`backend/scripts/validate_biology.py`, using only free SGD/Ensembl downloads (the ground-truth bigwigs are
in a requester-pays bucket):

1. RNA-seq block (1148–4200) mean over an annotated ORF ÷ mean over matched intergenic — record it, expect ≫1.
2. The predicted RNA-seq peak's bin boundaries align with the annotated ORF start/end to within ±2 bins.
   *This is what pins the absolute coordinate mapping against real annotation.*
3. The §6.1 species-index sweep. **This is a launch gate, not a nice-to-have.**

Run once, put the numbers in the README, re-run on any model or preprocessing change.

### 7.4 Contract, load, and regression

- **Golden response snapshot** — one fixed sequence, one fixed track list; diff the full JSON.
- **Non-regression on yorzoi** — a fixed 1 kb `/generate` request diffed against a stored response, run
  against the *ephemeral* `modal serve` URL **before** `modal deploy`, not after against production.
- **Load** — 8 concurrent full-matrix npz requests; confirm the container survives and `/health` still
  answers during a forward pass (the scenario where a sync handler would stall).
- **Fuzz the validator** — empty, whitespace-only, all-N, FASTA header, IUPAC `R`/`U`/`-`, NBSP and curly
  quotes from a Word paste, 200 kb, 100 sequences, budget overruns. Every one must be a 422 with a `code`,
  never a 500.

---

## 8. Cost / perf expectations

**GPU: L40S.** Modal per-second pricing: L40S **$0.000542** (48 GB) vs A100-40GB **$0.000583** vs A10
**$0.000306** (24 GB). At 14.2 M params and 19.7 GFLOP/seq this model is small — H100-class hardware is
pure waste, since past B≈8 you are bound by moving the head output off the GPU. **Benchmark A10 with
`?bench=true`**; if latency holds that is a 44 % cost cut. Re-check pricing at deploy.

**Memory — there is no GPU OOM here.** At B=8 fp32 under `inference_mode`: weights 57 MB, expanded
170-channel input 89 MB, `conv_dna` output 50 MB, all res-tower levels and retained U-Net skips <90 MB
combined, head output 149.5 MB plus one transposed copy. **Peak ≈ 500 MB of tensors, ~2 GB with the CUDA
context**, on a 48 GB card. The real exposure is **host** RAM on the npz path (~60–80 MB per sequence per
request through GPU → `.cpu()` → `BytesIO` → Starlette body copy), hence `memory=(4096, 16384)` and
`max_inputs=4`.

**Cold start [EST]:** 20–45 s, dominated by the multi-GB CUDA+torch image pull; then ~1–3 s to ready
(safetensors load is 0.06 s measured, plus `.to("cuda:0")` and first-kernel warm-up). Weights are baked, so
no network I/O at boot.

**Cost is dominated by idle, not compute.** L40S + 2 cores + memory ≈ $0.000586/s, so a warm 100 ms request
is ~$0.00006 — negligible. But `scaledown_window=120` means an isolated request drags a **120 s tail ≈
$0.07**, plus $0.012–0.026 of cold start. One curl from cold is ~$0.10; ten simultaneous users at
`target_inputs=2` provision five containers ≈ $0.35 of tail. `max_containers=10` is the ceiling.

**Warm latency [EST — no agent had CUDA access; anchor is 0.16 s/seq fp32 on Apple-silicon CPU]:**

| request | tiles | end-to-end |
|---|---|---|
| 1 kb, default preset | 1 | ~50–100 ms |
| 16384 bp, 64 tracks, native | 1 | ~80–150 ms |
| 16384 bp, all 5215, npz fp32 | 1 | ~300–500 ms |
| 8 × 16384 bp, aggregate, all tracks | 8 | ~0.5–1 s |
| 100 kb, 64 tracks | 13 | ~0.5 s |

Compute splits `res_tower` 32.9 % + `conv_dna` 29.9 % + `head` 18.2 % + transformer 12.9 % — the transformer
is *not* the bottleneck at 128 tokens.

**Payload sizes (measured on real fold-0 output, gzip -9):**

| response | raw | gzipped | gzip CPU |
|---|---:|---:|---:|
| default preset, 1 kb seq (4 × 63) | ~4 KB | **~1 KB** | ~0 |
| 10 tracks × 896, sig3 | 183 KB | **21.9 KB** | 0.01 s |
| `GET /tracks` (5215 entries) | 357 KB | **51 KB** | 0.03 s |
| aggregate, all 5215 tracks | 95 KB | **28 KB** | 0.03 s |
| all 5215 × 896, npz fp32 | **18.7 MB** | *(not gzipped)* | — |
| ⚠️ all 5215 × 896 as naive JSON `.tolist()` | 93 MB | 41.5 MB | **10.0 s** |
| ⚠️ 8 sequences of the above | 739 MB | 103 MB | **107 s** |

The last two rows are why the caps exist. gzip on JSON floats is only 2.24× (shortest-round-trip float reprs
are near-maximum-entropy) and gzip on binary is 1.10–1.16× — so binary responses set
`Content-Encoding: identity` and bypass the middleware entirely.

---

## 9. Out of scope for v1

Each a clean follow-up: reverse-complement TTA (needs the §6.4 benchmark); fold ensembling and shift
augmentation (per-fold **standard deviation** is the genuinely valuable output and should ship with it);
`intervals` for gene-scoped aggregation (v1 aggregates over the whole predicted span — the client slices
bins itself with zero coordinate bugs); `padding="flank"`/`"random"`/`"circular"`; configurable
`tile_step_bp` and `precision`; fp16 npz; per-bp expansion (**never** — the model has no sub-16 bp
resolution and claiming otherwise is fabrication); `/tracks` faceting and ETags; a dedicated variant-effect
endpoint (`mode="aggregate"` already makes it three client-side lines); NDJSON streaming; auth and rate
limiting (note `?bench=true` is an unauthenticated free GPU forward — remember it when auth arrives); the
`conv_dna` species-channel fold (−29 % FLOP, exact, but the forward is already 15–40 ms).

---

## 10. Decisions you need to make

1. **Species index — resolve before launch, and it is bigger than this endpoint.** §6.1 is the only item
   that can make everything else correct and still wrong, and if `109` is the wrong constant it invalidates
   `shorkie-pytorch`'s headline equivalence claim (the golden TF data uses the same constant, so the R=1.0
   result would be measuring the wrong thing accurately). *Recommended:* spend the 20-minute lookup this
   week; if inconclusive, run the 2-hour ORF-vs-intergenic sweep before announcing anything.

2. **What the default preset returns.** An unweighted mean over 1128/3053/1014 non-depth-normalized tracks
   (§6.2) is dominated by whichever tracks have the largest depth factor — the default response would be
   scale-dominated and hard to defend. *Recommended:* normalize each track by its own mean over the returned
   window before averaging, and label the preset rows `"units": "relative_to_window_mean"`. Raw single-track
   requests stay raw. *Alternative:* pick ~4 named representative tracks instead of means, which is simpler
   but requires choosing them.

3. **Whether the frontend switches to Shorkie, adds it, or offers a switcher.** Affects Step 10 and what
   happens to `backend/track_annotation.json`. *Recommended:* a model switcher — both models answer the same
   question and the comparison is the interesting part. The backend ships and is curl-usable either way.

4. **Whether to fix the live `/generate` first (Step −1).** The unbounded-input hole is reachable by one
   curl today and costs ~15 GB of host RAM; the `3000 < L ≤ 4992` bug is returning up to 39.9 % hard zeros
   as predictions right now. *Recommended:* yes, as separate commits before any Shorkie work, so a later
   regression is unambiguously attributable.
