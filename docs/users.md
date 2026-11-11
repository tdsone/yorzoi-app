# User guide

Yorzoi predicts genomic assay coverage from DNA sequences. Use the browser editor
for the Yorzoi RNA-seq view, or the versioned API for model selection, track
selection, batch predictions, and in-silico saturation mutagenesis (ISM).

See the [project overview](../README.md) for orientation and the
[developer guide](developers.md) for local setup and deployment.

Jump to [browser usage](#using-the-browser),
[API connection](#connecting-to-the-versioned-api),
[models and limits](#choosing-a-model-and-input),
[predictions](#requesting-predictions),
[ISM](#in-silico-saturation-mutagenesis), or
[errors and caveats](#errors-and-interpretation-limits).

## Using the browser

1. Open [yorzoi.eu](https://www.yorzoi.eu).
2. Replace the placeholder sequence in the editor, or use its import tool to load
   a sequence. The sequence itself must contain only `A`, `C`, `G`, `T`, and `N`;
   lowercase is accepted.
3. A valid sequence change starts a prediction automatically. The toolbar shows
   **Predicting coverage...**, and editing is temporarily disabled while it runs.
4. When **Predicted Coverage!** appears, switch between the sequence panels to
   view the available assays. Editing the sequence triggers a fresh prediction.

The [frontend source](../frontend/src/App.jsx) selects three assays: YPD
(Nanopore), Synthetic Defined -URA (Illumina), and BY4741 UBR1 -469A>T
(Illumina). Each panel uses forward- and reverse-strand predictions. The other
two constant traces in the chromatogram display are placeholders, not additional
model predictions.

The frontend is configured to call the legacy Yorzoi endpoint:

```text
POST https://tom-ellis-lab--yorzoi-app-fastapi-app.modal.run/generate
```

It does not currently select the other models or expose ISM. The versioned API
described below is implemented separately in this repository; source code alone
does not establish which version is running at a particular public URL.

### Legacy API compatibility

The [legacy handler](../backend/index.py) accepts `{"sequences": ["ACGT..."]}`
and returns an object keyed by each normalized input sequence. Each value is a
`[track][position]` array with 162 tracks and per-base positions. The first 81
tracks and the next 81 are strand pairs. Identical input sequences share the same
object key, so this format cannot preserve duplicate entries separately.

The legacy implementation fills positions outside its predicted span with zeros.
This affects sequence edges, including sequences longer than its 3,000 bp output
span. Those zeros must not be interpreted as measured or predicted absence of
expression. Its per-base output is expanded from coarser bins; it is not an
independent model prediction at every base.

## Connecting to the versioned API

Obtain the prediction service URL for your model from the deployment operator, or
use a local server configured through the developer guide. The deployment source
defines separate prediction services and a separate ISM gateway. A URL that serves
one model need not serve another model, and the legacy `/generate` URL is not a
versioned API base URL.

In the shell examples, set `API_BASE` to the prediction service origin, with no
trailing slash. Set `ISM_BASE` separately to the ISM gateway origin when using ISM.
The examples require curl and Python 3; Python HTTP examples also use `requests`,
and NPZ examples use `numpy`.

Inspect the service before submitting work:

```bash
: "${API_BASE:?Set API_BASE to your prediction service URL}"
curl --fail-with-body --silent --show-error "$API_BASE/health"
curl --fail-with-body --silent --show-error "$API_BASE/v1/models/yorzoi"
curl --fail-with-body --silent --show-error "$API_BASE/v1/models/yorzoi/tracks"
```

`/health` lists the models in `loaded` and the ISM models served in `ism`.
`GET /v1/models` lists the model registry, including models that may not be loaded
on that service. `GET /v1/models/{model}` gives geometry, limits, the default
preset, and provenance. Its `ism_supported` flag describes model capability;
check the gateway's `/health` to discover its available ISM routes.

| Route | Purpose |
| --- | --- |
| `GET /health` | Service health and loaded models |
| `GET /v1/models` | Model registry |
| `GET /v1/models/{model}` | Model geometry, limits, and provenance |
| `GET /v1/models/{model}/tracks` | Track identifiers, groups, and presets |
| `POST /v1/models/{model}/predict` | Synchronous predictions |
| `POST /v1/models/{model}/ism` | Submit an ISM job on the gateway |
| `GET /v1/ism/jobs/{job_id}` | Poll an ISM job or retrieve its result |

## Choosing a model and input

The source defines these model geometries. Coordinates and lengths are in base
pairs (bp).

| Model | Input window | Crop per edge | Output bins | Bin size | Raw tracks | Prediction input length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yorzoi` | 4,992 | 996 | 300 | 10 | 162 | 10–131,072 |
| `shorkie` | 16,384 | 1,024 | 896 | 16 | 5,215 | 16–131,072 |
| `borzoi` | 524,288 | 163,840 | 6,144 | 32 | 7,611 | 32–524,288 |
| `flashzoi` | 524,288 | 163,840 | 6,144 | 32 | 7,611 | 32–524,288 |

Yorzoi and Shorkie provide yeast predictions; Borzoi and Flashzoi use human assay
tracks. Consult the returned track catalog and model metadata when selecting a
model for an experiment.

For the versioned API, send one to eight sequences as strings or as
`{"id": "sample-name", "seq": "ACGT..."}` objects. IDs have a maximum length of
128 characters. String inputs receive IDs `"0"`, `"1"`, and so on. Responses
preserve input order, including duplicate sequences.

Input normalization removes FASTA header lines and whitespace and converts bases
to uppercase. Only `ACGTN` is accepted after normalization. Split a multi-record
FASTA into separate request entries yourself: multiple FASTA records inside one
string are concatenated, not treated as separate samples. Length limits apply
after normalization. `N` represents unknown sequence, not a fifth nucleotide.

The additional per-request limits are:

- At most eight sequences and 64 model tiles.
- At most 262,144 total input bp for Yorzoi/Shorkie, or 1,048,576 for
  Borzoi/Flashzoi.
- At most 64 output rows for JSON matrix predictions.
- At most 25 million values for matrix predictions, including NPZ. A full
  7,611-track Borzoi or Flashzoi native matrix exceeds this budget even for one
  sequence; select fewer tracks or use aggregate mode.

These limits apply together. A sequence that meets the length limit can still
exceed the tile budget. The service's model description is the source of truth
for the limits in your deployment.

## Requesting predictions

This small curl request is valid and useful for checking connectivity. Its
repetitive 40 bp sequence is an example input, not a meaningful biological test;
short inputs have little model context.

```bash
curl --fail-with-body --silent --show-error \
  -X POST "$API_BASE/v1/models/yorzoi/predict" \
  -H 'Content-Type: application/json' \
  --data '{"sequences":[{"id":"example","seq":"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"}]}' \
  > prediction.json
```

### Track selection

Use the catalog returned by `/v1/models/{model}/tracks` to choose:

- `tracks`: a list of raw integer indices, or a list of string identifiers. Do
  not mix index and identifier types in one list.
- `track_groups`: group names from the catalog. These select every raw track in
  those groups.
- `preset`: a named profile preset, or `"all"` for every raw track.

Selections can be combined. Profile rows appear first; explicit raw tracks are
deduplicated and ordered by index. Read the response's `tracks` array to map
each output row to its meaning, rather than assuming request order.

When all selection fields are omitted, the defaults are:

| Model | Default preset | Returned profiles |
| --- | --- | --- |
| `yorzoi` | `strand_profiles` | Plus and minus strand |
| `shorkie` | `assay_profiles` | ChIP-exo, ChIP-MNase, RNA-seq (IDEA), RNA-seq (strains) |
| `borzoi`, `flashzoi` | `assay_profiles` | CAGE, DNase, ATAC, ChIP, RNA-seq |

Prediction profile presets normalize each contributing track by its own mean
over the returned window before averaging tracks. Their units are
`relative_to_window_mean`: use them to compare profile shapes. Raw track
selections preserve the model's value space.

### Window and output options

| Field or query | Default | Meaning |
| --- | --- | --- |
| `window` | `"auto"` | Uses `native` for an exact model-window-length input; otherwise uses `full` |
| `window: "native"` | — | Requires an exact input window; returns only its central predicted span, without padding |
| `window: "full"` | — | Covers the entire sequence through padding and tiling; overlapping predictions are averaged |
| `step` | Model-specific | Optional stride for `full` tiling; positive, a multiple of bin size, and no larger than the predicted span |
| `mode` | `"matrix"` | Returns a row-by-bin matrix; `"aggregate"` returns the sum over bins for each row |
| `?format=json` | `json` | JSON output, rounded to three significant digits |
| `?format=npz` | — | Binary NumPy output, with float32 prediction arrays |
| `?bench=true` | `false` | Runs prediction but returns metadata/timings instead of values |

Aggregate mode is a sum of output bins, not a mean, and does not weight a partial
last bin by its covered bp count. For normalized profile presets this sum largely
reflects window length; choose raw tracks for magnitude comparisons.

This Python example compares two 1,000 bp sequences across every Shorkie track.
Set `API_BASE` to a service with Shorkie loaded. Replace the artificial reference
with your sequence for analysis.

```python
import os
import requests

base = os.environ["API_BASE"].rstrip("/")
reference = "ACGT" * 250
alternate = reference[:500] + "T" + reference[501:]
response = requests.post(
    f"{base}/v1/models/shorkie/predict",
    json={
        "sequences": [
            {"id": "reference", "seq": reference},
            {"id": "alternate", "seq": alternate},
        ],
        "preset": "all",
        "mode": "aggregate",
    },
    timeout=180,
)
response.raise_for_status()
result = response.json()
reference_values, alternate_values = (
    prediction["values"] for prediction in result["predictions"]
)
print(result["tracks"][0], alternate_values[0] - reference_values[0])
```

### Reading JSON coordinates

The prediction response contains `meta`, `tracks`, and `predictions`. In matrix
mode, each prediction's `values` has shape `[output row][bin]`. For zero-based
bin index `k`, its interval in the normalized input sequence is:

```text
[bin_start_bp + k * bin_size_bp, bin_start_bp + (k + 1) * bin_size_bp)
```

Ranges are zero-based and half-open. Full-mode output starts at bp 0. Native-mode
output starts at the model's crop: 996 for Yorzoi, 1,024 for Shorkie, and 163,840
for Borzoi/Flashzoi. Do not assume every returned matrix starts at sequence bp 0.

For a full-mode input whose length is not divisible by bin size,
`last_bin_partial` is true and `last_bin_bp` gives the number of real sequence bp
in that bin. `pad_fraction`, `n_fraction`, and `warnings` describe missing
context. In aggregate mode, `values` is one-dimensional and `n_bins` and the
last-bin fields are null.

### Saving NPZ predictions

NPZ removes the 64-row JSON matrix limit, but retains the 25-million-value
matrix budget. This example selects two raw Yorzoi tracks:

```python
import io
import json
import os
from pathlib import Path
import numpy as np
import requests

base = os.environ["API_BASE"].rstrip("/")
response = requests.post(
    f"{base}/v1/models/yorzoi/predict?format=npz",
    json={"sequences": ["ACGT" * 250], "tracks": [0, 81]},
    timeout=180,
)
response.raise_for_status()
Path("prediction.npz").write_bytes(response.content)
with np.load(io.BytesIO(response.content), allow_pickle=False) as arrays:
    meta = json.loads(arrays["meta_json"].tobytes().decode("utf-8"))
    print(meta["npz_layout"], arrays["bin_start_bp"], arrays["track_ids"])
    print(arrays["predictions"].shape)  # (1 sequence, 2 tracks, 100 bins)
```

Equal-length output arrays are stored as `predictions` with shape
`[sequence][row][bin]`. If output bin counts differ, `meta_json.npz_layout` is
`per_sequence` and arrays are named `pred_000`, `pred_001`, etc. Other members
include `sequence_ids`, `bin_start_bp`, `track_ids`, and `track_indices`.
Profile rows have track index `-1`. In aggregate mode the final dimension is 1.

## In-silico saturation mutagenesis

ISM mutates each position in a chosen window to the other three bases and measures
the effect on a scalar metric. The gateway supports Yorzoi and Shorkie. It returns
a job ID immediately; the calculation runs separately and may take minutes.

Set `ISM_BASE` to the gateway URL and check `$ISM_BASE/health`. Prediction
services do not register the ISM routes.

An ISM request contains:

- `sequence`: one string or `{id, seq}` object, no longer than one model input
  window: 4,992 bp for Yorzoi or 16,384 bp for Shorkie. Prediction's larger tiled
  length limit does not apply.
- `ism_window`: optional `{start, end}` range of positions to mutate. Omit it
  to mutate the whole sequence. Every position within it must be `A`, `C`, `G`,
  or `T`; `N` remains allowed elsewhere.
- `metric`: required object containing `interval: {start, end}`. Its name
  defaults to `interval_mean`; it accepts `tracks`, `track_groups`, and `preset`
  selections, `agg: "mean" | "sum"`, and `strand_average: true | false`.
- `plots`: defaults to including a mean-centered logo. Set `include: false`
  to skip PNG generation. `normalization` accepts `mean`, `none`, or `both`;
  `width_px` is 300–4,000 (default 1,600) and `dpi` is 50–200 (default 100).

The metric interval must lie within the sequence's predicted span. Mutation
positions may lie outside that span because context can affect predictions
inside it. An exact-window input has the native central predicted span; shorter
sequences are positioned within one model window. For a short 1,000 bp Yorzoi
input, intervals within `[0, 1000)` are valid.

The metric averages the selected raw channels, then reduces bins using their bp
overlap with the interval. `agg: "mean"` divides by total overlap; `"sum"`
does not. ISM does **not** apply prediction-profile mean normalization, even
when a preset selects the channels. This preserves changes in overall magnitude.
`strand_average: true` averages forward and reverse-complement scores, swaps
strand-paired channels where applicable, and doubles the forward passes.

### Submitting with curl

This submits a ten-position Yorzoi scan using a 1,000 bp example sequence:

```bash
: "${ISM_BASE:?Set ISM_BASE to your ISM gateway URL}"
python3 - <<'PY' > ism-request.json
import json
print(json.dumps({
    "sequence": {"id": "example", "seq": "ACGT" * 250},
    "ism_window": {"start": 100, "end": 110},
    "metric": {"interval": {"start": 100, "end": 300}, "tracks": [0]},
    "plots": {"include": False},
}))
PY
curl --fail-with-body --silent --show-error \
  -X POST "$ISM_BASE/v1/models/yorzoi/ism" \
  -H 'Content-Type: application/json' \
  --data-binary @ism-request.json > ism-submission.json
python3 -m json.tool ism-submission.json
```

A successful submission returns HTTP 202 with `job_id`, `status: "submitted"`,
`n_mutants`, `n_forwards`, and a relative `poll` path. This example has 30 mutants
and 31 forward evaluations, including the reference. Send `GET` requests to the
gateway plus that `poll` path until the job finishes.

### Polling, saving matrices, and downloading logos

This complete Python example submits a small job, handles failure explicitly,
saves the JSON result and PNGs, and retrieves NPZ after completion:

```python
import base64
import json
import os
import time
from pathlib import Path
import requests

gateway = os.environ["ISM_BASE"].rstrip("/")
response = requests.post(
    f"{gateway}/v1/models/yorzoi/ism",
    json={
        "sequence": {"id": "example", "seq": "ACGT" * 250},
        "ism_window": {"start": 100, "end": 110},
        "metric": {
            "interval": {"start": 100, "end": 300},
            "tracks": [0],
            "strand_average": True,
        },
        "plots": {"normalization": "both", "width_px": 1200},
    },
    timeout=180,
)
response.raise_for_status()
job = response.json()
poll_url = gateway + job["poll"]
deadline = time.monotonic() + 35 * 60  # Adjust for your queue and job size.

while True:
    response = requests.get(poll_url, timeout=180)
    response.raise_for_status()  # Includes 404 unknown and 410 expired jobs.
    status = response.json()
    if status["status"] == "succeeded":
        result = status["result"]
        break
    if status["status"] == "failed":
        raise RuntimeError(status.get("error"))
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Polling deadline reached; retain job ID {job['job_id']}")
    time.sleep(5)

Path("ism-result.json").write_text(json.dumps(result), encoding="utf-8")
for index, logo in enumerate(result["logos"]):
    Path(f"ism-logo-{index}.png").write_bytes(base64.b64decode(logo["png_base64"]))

response = requests.get(poll_url + "?format=npz", timeout=180)
response.raise_for_status()
if "application/octet-stream" not in response.headers.get("Content-Type", ""):
    raise RuntimeError(response.text)  # Running/failed jobs return JSON, not NPZ.
Path("ism-result.npz").write_bytes(response.content)
```

Polling returns `running`, `failed`, or `succeeded`. A failed job is reported with
HTTP 200 and `status: "failed"`, so HTTP status alone is insufficient. Keep the
job ID to resume polling; a client timeout does not cancel the job. The Modal
worker timeout in the deployment source is 30 minutes. Results are temporary;
download them when finished. An expired result returns HTTP 410 and requires a
new submission.

### Interpreting ISM output

For a mutation window of width `W`, `matrices.alphabet` is `A, C, G, T` and
`matrices.start_bp` locates its first column in the submitted sequence:

- `raw`: shape `[4][W]`, the metric value for each base at each position. The
  reference-base entry is the unmutated score, `meta.u0`.
- `centered`: shape `[4][W]`, raw values minus the four-base mean at each
  position. These are the hypothetical attributions.
- `projected`: shape `[W]`, the centered value for the observed reference base;
  equivalently, `u0 - mean(raw[:, position])` before output rounding.

The `mean` logo shows projected attribution; `none` shows the per-base change
`raw - u0`, and `both` returns both images. These options affect images, not the
matrices. Attribution is not z-scored. Positive and negative values describe
effects on the requested metric, not a universal beneficial or harmful effect.

JSON matrices use three significant digits. ISM NPZ contains `raw`, `centered`,
`projected`, `u0`, `alphabet`, `start_bp`, and `meta_json`, without PNGs. Unlike
prediction NPZ, ISM NPZ is built from the already-rounded stored result and does
not recover additional precision.

## Errors and interpretation limits

Versioned API errors raised after structural validation use an `error` object
with a stable `code`, a `message`, and, when available, `field` and `request_id`.
Malformed JSON/request fields use FastAPI's HTTP 422 `detail` format. Preserve
the error body when reporting a problem.

| Status | Common cause | Next step |
| --- | --- | --- |
| 422 | Invalid bases/ranges, unknown tracks, wrong native length, or response budget exceeded | Correct the request; narrow tracks, reduce input, or choose aggregate mode as appropriate |
| 404 | Unknown model/job or route absent from this service | Check the base URL, model name, and job ID |
| 410 | ISM result expired | Submit again and download the completed result |
| 503 | Requested model is not loaded in this service | Check `/health` and use the model's prediction service |
| 200 with ISM `status: "failed"` | Worker error or job timeout | Read the returned error; retain the job ID for diagnosis |

Keep these scientific and output caveats in mind:

- **Context affects predictions.** Short inputs are padded with `N`, and
  full-mode edges lack context. Inspect `pad_fraction` and warnings such as
  `low_context`, `very_low_context`, `extreme_low_context`, and `mostly_n`.
  Compare sequences with matched lengths and context; short-input magnitudes
  should not be treated as fully contextualized predictions.
- **Native and full results differ.** At exactly one model input window,
  `native` makes one unpadded prediction of the center. `full` also predicts
  edges through padded tiles. Specify the mode consistently when comparing
  experiments.
- **Raw tracks are not depth-normalized.** Magnitudes across different tracks
  are not directly comparable. Yorzoi reports coverage, Shorkie reports binned
  coverage sums, and Borzoi/Flashzoi retain their scaled and squashed training
  value space. Read `value_space` and `depth_normalized` in metadata.
- **Shorkie's yeast species index remains unverified.** The adapter uses index
  109 and exposes `species_index_verified: false`. Agreement with the reference
  implementation using the same index does not independently validate the
  species assignment.
- **Yorzoi's geometry remains unverified.** The served 4,992 bp window and
  996 bp crop differ from nominal training descriptions; metadata exposes
  `geometry_verified: false`. Coordinate-sensitive interpretation needs an
  independently validated reference.

The [API implementation](../backend/app/api.py),
[request schemas](../backend/app/schemas.py), and
[model specifications](../backend/app/core/spec.py) define the contract described
here. For environment setup, model weights, or deployment changes, continue with
the [developer guide](developers.md).
