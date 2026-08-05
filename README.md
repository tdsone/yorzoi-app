# Yorzoi Website and API

Yorzoi (RNA-seq coverage prediction from DNA) is freely available at
[yorzoi.eu](https://www.yorzoi.eu) without coding, and via API.

## API

### Current (deployed)

`POST https://tom-ellis-lab--yorzoi-app-fastapi-app.modal.run/generate`

```jsonc
// request
{ "sequences": ["GATC…", "GTA…"] }     // any length; shorter inputs are N-padded and
                                       // centred, longer ones are tiled and averaged

// response — an object keyed by each input sequence
{ "GATC…": [[float]] }                 // [track][position], 162 tracks, per-bp
```

Add `?bench=true` to get timings instead of the payload.

Source: [`backend/index.py`](backend/index.py).

### Next (built, not yet deployed)

[`backend/`](backend/README.md) contains a rewritten, model-agnostic service that
serves both **yorzoi** and **shorkie** (a yeast sequence-to-expression model by Chao
et al., [reimplemented in PyTorch](https://github.com/tdsone/shorkie-pytorch)) through
one API:

```
GET  /health
GET  /v1/models                     geometry of every model
GET  /v1/models/{model}/tracks      track catalog
POST /v1/models/{model}/predict     ?format=json|npz
```

It fixes two defects in the deployed endpoint — sequences between 3000 and 4992 bp
currently come back with up to 40% hard zeros presented as predictions, and a coverage
gap is silently filled with zeros rather than reported — and it is validated end to end
against the original TensorFlow Shorkie (Pearson 1.0000000000). See
[`backend/README.md`](backend/README.md) for the API, the test tiers, and two geometry
facts that remain unverified.

The rewrite does not touch the deployed service; cutover is a deliberate, separate step.

## Development

```bash
cd frontend && npm run dev     # frontend dev server
cd backend  && uv run pytest   # backend test suite
```
