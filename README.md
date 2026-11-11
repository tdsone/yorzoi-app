# Yorzoi Website and API

Explore predicted genomic signal from DNA sequences through the
[Yorzoi website](https://www.yorzoi.eu) or the prediction API.

Call the hosted API directly with the **[Python quickstart](docs/users.md#python-quickstart)**;
no Modal account or local model installation is required. See the
[public service URLs](docs/users.md#connecting-to-the-versioned-api) and
[interactive Yorzoi API docs](https://tom-ellis-lab--yorzoi-api-yorzoiservice-web.modal.run/docs).

## Current capabilities

- **Sequence editor:** a React interface built on Open Vector Editor, displaying
  forward- and reverse-strand Yorzoi coverage for three selected datasets.
- **Prediction API:** a shared `/v1` interface for **Yorzoi, Shorkie, Borzoi and
  Flashzoi**, with track selection, sequence tiling, matrix or aggregate output,
  and JSON or NumPy archives.
- **In-silico saturation mutagenesis (ISM):** asynchronous jobs for **Yorzoi and
  Shorkie**, returning mutation scores, attribution matrices and optional sequence
  logos.
- **Deployment and tests:** Modal services with separate prediction and ISM
  workers, plus tests for coordinates, API behavior, model equivalence and ISM.

The frontend currently calls the legacy Yorzoi `/generate` endpoint defined in
[`backend/index.py`](backend/index.py). The deployed `/v1` services are defined
separately in [`backend/modal_app.py`](backend/modal_app.py). The frontend does
not yet expose the other models or ISM.

## Documentation

- **[User guide](docs/users.md):** using the editor, making prediction and ISM
  requests, interpreting outputs, limits and known model caveats.
- **[Developer guide](docs/developers.md):** local setup, architecture, testing,
  model assets, extending the service and deployment.

## Local development

Frontend (Node.js and npm):

```bash
cd frontend
npm ci
npm run dev
```

Backend (Python 3.12+ and uv):

```bash
cd backend
uv sync --group dev
uv run pytest
```

The frontend uses its configured remote API even during local development. See the
[developer guide](docs/developers.md) for local API smoke tests and model setup.
