# Yorzoi Website and API

Clex is freely available at domain.com

Use Clex (RNA-seq coverage prediction from DNA) without coding.

## API

Endpoint: https://tom-ellis-lab--yorzoi-app-model-generate.modal.run

### Predict coverage for a DNA sequence

Runs a forward pass using the posted sequence and returns the unbinned and untransformed prediction.

Request: `POST https://tom-ellis-lab--yorzoi-app-model-generate.modal.run`

```
{
    sequences: List[dna] // e.g. ["G...AGC", "GTA...TCG"] each DNA seq needs to be 4992 bp long!
}
```

Response

```
{
    prediction: [[float]] // 2d array of shape tracks x pred window (e.g. 162 x 3000)
}
```

## Development

1. `cd frontend && npm run dev` to start the frontend server

## Next Steps

1. Once, the preprint is out: publish the model as a package on pypi.
2. Finish backend:
   1. Load the model
   2. Have the model make predictions
   3. Send predictions in json format back to client
