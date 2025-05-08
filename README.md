# Clex app

UI to use clex without coding.

## API

### Predict coverage for a DNA sequence
Runs a forward pass using the posted sequence and returns the unbinned and untransformed prediction.

Request: `POST /api/predict-coverage`
```
{
    sequence: str //
}
```
Response
```
{
    prediction: [[float]] // 2d array of shape tracks x pred window (e.g. 162 x 3000)
    annotation: [str] // array of track names
}
```

# Next Steps
1. Once, the preprint is out: publish the model as a package on pypi.
2. Finish backend:
    1. Load the model
    2. Have the model make predictions
    3. Send predictions in json format back to client
