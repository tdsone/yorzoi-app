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