"""In-silico saturation mutagenesis (ISM).

Mutate every position in a window to every base, score each mutant with a scalar
metric u, and report tangermeme-equivalent attribution matrices plus rendered logos.
The math is reimplemented on the adapter contract rather than importing tangermeme:
the serving images stay small, and the semantics are pinned by tests instead of by a
dependency.

Module map: `metrics` (the pluggable u registry), `engine` (placement, mutant
generation, batched execution), `logo` (matplotlib rendering), `worker` (the function
that runs inside the GPU container), `jobs` (the launcher seam the API talks to).
"""

ISM_VERSION = "1.0.0"

# Models the ISM feature is enabled for. Adding a model here requires a worker
# service in modal_app.py that loads its adapter on a GPU.
ISM_MODELS = ("yorzoi", "shorkie")
