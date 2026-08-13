"""Request and response models.

Structural validation lives here so FastAPI answers a malformed request with a 422
automatically. The previous implementation raised a bare ValueError inside the handler
body, which reached the client as a 500 with a stack trace containing their sequence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SequenceItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    seq: str = Field(..., min_length=1)


class PredictRequest(BaseModel):
    sequences: list[str | SequenceItem] = Field(..., min_length=1, max_length=8)

    tracks: list[int | str] | None = None
    track_groups: list[str] | None = None
    preset: str | None = None

    mode: Literal["matrix", "aggregate"] = "matrix"
    window: Literal["auto", "full", "native"] = "auto"
    step: int | None = Field(default=None, gt=0)

    @field_validator("tracks")
    @classmethod
    def _no_mixed_track_types(cls, value):
        if value and len({type(v) is int for v in value}) > 1:
            raise ValueError("tracks must be all indices or all identifiers, not a mix")
        return value

    @model_validator(mode="after")
    def _needs_a_selection(self):
        if not (self.tracks or self.track_groups or self.preset):
            self.preset = None  # resolved to the model default downstream
        return self

    def items(self) -> list[tuple[str, str]]:
        out = []
        for n, entry in enumerate(self.sequences):
            if isinstance(entry, SequenceItem):
                out.append((entry.id, entry.seq))
            else:
                out.append((str(n), entry))
        return out


class _BpRange(BaseModel):
    """A 0-based half-open bp range in the coordinates of the submitted sequence."""

    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self


class IsmWindow(_BpRange):
    """The bp range whose positions are mutated."""


class IsmInterval(_BpRange):
    """The bp range the metric reduces over (e.g. a gene body)."""


class IsmMetricSpec(BaseModel):
    """The scalar u computed for the reference and every mutant.

    `interval_mean` is the prototype: per-bin values averaged over `interval`
    (overlap-weighted for partial edge bins), averaged across the selected tracks.
    `strand_average` is reverse-complement test-time augmentation:
    u = (u_forward + u_revcomp) / 2, doubling the forward passes.
    """

    name: Literal["interval_mean"] = "interval_mean"
    interval: IsmInterval

    tracks: list[int | str] | None = None
    track_groups: list[str] | None = None
    preset: str | None = None       # None resolves to the model default downstream

    strand_average: bool = False
    agg: Literal["mean", "sum"] = "mean"

    @field_validator("tracks")
    @classmethod
    def _no_mixed_track_types(cls, value):
        if value and len({type(v) is int for v in value}) > 1:
            raise ValueError("tracks must be all indices or all identifiers, not a mix")
        return value


class IsmPlots(BaseModel):
    include: bool = True
    normalization: Literal["mean", "none", "both"] = "mean"
    """Which logos to render. All matrices are always in the result, so the client
    can switch between mean-centered and raw views without re-running the job."""

    width_px: int = Field(default=1600, ge=300, le=4000)
    dpi: int = Field(default=100, ge=50, le=200)


class IsmRequest(BaseModel):
    """Single sequence, single model window (no tiling) in v1."""

    sequence: str | SequenceItem
    ism_window: IsmWindow | None = None     # None mutates the whole sequence
    metric: IsmMetricSpec
    plots: IsmPlots = Field(default_factory=IsmPlots)

    def item(self) -> tuple[str, str]:
        if isinstance(self.sequence, SequenceItem):
            return self.sequence.id, self.sequence.seq
        return "0", self.sequence


class ErrorBody(BaseModel):
    code: str
    message: str
    field: str | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
