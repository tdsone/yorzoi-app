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


class ErrorBody(BaseModel):
    code: str
    message: str
    field: str | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
