"""Typed errors with stable machine-readable codes.

Everything raised on the request path must carry a `code` clients can branch on.
Bare `assert` is banned here: it raises an unhandled 500 and `python -O` deletes it.
"""

from __future__ import annotations


class ApiError(Exception):
    """Base for anything that should reach the client as a structured error."""

    code = "internal_error"
    status = 500

    def __init__(self, message: str, *, field: str | None = None, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.details = details

    def to_dict(self, request_id: str | None = None) -> dict:
        body: dict = {"code": self.code, "message": self.message}
        if self.field is not None:
            body["field"] = self.field
        body.update(self.details)
        if request_id is not None:
            body["request_id"] = request_id
        return {"error": body}


class BadRequest(ApiError):
    code = "bad_request"
    status = 422


class InvalidAlphabet(BadRequest):
    code = "invalid_alphabet"


class SequenceTooLong(BadRequest):
    code = "sequence_too_long"


class SequenceTooShort(BadRequest):
    code = "sequence_too_short"


class TooManySequences(BadRequest):
    code = "too_many_sequences"


class UnknownModel(ApiError):
    code = "unknown_model"
    status = 404


class UnknownTrack(BadRequest):
    code = "unknown_track"


class ResponseBudgetExceeded(BadRequest):
    """The request is well formed; the *result* is too large to return.

    422 rather than 413 — 413 is for oversized request bodies, and this body is small.
    """

    code = "response_budget_exceeded"


class NativeWindowUnavailable(BadRequest):
    code = "native_window_unavailable"


class IsmUnsupportedModel(BadRequest):
    code = "ism_unsupported_model"


class IsmSequenceTooLong(BadRequest):
    """ISM is single-window: the cap is `seq_length`, not the tiled `max_seq_len`."""

    code = "ism_sequence_too_long"


class IsmWindowInvalid(BadRequest):
    code = "ism_window_invalid"


class IsmWindowContainsN(BadRequest):
    """An N has no reference base, so the raw matrix column would be meaningless.
    Ns outside the mutated window stay legal, exactly as in predict."""

    code = "ism_window_contains_n"


class IntervalOutsidePredictedSpan(BadRequest):
    code = "interval_outside_predicted_span"


class StrandAverageUnsupported(BadRequest):
    code = "strand_average_unsupported"


class UnknownMetric(BadRequest):
    code = "unknown_metric"


class JobNotFound(ApiError):
    code = "job_not_found"
    status = 404


class JobExpired(ApiError):
    """The job finished, but its stored result has passed the retention window."""

    code = "job_expired"
    status = 410


class ModelLoading(ApiError):
    code = "model_loading"
    status = 503


class ModelUnavailable(ApiError):
    """The model is known but no adapter is loaded in this container."""

    code = "model_unavailable"
    status = 503

    def __init__(self, model: str) -> None:
        super().__init__(
            f"model {model!r} is not loaded in this container", field="model", model=model
        )


class TilingGap(ApiError):
    """A planning bug left part of the sequence uncovered.

    Never paper over this by clamping the overlap counter — that returns a block of
    zeros as if they were predictions (the bug in the previous implementation).
    """

    code = "tiling_gap"
    status = 500
