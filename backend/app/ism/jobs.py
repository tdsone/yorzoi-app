"""The launcher seam between the API and the compute backend.

Modal-free by design: the Modal-backed launcher lives in `modal_app.py` and is
injected into `create_app`; tests inject `InProcessLauncher`. This is what keeps
`app/` importable (and the whole API testable) without modal installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from app.core.errors import JobNotFound


@dataclass
class JobStatus:
    state: Literal["running", "done", "failed"]
    body: dict | None = None
    """`done` -> the worker's return envelope ({"status": "succeeded"|"failed", ...}).
    `failed` -> {"error": {...}} for infrastructure-level failures (timeout, crash),
    where no worker envelope exists."""


@runtime_checkable
class IsmLauncher(Protocol):
    models: tuple[str, ...]

    def submit(self, model: str, payload: dict) -> str:
        """Start the job, return an opaque job id."""

    def status(self, job_id: str) -> JobStatus:
        """Raises JobNotFound (404) / JobExpired (410) for unknown or expired ids."""


class InProcessLauncher:
    """Synchronous launcher for tests: runs the worker inline at submit time.

    `defer=True` reports each job as running exactly once before revealing the
    result, so polling logic gets exercised."""

    def __init__(self, adapters: dict, *, models=None, defer: bool = False) -> None:
        self.models = tuple(models if models is not None else adapters)
        self._adapters = adapters
        self._defer = defer
        self._jobs: dict[str, dict] = {}
        self._pending: set[str] = set()
        self._counter = 0

    def submit(self, model: str, payload: dict) -> str:
        from app.ism.worker import execute

        job_id = f"ism-{self._counter}"
        self._counter += 1
        self._jobs[job_id] = execute(model, payload, self._adapters[model])
        if self._defer:
            self._pending.add(job_id)
        return job_id

    def status(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise JobNotFound(f"unknown job {job_id!r}", field="job_id")
        if job_id in self._pending:
            self._pending.discard(job_id)
            return JobStatus("running")
        return JobStatus("done", self._jobs[job_id])
