"""The experiment-tracking interface every caller uses, and the no-op implementation.

`Tracker.run(name, params, tags, nested)` is a context manager around one tracked run: params are logged
once at the start, a run left by an exception is marked failed. `log_metrics(metrics, step)` records
numbers against a step (a Backbone version or a round number); `log_artifact(path)` attaches a file.
`NullTracker` does nothing, so code that tracks never needs a tracking backend installed.
"""

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol

Scalar = str | int | float | bool


class Tracker(Protocol):
    def run(
        self, name: str, params: Mapping[str, Scalar], tags: Mapping[str, str] | None = None, nested: bool = False
    ) -> AbstractContextManager[None]: ...

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None: ...

    def log_artifact(self, path: Path) -> None: ...


class NullTracker:
    @contextmanager
    def run(
        self, name: str, params: Mapping[str, Scalar], tags: Mapping[str, str] | None = None, nested: bool = False
    ) -> Iterator[None]:
        yield

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass
