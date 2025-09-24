import time
from contextlib import contextmanager
from typing import Dict, Iterator

from rag_lab.contracts.pipeline import RunSpan


@contextmanager
def span(stage: str, collector: list[RunSpan], meta: Dict | None = None) -> Iterator[None]:
    start_ms = int(time.time() * 1000)
    try:
        yield
    finally:
        end_ms = int(time.time() * 1000)
        collector.append(RunSpan(stage=stage, start_ms=start_ms, end_ms=end_ms, meta=meta or {}))
