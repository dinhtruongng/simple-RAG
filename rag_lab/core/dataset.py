from typing import Protocol, runtime_checkable

from rag_lab.contracts.dataset import DatasetBatch


@runtime_checkable
class DatasetAdapter(Protocol):
    """
    Minimal interface to load a split into a uniform structure.
    Implementations may stream internally but must return an in-memory batch for simplicity.
    """

    name: str

    def load(self) -> DatasetBatch: ...
