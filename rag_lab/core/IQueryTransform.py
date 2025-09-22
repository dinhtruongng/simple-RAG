from typing import List, Protocol, Sequence, runtime_checkable

from rag_lab.contracts.query import Query


@runtime_checkable
class IQueryTransform(Protocol):
    name: str

    def transform(self, queries: Sequence[Query]) -> List[Query]:
        """E.g., HyDE, paraphrase/PRF, routing (may add metadata tags)."""
        ...
