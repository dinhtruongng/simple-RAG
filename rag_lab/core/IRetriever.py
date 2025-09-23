from typing import List, Protocol, Sequence, runtime_checkable

from rag_lab.contracts.hit import DocHit
from rag_lab.contracts.query import Query


@runtime_checkable
class IRetriever(Protocol):
    name: str

    def retrieve(self, queries: Sequence[Query], top_k: int = 10) -> List[List[DocHit]]:
        """Batch retrieve: returns a list of hits per query."""
        ...
