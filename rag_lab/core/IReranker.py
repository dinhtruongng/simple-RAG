from typing import List, Protocol, Sequence, runtime_checkable

from rag_lab.contracts.hit import DocHit
from rag_lab.contracts.query import Query


@runtime_checkable
class IReranker(Protocol):
    name: str

    def rerank(
        self, query: Sequence[Query], hits: Sequence[Sequence[DocHit]], top_k: int = 10
    ) -> List[List[DocHit]]: ...
