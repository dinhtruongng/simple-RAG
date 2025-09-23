from typing import List, Protocol, Sequence, runtime_checkable

from rag_lab.contracts.hit import DocHit
from rag_lab.contracts.query import Query


@runtime_checkable
class IReranker(Protocol):
    name: str

    def rerank(
        self, queries: Sequence[Query], hits: Sequence[Sequence[DocHit]], top_k: int = 10
    ) -> List[List[DocHit]]:
        """
        Batch rerank. The i-th list in `hits` corresponds to the i-th query in `queries`.

        Contract:
        - len(queries) == len(hits)
        - Input `hits[i]` may be any length; reranker may use all candidates.
        - Truncation with `top_k` happens *after* reranking on the reranked order.
        - Scores returned in DocHit.score are comparable *within* this reranker’s output
          (per-query), not across different reranker implementations.
        """
