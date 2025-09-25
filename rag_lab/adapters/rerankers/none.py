from typing import List, Sequence

from rag_lab.contracts.hit import DocHit
from rag_lab.contracts.query import Query


class NoOpReranker:
    name = "none"

    def rerank(
        self, queries: Sequence[Query], hits: Sequence[Sequence[DocHit]], top_k: int = 10
    ) -> List[List[DocHit]]:
        return [list(h)[:top_k] for h in hits]
