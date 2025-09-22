from typing import Protocol, runtime_checkable

from rag_lab.contracts.index import (
    IndexAddRequest,
    IndexAddResponse,
    IndexSearchRequest,
    IndexSearchResult,
)


@runtime_checkable
class IIndexer(Protocol):
    name: str
    kind: str  # "pgvector", "milvus", etc.

    def add(self, request: IndexAddRequest) -> IndexAddResponse: ...
    def search(self, request: IndexSearchRequest) -> IndexSearchResult: ...
