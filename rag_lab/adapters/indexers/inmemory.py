from typing import List

from rag_lab.contracts.document import Chunk
from rag_lab.contracts.index import (
    IndexAddRequest,
    IndexAddResponse,
    IndexSearchRequest,
    IndexSearchResult,
)


class InMemoryIndex:
    name = "inmemory"
    kind = "stub"

    def __init__(self):
        self._chunks: List[Chunk] = []

    def add(self, request: IndexAddRequest) -> IndexAddResponse:
        chunks = request.chunks
        self._chunks.extend(chunks)
        return IndexAddResponse(added=len(chunks))

    def search(self, request: IndexSearchRequest) -> IndexSearchResult:
        # Not used in Sprint-1 BM25 path; return empty
        return IndexSearchResult(hits=[])
