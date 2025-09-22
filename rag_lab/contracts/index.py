from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class IndexAddRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunks: List["Chunk"]
    upsert: bool = True


class IndexAddResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    added: int
    failed: int = 0


class IndexSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Either vector or sparse (BM25) terms — your indexer decides
    vectors: Optional[List[List[float]]] = None  # one or more vectors
    sparse_terms: Optional[Dict[str, float]] = None
    top_k: int = 10


class IndexSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    hits: List["DocHit"]


# avoid circular refs
from .document import Chunk
from .hit import DocHit

IndexAddRequest.model_rebuild()
IndexSearchResult.model_rebuild()
