from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(..., description="Stable document ID")
    text: str = Field(..., description="Raw text of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Optional vector for precomputed embeddings (multi-vector allowed)
    vectors: Optional[List[List[float]]] = None
