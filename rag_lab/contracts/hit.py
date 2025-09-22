from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocHit(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str = Field(..., description="ID of matched doc or chunk")
    score: float = Field(..., description="Higher is better, normalized per retriever")
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
