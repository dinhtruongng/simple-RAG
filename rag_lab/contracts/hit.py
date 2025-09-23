from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocHit(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str = Field(..., description="ID of matched doc or chunk")
    score: float = Field(
        ...,
        description=(
            "Higher is better. Comparable *within the same retriever/reranker call* "
            "(per-query monotonicity). Do NOT compare raw scores across different "
            "retrievers without calibration."
        ),
    )
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
