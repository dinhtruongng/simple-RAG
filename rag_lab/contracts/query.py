from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(..., description="Stable query ID")
    text: str = Field(..., description="User query string")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # For routing/AB tests
    tag: Optional[str] = None
