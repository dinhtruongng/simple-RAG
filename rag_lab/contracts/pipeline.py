from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    stage: str  # e.g., "retrieve", "rerank", "generate", "evaluate"
    start_ms: int
    end_ms: int
    meta: Dict[str, Any] = Field(default_factory=dict)


class ComponentInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: str  # "retriever", "reranker", "indexer", "generator", "chunker"
    name: str  # implementation name (e.g., "e5_small", "bge_reranker_base")
    version: Optional[str] = None


class RunRecord(BaseModel):
    """
    Standardized run metadata. Write one JSON per run so you can diff experiments.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    started_at: datetime
    seed: int

    dataset_name: str
    split: str

    components: List[ComponentInfo]

    # Lightweight observability
    spans: List[RunSpan] = Field(default_factory=list)
    token_counts: Dict[str, int] = Field(default_factory=dict)  # {"in":..., "out":...}
    cache_hits: Dict[str, int] = Field(default_factory=dict)  # per stage

    # Free-form notes: hyperparams, budgets, etc.
    notes: Dict[str, Any] = Field(default_factory=dict)
