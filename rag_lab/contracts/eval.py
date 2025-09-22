from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class IRMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: Optional[float] = None


class RAGMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    faithfulness: float
    answer_relevancy: float
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_name: str
    split: str
    ir: IRMetrics
    rag: Optional[RAGMetrics] = None
    notes: Dict[str, Any] = {}
