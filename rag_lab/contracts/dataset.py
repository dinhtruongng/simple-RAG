from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .document import Document
from .query import Query


class DatasetBatch(BaseModel):
    """
    A fully-materialized batch for a given split.
    - IR-style datasets: populate `qrels` (query_id -> {doc_id: relevance}).
    - QA-style datasets: populate `references` (query_id -> [gold_answers]).
    Both can coexist.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Dataset name (e.g., hotpotqa)")
    split: str = Field(..., description="Split name (e.g., train/dev/test)")
    queries: List[Query]
    documents: List[Document]

    # IR supervision (BEIR style)
    qrels: Optional[Dict[str, Dict[str, int]]] = None  # query_id -> {doc_id: rel}

    # QA supervision (Hotpot/NQ style)
    references: Optional[Dict[str, List[str]]] = None  # query_id -> answers
