from typing import Protocol, Sequence, runtime_checkable

from rag_lab.contracts.document import Document
from rag_lab.contracts.eval import EvaluationResult
from rag_lab.contracts.query import Query


@runtime_checkable
class IEvaluator(Protocol):
    name: str

    def evaluate(
        self,
        dataset_name: str,
        queries: Sequence[Query],
        docs: Sequence[Document],
        retrieval_path: str,
    ) -> EvaluationResult: ...
