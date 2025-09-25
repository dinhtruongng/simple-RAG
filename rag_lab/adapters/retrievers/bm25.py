from typing import List, Optional, Sequence

from rank_bm25 import BM25Okapi

from rag_lab.contracts.document import Document
from rag_lab.contracts.hit import DocHit
from rag_lab.contracts.query import Query


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    name = "bm25"

    def __init__(self) -> None:
        """Initialize BM25Retriever with empty state."""
        self._docs: List[Document] = []
        self._bm25: Optional[BM25Okapi] = None

    def _ensure_index(self, docs: List[Document]) -> None:
        """Ensure BM25 index is built and up to date."""
        if self._bm25 is None or len(self._docs) != len(docs):
            self._docs = docs
            tokenized_docs = [_tokenize(doc.text) for doc in docs]
            self._bm25 = BM25Okapi(tokenized_docs)

    def set_corpus(self, documents: List[Document]) -> None:
        """Set the document corpus for retrieval.

        Args:
            documents: List of documents to index

        Raises:
            ValueError: If documents list is empty or contains invalid documents
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All documents must be Document instances")

        self._docs = documents
        tokenized_docs = [_tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, queries: Sequence[Query], top_k: int = 10) -> List[List[DocHit]]:
        """Retrieve documents using BM25 scoring.

        Args:
            queries: Sequence of queries to process
            top_k: Number of top documents to return per query

        Returns:
            List of document hits for each query

        Raises:
            ValueError: If queries is empty, top_k is invalid, or corpus not set
            RuntimeError: If BM25 index fails to initialize
        """
        if not queries:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if not self._docs:
            raise ValueError("Corpus must be set before retrieval")

        if self._bm25 is None:
            raise RuntimeError("BM25 index failed to initialize")

        results: List[List[DocHit]] = []
        for query in queries:
            if not isinstance(query, Query):
                raise ValueError("All queries must be Query instances")

            try:
                scores = self._bm25.get_scores(_tokenize(query.text))
                pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
                hits = [
                    DocHit(document_id=self._docs[i].id, score=float(s), text=self._docs[i].text)
                    for i, s in pairs
                ]
                results.append(hits)
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve documents for query '{query.text}': {e}")

        return results
