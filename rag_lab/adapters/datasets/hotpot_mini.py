from typing import Dict, List, Optional

from datasets import load_dataset
from rag_lab.contracts.dataset import DatasetBatch
from rag_lab.contracts.document import Document
from rag_lab.contracts.query import Query


class HotpotMini:
    name = "hotpotqa"

    def __init__(self, split: str = "validation", max_examples: Optional[int] = 500):
        self.split = split
        self.max_examples = max_examples

    def load(self) -> DatasetBatch:
        """Load and process the HotpotQA dataset into a DatasetBatch."""
        raw_data = self._load_raw_dataset()
        queries = self._process_queries(raw_data)
        documents, qrels = self._process_documents(raw_data)

        return DatasetBatch(
            name=self.NAME, split=self.split, queries=queries, documents=documents, qrels=qrels
        )

    def _load_raw_dataset(self):
        """Load the raw dataset and optionally limit examples."""
        try:
            ds = load_dataset("hotpot_qa", "distractor", split=self.split)
            if self.max_examples:
                # Use take() for more efficient limiting without needing len()
                ds = ds.take(self.max_examples)
            return ds
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.NAME}: {e}")

    def _process_queries(self, raw_data) -> List[Query]:
        """Process raw dataset rows into Query objects."""
        queries = []
        for i, row in enumerate(raw_data):
            query_id = f"q{i}"
            queries.append(Query(id=query_id, text=row["question"]))
        return queries

    def _process_documents(self, raw_data) -> tuple[List[Document], Dict[str, Dict[str, str]]]:
        """Process raw dataset rows into Document objects and relevance mapping."""
        documents = []
        qrels: Dict[str, Dict[str, int]] = {}

        for i, row in enumerate(raw_data):
            query_id = f"q{i}"
            # Extract supporting fact titles for relevance determination
            support_titles = {title for title, _ in row["supporting_facts"]}
            ctxs = row["context"]  # [[title, [sentences]], ...]

            query_doc_map = {}
            for j, (title, sentences) in enumerate(ctxs):
                doc_id = f"d{i}_{j}"
                documents.append(
                    Document(id=doc_id, text=" ".join(sentences), metadata={"title": title})
                )
                # Store title for relevance determination
                query_doc_map[doc_id] = 1 if title in support_titles else 0

            qrels[query_id] = {k: v for k, v in query_doc_map.items() if v == 1}

        return documents, qrels


# print(isinstance(HotpotMini, DatasetAdapter))
