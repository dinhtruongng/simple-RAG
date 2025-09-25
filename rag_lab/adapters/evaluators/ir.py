import csv
import math
from collections import defaultdict
from typing import Dict, List, Sequence

from rag_lab.contracts.document import Document
from rag_lab.contracts.eval import EvaluationResult, IRMetrics
from rag_lab.contracts.query import Query


class IREvaluator:
    name = "ir_basic"

    def evaluate(
        self,
        dataset_name: str,
        queries: Sequence[Query],
        docs: Sequence[Document],
        retrieval_path: str,
    ) -> EvaluationResult:
        # Read reranked results from retrieval.csv (keeps evaluator decoupled)
        reranked: Dict[str, List[str]] = defaultdict(list)
        with open(retrieval_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["stage"] == "reranked":
                    reranked[row["query_id"]].append(row["doc_id"])

        # Qrels path: load from dataset adapter – for Sprint-1, stash on self (hack)
        # In practice, pass qrels explicitly or persist alongside dataset.
        try:
            from rag_lab.adapters.datasets.hotpot_mini import HotpotMini

            qrels = HotpotMini._LAST_QRELS  # type: ignore
        except Exception:
            qrels = {}

        ks = [5, 10, 20]
        recall_at_k = {k: 0.0 for k in ks}
        ndcg_at_k = {k: 0.0 for k in ks}
        mrr = 0.0
        n = 0

        for q in queries:
            rel_docs = set(qrels.get(q.id, {}).keys())
            if not rel_docs:
                continue
            n += 1
            ranking = reranked.get(q.id, [])
            # MRR
            rr = 0.0
            for rank, d in enumerate(ranking, start=1):
                if d in rel_docs:
                    rr = 1.0 / rank
                    break
            mrr += rr
            # Recall & nDCG
            for k in ks:
                topk = ranking[:k]
                hits = [1 if d in rel_docs else 0 for d in topk]
                recall_at_k[k] += sum(hits) / max(1, len(rel_docs))
                # nDCG
                dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
                ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel_docs), k)))
                ndcg_at_k[k] += (dcg / ideal) if ideal > 0 else 0.0

        if n > 0:
            recall_at_k = {k: v / n for k, v in recall_at_k.items()}
            ndcg_at_k = {k: v / n for k, v in ndcg_at_k.items()}
            mrr /= n

        return EvaluationResult(
            dataset_name=dataset_name,
            split="dev",
            ir=IRMetrics(recall_at_k=recall_at_k, ndcg_at_k=ndcg_at_k, mrr=mrr),
            rag=None,
            notes={},
        )


# print(isinstance(IREvaluator, IEvaluator))
