import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pydantic import TypeAdapter

from rag_lab.contracts.document import Chunk, Document
from rag_lab.contracts.eval import EvaluationResult
from rag_lab.contracts.generation import GenerationRequest, Message, Prompt
from rag_lab.contracts.hit import DocHit
from rag_lab.contracts.pipeline import ComponentInfo, RunRecord
from rag_lab.contracts.query import Query
from rag_lab.pipeline import registry
from rag_lab.pipeline.spans import span


def _ensure_outdir(dir_: str) -> Path:
    p = Path(dir_)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_pipeline(cfg: Dict[str, Any]) -> None:
    random.seed(cfg.get("seed", 42))

    outdir = _ensure_outdir(cfg["hydra"]["run"]["dir"])
    spans = []
    components: List[ComponentInfo] = []

    # 1) Dataset
    with span("load_dataset", spans):
        ds = registry.build_dataset(cfg["dataset"])
        batch = ds.load(cfg["dataset"].get("split", "validation"))
        queries: List[Query] = batch.queries[: cfg.get("limit_queries") or len(batch.queries)]
        docs: List[Document] = batch.documents

    # 2) Chunk
    with span("chunk", spans, meta={"n_docs": len(docs)}):
        chunker = registry.build_chunker(cfg["chunker"])
        chunks: List[Chunk] = chunker.split(docs)

    # 3) Index
    with span("index", spans, meta={"n_chunks": len(chunks)}):
        indexer = registry.build_indexer(cfg["indexer"])
        indexer.add(TypeAdapter(List[Chunk]).validate_python(chunks))  # InMemory stub ignores

    # 4) Retrieve
    with span("retrieve", spans, meta={"top_k_initial": cfg["top_k"]["initial"]}):
        retriever = registry.build_retriever(cfg["retriever"])
        initial_hits: List[List[DocHit]] = retriever.retrieve(
            queries, top_k=cfg["top_k"]["initial"]
        )

    # 5) Rerank
    with span("rerank", spans, meta={"top_k_rerank": cfg["top_k"]["rerank"]}):
        reranker = registry.build_reranker(cfg["reranker"])
        final_hits: List[List[DocHit]] = reranker.rerank(
            queries, initial_hits, top_k=cfg["top_k"]["rerank"]
        )

    # 6) Generate (stub)
    gen_rows = []
    with span("generate", spans, meta={"n_queries": len(queries)}):
        generator = registry.build_generator(cfg["generator"])
        for q, hits in zip(queries, final_hits):
            ctx = [h.text or "" for h in hits]
            prompt = Prompt(
                messages=[
                    Message(role="system", content=cfg["prompts"]["system"]),
                    Message(
                        role="user", content=cfg["prompts"]["user_template"].format(question=q.text)
                    ),
                ],
                version=cfg["prompts"]["version"],
            )
            req = GenerationRequest(
                prompt=prompt,
                context=ctx,
                max_tokens=cfg["generator"]["max_tokens"],
                temperature=cfg["generator"]["temperature"],
            )
            resp = generator.generate(req)
            gen_rows.append((q.id, resp.text, resp.tokens_in, resp.tokens_out))

    # 7) Evaluate (IR only for Sprint-1)
    with span("evaluate", spans):
        evaluator = registry.build_irevaluator()
        eval_result: EvaluationResult = evaluator.evaluate(
            batch.name, queries, docs
        )  # IR eval uses saved rankings

    # Write artifacts
    # retrieval.csv
    with (outdir / "retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "rank", "doc_id", "score", "stage"])
        for q, hits in zip(queries, initial_hits):
            for r, h in enumerate(hits, start=1):
                w.writerow([q.id, r, h.document_id, h.score, "initial"])
        for q, hits in zip(queries, final_hits):
            for r, h in enumerate(hits, start=1):
                w.writerow([q.id, r, h.document_id, h.score, "reranked"])

    # predictions.csv
    with (outdir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "answer_text", "tokens_in", "tokens_out"])
        w.writerows(gen_rows)

    # metrics.json
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_result.model_dump(), f, indent=2)

    # run.json
    record = RunRecord(
        run_id=datetime.now().strftime("%Y%m%d%H%M%S"),
        started_at=datetime.now(),
        seed=cfg["seed"],
        dataset_name=batch.name,
        split=batch.split,
        components=components,
        spans=spans,
        notes={"latency_budget_ms": cfg.get("latency_budget_ms")},
    )
    with (outdir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(record.model_dump(mode="json"), f, indent=2)
