from typing import Any, Dict

# Chunkers
from rag_lab.adapters.chunkers.recursive import RecursiveChunker

# Datasets
from rag_lab.adapters.datasets.hotpot_mini import HotpotMini

# Evaluators
from rag_lab.adapters.evaluators.ir import IREvaluator

# Generators
from rag_lab.adapters.generators.echo import EchoGenerator

# Indexers
from rag_lab.adapters.indexers.inmemory import InMemoryIndex

# Rerankers
from rag_lab.adapters.rerankers.none import NoOpReranker

# Retrievers
from rag_lab.adapters.retrievers.bm25 import BM25Retriever

from rag_lab.core.dataset import DatasetAdapter
from rag_lab.core.IChunker import IChunker
from rag_lab.core.IEvaluator import IEvaluator
from rag_lab.core.IGenerator import IGenerator
from rag_lab.core.IIndexer import IIndexer
from rag_lab.core.IReranker import IReranker
from rag_lab.core.IRetriever import IRetriever


def build_dataset(cfg: Dict[str, Any]) -> DatasetAdapter:
    return HotpotMini(split=cfg.get("split", "validation"), max_examples=cfg.get("max_examples"))


def build_chunker(cfg: Dict[str, Any]) -> IChunker:
    return RecursiveChunker(cfg["chunk_chars"], cfg["overlap_chars"])


def build_indexer(cfg: Dict[str, Any]) -> IIndexer:
    return InMemoryIndex()


def build_retriever(cfg: Dict[str, Any]) -> IRetriever:
    return BM25Retriever(name=cfg.get("name"))


def build_reranker(cfg: Dict[str, Any]) -> IReranker:
    return NoOpReranker()


def build_generator(cfg: Dict[str, Any]) -> IGenerator:
    return EchoGenerator(
        max_tokens=cfg.get("max_tokens", 128), temperature=cfg.get("temperature", 0.0)
    )


def build_irevaluator() -> IEvaluator:
    return IREvaluator()
