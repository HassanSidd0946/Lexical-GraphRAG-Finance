"""
Hybrid Search Engine for SEC Financial Filings
===============================================
Combines dense vector search (ChromaDB + Azure OpenAI) with sparse BM25
lexical search via score fusion, following the benchmark-optimal weighting.

Architecture:
    DataLoader  ──►  HybridSearchEngine  ──►  SearchResult
                          │
                    ┌─────┴──────┐
              VectorIndex   BM25Index
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_openai import AzureOpenAIEmbeddings

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class SearchConfig:
    """Centralised, immutable configuration for the search engine."""

    data_file: Path = Path("sec_semantic_chunks_master.jsonl")
    collection_name: str = "sec_financial_filings"

    # Fusion hyper-parameter: weight given to the dense (vector) component.
    # 0.0 = pure BM25, 1.0 = pure vector; 0.7 is benchmark-optimal.
    alpha: float = 0.7

    # How many results to return by default
    default_top_k: int = 5

    # Maximum snippet length (characters) shown in results
    snippet_length: int = 350

    # Use a persistent Chroma directory so embeddings survive restarts.
    chroma_persist_dir: Optional[Path] = Path("./chroma_db")

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}")
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be ≥ 1")


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class Chunk:
    """A single document chunk loaded from the JSONL file."""
    chunk_id: str
    text: str
    ticker: str
    section: str
    filing_date: str

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        required = {"chunk_id", "text", "ticker", "section", "filing_date"}
        missing = required - d.keys()
        if missing:
            raise ValueError(f"Chunk dict is missing keys: {missing}")
        return cls(
            chunk_id=d["chunk_id"],
            text=d["text"],
            ticker=d["ticker"],
            section=d["section"],
            filing_date=d["filing_date"],
        )

    @property
    def metadata(self) -> dict:
        return {
            "ticker": self.ticker,
            "section": self.section,
            "filing_date": self.filing_date,
        }


@dataclass
class SearchResult:
    """A single ranked result returned by hybrid_search."""
    rank: int
    chunk_id: str
    score: float
    snippet: str
    ticker: str
    section: str
    filing_date: str

    vector_score: float = 0.0
    bm25_score: float = 0.0

    def __str__(self) -> str:
        return (
            f"Rank {self.rank:>2} │ Score: {self.score:.4f} "
            f"(vec={self.vector_score:.4f}, bm25={self.bm25_score:.4f})\n"
            f"  Source : {self.ticker} › {self.section} ({self.filing_date})\n"
            f"  ID     : {self.chunk_id}\n"
            f"  Snippet: {self.snippet}"
        )


# ──────────────────────────────────────────────
# Data loading & Utilities
# ──────────────────────────────────────────────
def load_chunks(data_file: Path) -> list[Chunk]:
    """Load and validate all chunks from a JSONL file."""
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    chunks: list[Chunk] = []
    errors = 0

    logger.info("Loading chunks from '%s' …", data_file)
    with data_file.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Skipping malformed line %d: %s", line_no, exc)
                errors += 1

    if not chunks:
        raise RuntimeError("No valid chunks were loaded. Check the data file.")

    logger.info("Loaded %d chunks (%d skipped due to errors).", len(chunks), errors)
    return chunks

def _make_sentence_snippet(text: str, max_len: int) -> str:
    """Creates a readable snippet that truncates at the nearest sentence boundary."""
    if len(text) <= max_len:
        return text
    
    truncated = text[:max_len]
    # Try to find the last period to end on a complete sentence
    last_period = truncated.rfind('. ')
    if last_period > 0:
        return truncated[:last_period + 1]
    
    # Fallback to the last whole word if no period is found
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + "..."
    
    return truncated + "..."


# ──────────────────────────────────────────────
# Azure OpenAI setup
# ──────────────────────────────────────────────
def _build_embedding_model() -> AzureOpenAIEmbeddings:
    """Construct and validate the Azure OpenAI embedding model from env vars."""
    load_dotenv()

    required_vars = {
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
    }
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "Ensure they are set in your .env file."
        )

    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )


class ChromaEmbeddingAdapter(EmbeddingFunction[Documents]):
    """Adapter to satisfy Chroma's EmbeddingFunction __call__(self, input) interface."""

    def __init__(self, embedding_model: AzureOpenAIEmbeddings) -> None:
        self._embedding_model = embedding_model

    def __call__(self, input: Documents) -> Embeddings:
        return self._embedding_model.embed_documents(list(input))

    @staticmethod
    def name() -> str:
        # Keep parity with pre-existing/default collections to avoid conflict on get_collection.
        return "default"

    @staticmethod
    def build_from_config(config: dict) -> "ChromaEmbeddingAdapter":
        raise ValueError(
            "ChromaEmbeddingAdapter cannot be rebuilt from config directly. "
            "Construct it with an AzureOpenAIEmbeddings instance."
        )

    def get_config(self) -> dict:
        return {"provider": "azure_openai_via_langchain"}


# ──────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────
class HybridSearchEngine:
    """Fused dense + sparse retrieval engine for SEC financial filings."""

    def __init__(
        self,
        chunks: list[Chunk],
        vector_collection: chromadb.Collection,
        bm25_index: BM25Okapi,
        config: SearchConfig,
    ) -> None:
        self._chunks = chunks
        self._vector_collection = vector_collection
        self._bm25_index = bm25_index
        self._config = config

        # Fast O(1) id → list-index lookup for vector alignment
        self._id_to_idx: dict[str, int] = {c.chunk_id: i for i, c in enumerate(chunks)}

    # ── factory ──────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, config: SearchConfig) -> "HybridSearchEngine":
        chunks = load_chunks(config.data_file)
        embedding_model = _build_embedding_model()

        vector_collection = cls._build_vector_index(chunks, embedding_model, config)
        bm25_index = cls._build_bm25_index(chunks)

        logger.info("HybridSearchEngine ready (%d chunks indexed).", len(chunks))
        return cls(chunks, vector_collection, bm25_index, config)

    # ── index builders ────────────────────────────────────────────────────────
    @staticmethod
    def _build_vector_index(
        chunks: list[Chunk],
        embedding_model: AzureOpenAIEmbeddings,
        config: SearchConfig,
    ) -> chromadb.Collection:
        logger.info("Connecting to ChromaDB …")
        t0 = time.perf_counter()
        chroma_embedding_fn = ChromaEmbeddingAdapter(embedding_model)

        if config.chroma_persist_dir:
            client = chromadb.PersistentClient(
                path=str(config.chroma_persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )

        # Safely handle existing collections to prevent re-embedding
        existing_collection = None
        try:
            existing_collection = client.get_collection(
                name=config.collection_name,
                embedding_function=chroma_embedding_fn,
            )
            existing_count = existing_collection.count()
            if existing_count == len(chunks):
                logger.info(
                    "Loaded existing vector index with %d chunks. Skipping Azure embedding.",
                    existing_count,
                )
                return existing_collection
            else:
                logger.warning("Chunk count mismatch detected. Rebuilding collection...")
                client.delete_collection(config.collection_name)
        except Exception as exc:
            logger.debug("Existing collection not loaded: %s", exc)

        # Create collection using COSINE space (Crucial for OpenAI embeddings)
        try:
            collection = client.create_collection(
                name=config.collection_name,
                embedding_function=chroma_embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            if "already exists" in str(exc).lower():
                logger.info(
                    "Collection '%s' already exists. Loading existing collection.",
                    config.collection_name,
                )
                collection = client.get_collection(
                    name=config.collection_name,
                    embedding_function=chroma_embedding_fn,
                )
            else:
                raise

        existing_count = collection.count()
        if existing_count > 0:
            if existing_count == len(chunks):
                logger.info(
                    "Using existing vector index with %d chunks. Skipping Azure embedding.",
                    existing_count,
                )
                return collection
            raise RuntimeError(
                "Existing collection has "
                f"{existing_count} chunks but expected {len(chunks)}. "
                "Delete the collection and rerun to rebuild cleanly."
            )

        logger.info("Embedding documents via Azure OpenAI (this may take a moment)...")
        BATCH = 100
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start : start + BATCH]
            collection.add(
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                ids=[c.chunk_id for c in batch],
            )
            logger.debug("  Embedded %d / %d chunks …", min(start + BATCH, len(chunks)), len(chunks))

        elapsed = time.perf_counter() - t0
        logger.info("Vector index built in %.1f s.", elapsed)
        return collection

    @staticmethod
    def _tokenize_fintech(text: str) -> list[str]:
        """Custom tokenizer for financial text. Preserves numbers, dates, and terms like '10-k'."""
        text = text.lower()
        # Matches words, numbers with decimals/commas, and hyphenated terms
        tokens = re.findall(r'\b[a-z0-9\.\-\,]+\b', text)
        cleaned = [t.strip('.,-') for t in tokens]
        return [t for t in cleaned if t]

    @classmethod
    def _build_bm25_index(cls, chunks: list[Chunk]) -> BM25Okapi:
        logger.info("Building BM25 lexical index with FinTech tokenizer …")
        tokenized = [cls._tokenize_fintech(c.text) for c in chunks]
        index = BM25Okapi(tokenized)
        logger.info("BM25 index ready.")
        return index

    # ── normalization ─────────────────────────────────────────────────────────
    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Scale scores to [0, 1]. Returns zeros for a constant array."""
        lo, hi = scores.min(), scores.max()
        if np.isclose(lo, hi):
            return np.zeros_like(scores, dtype=float)
        return (scores - lo) / (hi - lo)

    # ── public search API ─────────────────────────────────────────────────────
    @lru_cache(maxsize=128)
    def _cached_vector_query(self, query: str, n_results: int):
        """Caches vector queries to save API calls on repeated searches."""
        return self._vector_collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances"],
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> list[SearchResult]:
        
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        k = top_k or self._config.default_top_k
        a = alpha if alpha is not None else self._config.alpha

        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {a}")

        n = len(self._chunks)
        # Pooling optimization for large datasets (max 1000 items retrieved for fusion)
        pool_size = min(n, 1000) 
        
        logger.info("Hybrid search | query='%s' | top_k=%d | alpha=%.2f", query, k, a)

        # ── A: Dense (vector) retrieval ──────────────────────────────────────
        vector_results = self._cached_vector_query(query, pool_size)
        
        raw_distances = np.array(vector_results["distances"][0])
        # We used cosine space. Cosine similarity = 1 - distance
        vector_scores = 1.0 - raw_distances 

        aligned_vector = np.zeros(n)
        for retrieved_id, score in zip(vector_results["ids"][0], vector_scores):
            idx = self._id_to_idx.get(retrieved_id)
            if idx is not None:
                aligned_vector[idx] = score

        # ── B: Sparse (BM25) retrieval ───────────────────────────────────────
        tokenized_query = self._tokenize_fintech(query)
        raw_bm25 = np.array(self._bm25_index.get_scores(tokenized_query))

        # ── C: Normalise + fuse ──────────────────────────────────────────────
        norm_vec = self._min_max_normalize(aligned_vector)
        norm_bm25 = self._min_max_normalize(raw_bm25)
        fused = (a * norm_vec) + ((1.0 - a) * norm_bm25)

        # ── D: Rank ──────────────────────────────────────────────────────────
        top_indices = np.argsort(fused)[::-1][:k]
        snippet_len = self._config.snippet_length

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[idx]
            snippet = _make_sentence_snippet(chunk.text, snippet_len)
            
            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    score=round(float(fused[idx]), 4),
                    snippet=snippet,
                    ticker=chunk.ticker,
                    section=chunk.section,
                    filing_date=chunk.filing_date,
                    vector_score=round(float(norm_vec[idx]), 4),
                    bm25_score=round(float(norm_bm25[idx]), 4),
                )
            )

        return results


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────
def _print_results(results: list[SearchResult]) -> None:
    separator = "─" * 75
    print(f"\n{separator}")
    print(f"  {len(results)} result(s) found")
    print(separator)
    for res in results:
        print(f"\n{res}")
    print(f"\n{separator}\n")


if __name__ == "__main__":
    # Ensure this matches your file paths
    config = SearchConfig(
        data_file=Path("sec_semantic_chunks_master.jsonl"),
        chroma_persist_dir=Path("./chroma_db"),
    )

    engine = HybridSearchEngine.build(config)

    # Phase 2 Validation Queries
    queries = [
        "What are the primary legal and regulatory risks facing JPMorgan regarding financial services regulations?",
        "How do changes in interchange fees or payment card network rules affect PayPal's business?",
        "Explain how generative AI and autonomous agents present both competitive advantages and cybersecurity risks to financial institutions."
    ]

    for query in queries:
        results = engine.search(query, top_k=3, alpha=0.7)
        print(f"\nQuery: {query}")
        _print_results(results)