"""
Retrieval Evaluation Pipeline for SEC Hybrid Search
====================================================
Evaluates HybridSearchEngine against a ground-truth dataset and produces
a comprehensive metrics report covering MRR, Precision, Recall, NDCG,
and per-query diagnostics.

Usage
-----
    python evaluate.py                          # default settings
    python evaluate.py --alpha 0.5 --top-k 10  # custom sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from hybrid_engine import HybridSearchEngine, SearchConfig

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class GroundTruthItem:
    """A single labelled evaluation example."""

    query: str
    expected_chunk_ids: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruthItem":
        if "query" not in d or "expected_chunk_ids" not in d:
            raise ValueError(f"Ground truth item missing required keys: {d!r}")
        if not d["expected_chunk_ids"]:
            raise ValueError(f"expected_chunk_ids must not be empty for query: {d['query']!r}")
        return cls(query=d["query"], expected_chunk_ids=list(d["expected_chunk_ids"]))


@dataclass
class QueryMetrics:
    """All metrics for a single query."""

    query: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    reciprocal_rank: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_5: float
    latency_ms: float

    @property
    def is_hit_at_1(self) -> bool:
        return self.recall_at_1 > 0.0


@dataclass
class AggregateMetrics:
    """Aggregated evaluation report across all queries."""

    total_queries: int
    alpha: float
    top_k: int

    # Core retrieval metrics
    mrr: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_5: float

    # Latency
    mean_latency_ms: float
    p95_latency_ms: float
    total_elapsed_s: float

    # Per-query breakdown (excluded from JSON summary by default)
    per_query: list[QueryMetrics] = field(default_factory=list, repr=False)

    def to_summary_dict(self) -> dict:
        """Flat dict suitable for JSON serialisation (no per-query rows)."""
        return {
            "total_queries": self.total_queries,
            "alpha": self.alpha,
            "top_k": self.top_k,
            "mrr": self.mrr,
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "precision_at_5": self.precision_at_5,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "ndcg_at_5": self.ndcg_at_5,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "total_elapsed_s": self.total_elapsed_s,
        }


# ──────────────────────────────────────────────
# Metric functions  (pure, stateless)
# ──────────────────────────────────────────────
def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result, or 0 if none found."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of relevant documents found in the top-k.
    Capped at 1.0 because ground truth may contain multiple expected IDs.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return min(hits / len(relevant), 1.0)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Binary relevance: 1 if relevant, 0 otherwise.
    """
    if not relevant or k <= 0:
        return 0.0

    # DCG: sum of rel_i / log2(i + 1) for i in 1..k
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )

    # Ideal DCG: assume all relevant docs are ranked first
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0.0 else 0.0


# ──────────────────────────────────────────────
# Ground truth loading
# ──────────────────────────────────────────────
def load_ground_truth(path: Path) -> list[GroundTruthItem]:
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError("ground_truth.json must be a JSON array of objects.")

    items, errors = [], 0
    for i, entry in enumerate(raw):
        try:
            items.append(GroundTruthItem.from_dict(entry))
        except ValueError as exc:
            logger.warning("Skipping ground truth entry %d: %s", i, exc)
            errors += 1

    if not items:
        raise RuntimeError("No valid ground truth entries loaded.")

    logger.info("Loaded %d ground truth items (%d skipped).", len(items), errors)
    return items


# ──────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────
class RetrievalEvaluator:
    """
    Runs the full evaluation loop and aggregates metrics.

    Parameters
    ----------
    engine      : A ready HybridSearchEngine instance.
    ground_truth: Labelled evaluation items.
    alpha       : Dense/sparse fusion weight (overrides engine default).
    top_k       : Maximum number of results to retrieve per query.
    """

    def __init__(
        self,
        engine: HybridSearchEngine,
        ground_truth: list[GroundTruthItem],
        alpha: float = 0.7,
        top_k: int = 5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        if top_k < 1:
            raise ValueError("top_k must be ≥ 1")

        self._engine = engine
        self._ground_truth = ground_truth
        self._alpha = alpha
        self._top_k = top_k

    def run(self) -> AggregateMetrics:
        """Execute evaluation and return an AggregateMetrics object."""
        n = len(self._ground_truth)
        logger.info(
            "Starting evaluation: %d queries | top_k=%d | alpha=%.2f",
            n, self._top_k, self._alpha,
        )

        per_query: list[QueryMetrics] = []
        global_start = time.perf_counter()

        for i, item in enumerate(self._ground_truth, start=1):
            qm = self._evaluate_single(item)
            per_query.append(qm)

            if i % max(1, n // 10) == 0 or i == n:
                logger.info("  Progress: %d / %d queries evaluated …", i, n)

        total_elapsed = time.perf_counter() - global_start

        return self._aggregate(per_query, total_elapsed)

    # ── private helpers ───────────────────────────────────────────────────────

    def _evaluate_single(self, item: GroundTruthItem) -> QueryMetrics:
        relevant = set(item.expected_chunk_ids)

        t0 = time.perf_counter()
        results = self._engine.search(item.query, top_k=self._top_k, alpha=self._alpha)
        latency_ms = (time.perf_counter() - t0) * 1_000

        retrieved = [r.chunk_id for r in results]

        return QueryMetrics(
            query=item.query,
            retrieved_ids=retrieved,
            expected_ids=list(relevant),
            reciprocal_rank=reciprocal_rank(retrieved, relevant),
            precision_at_1=precision_at_k(retrieved, relevant, k=1),
            precision_at_3=precision_at_k(retrieved, relevant, k=3),
            precision_at_5=precision_at_k(retrieved, relevant, k=5),
            recall_at_1=recall_at_k(retrieved, relevant, k=1),
            recall_at_3=recall_at_k(retrieved, relevant, k=3),
            recall_at_5=recall_at_k(retrieved, relevant, k=5),
            ndcg_at_5=ndcg_at_k(retrieved, relevant, k=5),
            latency_ms=round(latency_ms, 2),
        )

    def _aggregate(
        self, per_query: list[QueryMetrics], total_elapsed: float
    ) -> AggregateMetrics:
        def mean(values: list[float]) -> float:
            return round(float(np.mean(values)), 4)

        latencies = [q.latency_ms for q in per_query]

        return AggregateMetrics(
            total_queries=len(per_query),
            alpha=self._alpha,
            top_k=self._top_k,
            mrr=mean([q.reciprocal_rank for q in per_query]),
            precision_at_1=mean([q.precision_at_1 for q in per_query]),
            precision_at_3=mean([q.precision_at_3 for q in per_query]),
            precision_at_5=mean([q.precision_at_5 for q in per_query]),
            recall_at_1=mean([q.recall_at_1 for q in per_query]),
            recall_at_3=mean([q.recall_at_3 for q in per_query]),
            recall_at_5=mean([q.recall_at_5 for q in per_query]),
            ndcg_at_5=mean([q.ndcg_at_5 for q in per_query]),
            mean_latency_ms=round(float(np.mean(latencies)), 2),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
            total_elapsed_s=round(total_elapsed, 2),
            per_query=per_query,
        )


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────
_SEPARATOR = "═" * 52

def print_report(metrics: AggregateMetrics) -> None:
    """Pretty-print the aggregate report to stdout."""
    print(f"\n{_SEPARATOR}")
    print("  RETRIEVAL EVALUATION REPORT")
    print(_SEPARATOR)
    print(f"  Queries evaluated : {metrics.total_queries}")
    print(f"  Alpha (vec weight): {metrics.alpha}")
    print(f"  Top-K retrieved   : {metrics.top_k}")
    print(_SEPARATOR)
    print("  RANKING QUALITY")
    print(f"  {'Mean Reciprocal Rank (MRR)':<30} {metrics.mrr:.4f}")
    print(f"  {'NDCG@5':<30} {metrics.ndcg_at_5:.4f}")
    print(_SEPARATOR)
    print("  PRECISION")
    print(f"  {'Precision@1':<30} {metrics.precision_at_1:.4f}")
    print(f"  {'Precision@3':<30} {metrics.precision_at_3:.4f}")
    print(f"  {'Precision@5':<30} {metrics.precision_at_5:.4f}")
    print(_SEPARATOR)
    print("  RECALL")
    print(f"  {'Recall@1':<30} {metrics.recall_at_1:.4f}")
    print(f"  {'Recall@3':<30} {metrics.recall_at_3:.4f}")
    print(f"  {'Recall@5':<30} {metrics.recall_at_5:.4f}")
    print(_SEPARATOR)
    print("  LATENCY")
    print(f"  {'Mean per query':<30} {metrics.mean_latency_ms:.1f} ms")
    print(f"  {'P95 per query':<30} {metrics.p95_latency_ms:.1f} ms")
    print(f"  {'Total elapsed':<30} {metrics.total_elapsed_s:.1f} s")
    print(_SEPARATOR)

    # Flag worst-performing queries for manual inspection
    failures = [
        q for q in metrics.per_query if q.reciprocal_rank == 0.0
    ]
    if failures:
        print(f"\n  ⚠  {len(failures)} queries with zero hits (MRR = 0):")
        for q in failures[:5]:
            print(f"     • {q.query[:80]}")
        if len(failures) > 5:
            print(f"     … and {len(failures) - 5} more (see per_query report).")
    print()


def save_reports(metrics: AggregateMetrics, output_dir: Path) -> None:
    """Write summary JSON and detailed per-query JSONL to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary (flat JSON for paper tables)
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics.to_summary_dict(), fh, indent=4)
    logger.info("Summary saved → %s", summary_path)

    # Per-query detail (JSONL for downstream analysis)
    detail_path = output_dir / "metrics_per_query.jsonl"
    with detail_path.open("w", encoding="utf-8") as fh:
        for q in metrics.per_query:
            row = {
                "query": q.query,
                "rr": q.reciprocal_rank,
                "p@1": q.precision_at_1,
                "p@3": q.precision_at_3,
                "p@5": q.precision_at_5,
                "r@1": q.recall_at_1,
                "r@3": q.recall_at_3,
                "r@5": q.recall_at_5,
                "ndcg@5": q.ndcg_at_5,
                "latency_ms": q.latency_ms,
                "retrieved": q.retrieved_ids,
                "expected": q.expected_ids,
            }
            fh.write(json.dumps(row) + "\n")
    logger.info("Per-query detail saved → %s", detail_path)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the SEC Hybrid Search Engine against ground truth."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("sec_semantic_chunks_master.jsonl"),
        help="Path to the JSONL chunk corpus.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("ground_truth.json"),
        help="Path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("./chroma_db"),
        help="ChromaDB persistence directory (use 'none' for ephemeral).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Dense vector weight in fusion (0.0–1.0). Default: 0.7.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query. Default: 5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./eval_results"),
        help="Directory where metric reports are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    chroma_dir: Optional[Path] = (
        None if str(args.chroma_dir).lower() == "none" else args.chroma_dir
    )

    # Build engine
    config = SearchConfig(
        data_file=args.data_file,
        chroma_persist_dir=chroma_dir,
        default_top_k=args.top_k,
    )
    engine = HybridSearchEngine.build(config)

    # Load labels
    ground_truth = load_ground_truth(args.ground_truth)

    # Evaluate
    evaluator = RetrievalEvaluator(
        engine=engine,
        ground_truth=ground_truth,
        alpha=args.alpha,
        top_k=args.top_k,
    )
    metrics = evaluator.run()

    # Report
    print_report(metrics)
    save_reports(metrics, args.output_dir)


if __name__ == "__main__":
    main()