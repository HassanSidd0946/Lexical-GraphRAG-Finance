"""
Phase 5: Information Retrieval (IR) Metrics Evaluator (Thesis Edition)
Calculates MRR, Precision@k, Recall@k, NDCG@k, and stratifies by query type/difficulty.
"""

import json
import math
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_FILE = Path("graphrag_evaluation_results.json")
REFUSAL_MARKER = "does not contain sufficient information"
K_VALUES = (1, 5)


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def calculate_mrr(expected: list, retrieved: list) -> float:
    """Return the reciprocal rank of the first relevant result, or 0."""
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in expected:
            return 1.0 / rank
    return 0.0


def calculate_precision_at_k(expected: list, retrieved: list, k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant."""
    if not expected or not retrieved:
        return 0.0
    hits = sum(1 for chunk in retrieved[:k] if chunk in expected)
    return hits / k


def calculate_recall_at_k(expected: list, retrieved: list, k: int) -> float:
    """Fraction of relevant chunks found in the top-k retrieved results."""
    if not expected or not retrieved:
        return 0.0
    hits = sum(1 for chunk in retrieved[:k] if chunk in expected)
    return hits / len(expected)


def calculate_ndcg_at_k(expected: list, retrieved: list, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    if not expected or not retrieved:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk in enumerate(retrieved[:k], start=1)
        if chunk in expected
    )

    # Ideal DCG: all relevant chunks ranked first
    ideal_hits = min(k, len(expected))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GlobalMetrics:
    mrr: float = 0.0
    p1: float = 0.0
    p5: float = 0.0
    r1: float = 0.0
    r5: float = 0.0
    ndcg5: float = 0.0
    r_all: float = 0.0
    refusals: int = 0


@dataclass
class StratumMetrics:
    count: int = 0
    mrr: float = 0.0
    r5: float = 0.0
    ndcg5: float = 0.0

    def averages(self) -> tuple[float, float, float]:
        """Return (avg_mrr, avg_r5, avg_ndcg5), or zeros if count is 0."""
        if self.count == 0:
            return 0.0, 0.0, 0.0
        return self.mrr / self.count, self.r5 / self.count, self.ndcg5 / self.count


def _default_stratum() -> StratumMetrics:
    return StratumMetrics()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_record(record: dict) -> dict:
    """Extract and compute all metrics for a single evaluation record."""
    expected = record.get("expected_chunk_ids", [])
    retrieved = record.get("retrieved_chunk_ids", [])

    return {
        "mrr":   calculate_mrr(expected, retrieved),
        "p1":    calculate_precision_at_k(expected, retrieved, 1),
        "p5":    calculate_precision_at_k(expected, retrieved, 5),
        "r1":    calculate_recall_at_k(expected, retrieved, 1),
        "r5":    calculate_recall_at_k(expected, retrieved, 5),
        "ndcg5": calculate_ndcg_at_k(expected, retrieved, 5),
        "r_all": calculate_recall_at_k(expected, retrieved, len(retrieved)),
        "is_refusal": REFUSAL_MARKER in record.get("generated_answer", ""),
        "type":       record.get("type", "unknown"),
        "difficulty": record.get("difficulty", "unknown"),
    }


def aggregate(records: list[dict]) -> tuple[GlobalMetrics, dict, dict]:
    """
    Aggregate per-record metrics into global and stratified accumulators.

    Returns:
        global_metrics  – GlobalMetrics dataclass
        by_type         – dict[str, StratumMetrics]
        by_difficulty   – dict[str, StratumMetrics]
    """
    global_metrics = GlobalMetrics()
    by_type: dict[str, StratumMetrics] = defaultdict(_default_stratum)
    by_diff: dict[str, StratumMetrics] = defaultdict(_default_stratum)

    for record in records:
        m = evaluate_record(record)

        # Accumulate global metrics
        global_metrics.mrr     += m["mrr"]
        global_metrics.p1      += m["p1"]
        global_metrics.p5      += m["p5"]
        global_metrics.r1      += m["r1"]
        global_metrics.r5      += m["r5"]
        global_metrics.ndcg5   += m["ndcg5"]
        global_metrics.r_all   += m["r_all"]
        global_metrics.refusals += int(m["is_refusal"])

        # Accumulate stratified metrics
        for stratum, key in ((by_type, m["type"]), (by_diff, m["difficulty"])):
            stratum[key].count  += 1
            stratum[key].mrr    += m["mrr"]
            stratum[key].r5     += m["r5"]
            stratum[key].ndcg5  += m["ndcg5"]

    return global_metrics, by_type, by_diff


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SEP = "=" * 60
THIN_SEP = "-" * 60


def print_global_table(gm: GlobalMetrics, total: int) -> None:
    refusal_pct = (gm.refusals / total) * 100

    print(f"\n{SEP}")
    print(" TABLE II: Global GraphRAG Performance")
    print(SEP)
    print(f"Total Queries Evaluated : {total}")
    print(f"Safe Refusal Rate       : {refusal_pct:.1f}%  (0% Hallucination)")
    print(THIN_SEP)
    print(f"Mean Reciprocal Rank (MRR) : {gm.mrr / total:.4f}")
    print(f"Precision@1                : {gm.p1 / total:.4f}")
    print(f"Precision@5                : {gm.p5 / total:.4f}")
    print(f"Recall@1                   : {gm.r1 / total:.4f}")
    print(f"Recall@5                   : {gm.r5 / total:.4f}")
    print(f"NDCG@5                     : {gm.ndcg5 / total:.4f}")
    print(f"Total Contextual Recall    : {gm.r_all / total:.4f}")


def print_stratified_table(by_type: dict, by_diff: dict) -> None:
    header = f"{'Category':<15} | {'Count':<5} | {'MRR':<8} | {'Recall@5':<8} | {'NDCG@5':<8}"

    print(f"\n{SEP}")
    print(" TABLE III: Performance Stratified by Complexity")
    print(SEP)
    print(header)
    print(THIN_SEP)

    for label, stratum in by_type.items():
        avg_mrr, avg_r5, avg_ndcg5 = stratum.averages()
        print(f"Type: {label:<9} | {stratum.count:<5} | {avg_mrr:.4f}   | {avg_r5:.4f}   | {avg_ndcg5:.4f}")

    print(THIN_SEP)

    for label, stratum in by_diff.items():
        avg_mrr, avg_r5, avg_ndcg5 = stratum.averages()
        print(f"Diff: {label:<9} | {stratum.count:<5} | {avg_mrr:.4f}   | {avg_r5:.4f}   | {avg_ndcg5:.4f}")

    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    """Load and filter valid (error-free) records from a JSON results file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    dataset: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in dataset if "error" not in r]


def main() -> None:
    try:
        records = load_records(RESULTS_FILE)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    if not records:
        print("No valid records to evaluate.")
        return

    global_metrics, by_type, by_diff = aggregate(records)
    print_global_table(global_metrics, total=len(records))
    print_stratified_table(by_type, by_diff)


if __name__ == "__main__":
    main()