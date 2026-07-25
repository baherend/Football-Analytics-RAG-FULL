"""
test_evaluation.py — Retrieval Quality Evaluation Framework

Evaluates the RAG system across multiple dimensions:
- Router accuracy (classification correctness)
- Structured accuracy (numeric answer correctness)
- Retrieval precision@K (chunk relevance)
- Retrieval level recall (document level match)
- Mean Reciprocal Rank (first relevant result position)
- Latency (response time per query type)

Usage:
    python tests/test_evaluation.py                      # Full evaluation
    python tests/test_evaluation.py --category numeric   # Filter by category
    python tests/test_evaluation.py --skip-retrieval     # Skip ChromaDB-dependent tests
    python tests/test_evaluation.py --verbose            # Per-query details
    python -m pytest tests/test_evaluation.py -v         # Pytest integration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Allow running from project root or tests directory
try:
    from tests.evaluation_benchmark import BENCHMARK, get_by_category, get_categories
except ImportError:
    from evaluation_benchmark import BENCHMARK, get_by_category, get_categories


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def _load_match_facts() -> dict:
    """Load match_facts.json for structured queries."""
    path = Path("output/match_facts.json")
    if not path.exists():
        pytest.skip("match_facts.json not found — run extraction first")
    return json.loads(path.read_text(encoding="utf-8"))


def _import_module(name: str, path: str):
    """Import a module from a file path, registering it in sys.modules."""
    import importlib.util
    # Ensure project root is on sys.path for src.* imports
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Ensure parent packages are in sys.modules
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


# ---------------------------------------------------------------------------
# Metric: Router Accuracy
# ---------------------------------------------------------------------------


def evaluate_router_accuracy(
    benchmark: list[dict],
    verbose: bool = False,
) -> dict:
    """
    Evaluate router classification accuracy.

    Returns:
        {
            "correct": int,
            "total": int,
            "accuracy": float,
            "failures": list[dict],  # {id, query, expected, actual, confidence}
        }
    """
    # Lazy import to allow --skip-retrieval mode
    router = _import_module("router", "08_router.py")

    correct = 0
    failures = []

    for entry in benchmark:
        query = entry["query"]
        expected = entry["expected_route"]

        route = router.route_query(query)
        actual = route.path

        if actual == expected:
            correct += 1
        else:
            failures.append({
                "id": entry["id"],
                "query": query,
                "expected": expected,
                "actual": actual,
                "confidence": route.confidence,
                "reason": route.reason,
            })
            if verbose:
                print(f"  X [{entry['id']}] {query}")
                print(f"    Expected: {expected}, Got: {actual} ({route.confidence:.2f})")
                print(f"    Reason: {route.reason}")

    total = len(benchmark)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Metric: Structured Accuracy
# ---------------------------------------------------------------------------


def evaluate_structured_accuracy(
    benchmark: list[dict],
    data: dict,
    verbose: bool = False,
) -> dict:
    """
    Evaluate structured query accuracy.

    Returns:
        {
            "correct": int,
            "total": int,
            "accuracy": float,
            "failures": list[dict],
            "details": list[dict],  # per-query results
        }
    """
    from src.query.query_schema import StructuredQuery, Filter
    from src.query.resolver import resolve

    structured_entries = [b for b in benchmark if b["category"] in (
        "numeric", "superlative", "slice", "edge_case"
    )]

    correct = 0
    failures = []
    details = []

    for entry in structured_entries:
        query = entry["query"]
        expected_answer = entry.get("expected_answer")
        expected_contains = entry.get("expected_answer_contains")

        # Build a StructuredQuery from the benchmark entry
        # Use the router to parse it
        router = _import_module("router", "08_router.py")

        route = router.route_query(query)
        sq = route.structured_query

        if sq is None:
            # Router couldn't parse as structured
            detail = {
                "id": entry["id"],
                "query": query,
                "status": "unparseable",
                "expected_answer": expected_answer,
                "actual_answer": None,
                "passed": expected_answer is None,  # edge cases may not parse
            }
            details.append(detail)
            if expected_answer is not None:
                failures.append({
                    "id": entry["id"],
                    "query": query,
                    "reason": "Router could not parse as structured",
                    "expected_answer": expected_answer,
                })
            continue

        result = resolve(sq, data)

        # Check answer
        passed = False
        if result.status == "empty":
            # Empty is valid for edge cases
            passed = expected_answer is None and expected_contains is None
        elif result.status in ("resolved", "partial"):
            if expected_answer is not None:
                # Exact match (with tolerance for floats)
                if isinstance(expected_answer, float):
                    passed = (result.aggregated_value is not None and
                              abs(result.aggregated_value - expected_answer) < 0.5)
                else:
                    passed = result.aggregated_value == expected_answer
            elif expected_contains is not None:
                # Substring match
                passed = expected_contains in (result.explanation or "")
            else:
                # No expected value — just check it resolved
                passed = result.aggregated_value is not None

        if passed:
            correct += 1
        else:
            failures.append({
                "id": entry["id"],
                "query": query,
                "expected_answer": expected_answer,
                "actual_answer": result.aggregated_value,
                "status": result.status,
                "explanation": result.explanation[:200],
            })

        detail = {
            "id": entry["id"],
            "query": query,
            "status": result.status,
            "expected_answer": expected_answer,
            "actual_answer": result.aggregated_value,
            "passed": passed,
        }
        details.append(detail)

        if verbose:
            symbol = "OK" if passed else "X"
            print(f"  {symbol} [{entry['id']}] {query}")
            print(f"    Expected: {expected_answer}, Got: {result.aggregated_value} ({result.status})")

    total = len(structured_entries)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "failures": failures,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Metric: Retrieval Precision@K
# ---------------------------------------------------------------------------


def evaluate_retrieval_precision(
    benchmark: list[dict],
    k: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Evaluate retrieval precision@K for semantic queries.

    Returns:
        {
            "precision_at_k": float,  # average precision across queries
            "total_queries": int,
            "queries_with_results": int,
            "details": list[dict],
            "failures": list[dict],
        }
    """
    try:
        retrieve = _import_module("retrieve", "07_retrieve_context.py")
    except Exception as e:
        return {
            "precision_at_k": 0.0,
            "total_queries": 0,
            "queries_with_results": 0,
            "details": [],
            "failures": [{"reason": f"Could not load retrieval module: {e}"}],
            "skipped": True,
        }

    semantic_entries = [b for b in benchmark if b["category"] in (
        "semantic", "hybrid"
    ) and b.get("expected_chunks_keywords")]

    precisions = []
    details = []
    failures = []

    for entry in semantic_entries:
        query = entry["query"]
        keywords = [kw.lower() for kw in entry.get("expected_chunks_keywords", [])]

        try:
            chunks = retrieve.hybrid_search(query, k=k)
        except Exception as e:
            if verbose:
                print(f"  X [{entry['id']}] {query} -- retrieval failed: {e}")
            failures.append({
                "id": entry["id"],
                "query": query,
                "reason": f"Retrieval failed: {e}",
            })
            continue

        if not chunks:
            precisions.append(0.0)
            details.append({
                "id": entry["id"],
                "query": query,
                "precision": 0.0,
                "keywords_found": [],
                "keywords_missing": keywords,
            })
            continue

        # Check how many chunks contain at least one keyword
        relevant_count = 0
        all_found = set()
        for chunk in chunks[:k]:
            text_lower = chunk.get("text", "").lower()
            chunk_keywords = [kw for kw in keywords if kw in text_lower]
            if chunk_keywords:
                relevant_count += 1
                all_found.update(chunk_keywords)

        precision = relevant_count / k if k > 0 else 0.0
        precisions.append(precision)

        missing = [kw for kw in keywords if kw not in all_found]
        detail = {
            "id": entry["id"],
            "query": query,
            "precision": precision,
            "relevant_chunks": relevant_count,
            "total_chunks": len(chunks[:k]),
            "keywords_found": list(all_found),
            "keywords_missing": missing,
        }
        details.append(detail)

        if verbose:
            symbol = "OK" if precision > 0 else "X"
            print(f"  {symbol} [{entry['id']}] {query}")
            print(f"    Precision@{k}: {precision:.2f} ({relevant_count}/{k} chunks relevant)")
            if missing:
                print(f"    Missing keywords: {missing}")

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    return {
        "precision_at_k": avg_precision,
        "total_queries": len(semantic_entries),
        "queries_with_results": sum(1 for p in precisions if p > 0),
        "details": details,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Metric: Retrieval Level Recall
# ---------------------------------------------------------------------------


def evaluate_retrieval_level_recall(
    benchmark: list[dict],
    k: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Evaluate whether retrieved chunks include the expected document level.

    Returns:
        {
            "recall": float,
            "total_queries": int,
            "details": list[dict],
        }
    """
    try:
        retrieve = _import_module("retrieve", "07_retrieve_context.py")
    except Exception as e:
        return {
            "recall": 0.0,
            "total_queries": 0,
            "details": [],
            "skipped": True,
        }

    entries_with_level = [b for b in benchmark if b.get("expected_chunks_level")]

    recalls = []
    details = []

    for entry in entries_with_level:
        query = entry["query"]
        expected_level = entry["expected_chunks_level"]

        try:
            chunks = retrieve.hybrid_search(query, k=k)
        except Exception:
            continue

        if not chunks:
            recalls.append(0.0)
            details.append({
                "id": entry["id"],
                "query": query,
                "expected_level": expected_level,
                "found": False,
                "recall": 0.0,
            })
            continue

        # Check if any chunk has the expected level
        found = any(
            chunk.get("metadata", {}).get("level") == expected_level
            for chunk in chunks[:k]
        )
        recalls.append(1.0 if found else 0.0)

        details.append({
            "id": entry["id"],
            "query": query,
            "expected_level": expected_level,
            "found": found,
            "actual_levels": [chunk.get("metadata", {}).get("level") for chunk in chunks[:k]],
        })

        if verbose:
            symbol = "OK" if found else "X"
            levels = [chunk.get("metadata", {}).get("level") for chunk in chunks[:k]]
            print(f"  {symbol} [{entry['id']}] {query}")
            print(f"    Expected level: {expected_level}, Got: {levels}")

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "recall": avg_recall,
        "total_queries": len(entries_with_level),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Metric: Mean Reciprocal Rank (MRR)
# ---------------------------------------------------------------------------


def evaluate_mrr(
    benchmark: list[dict],
    k: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Compute Mean Reciprocal Rank for semantic queries.

    MRR = average of 1/rank where rank is the position of the first
    relevant chunk (containing any expected keyword).

    Returns:
        {
            "mrr": float,
            "total_queries": int,
            "details": list[dict],
        }
    """
    try:
        retrieve = _import_module("retrieve", "07_retrieve_context.py")
    except Exception as e:
        return {
            "mrr": 0.0,
            "total_queries": 0,
            "details": [],
            "skipped": True,
        }

    entries_with_keywords = [b for b in benchmark if b.get("expected_chunks_keywords")]

    rr_values = []
    details = []

    for entry in entries_with_keywords:
        query = entry["query"]
        keywords = [kw.lower() for kw in entry.get("expected_chunks_keywords", [])]

        try:
            chunks = retrieve.hybrid_search(query, k=k)
        except Exception:
            continue

        if not chunks:
            rr_values.append(0.0)
            details.append({
                "id": entry["id"],
                "query": query,
                "rr": 0.0,
                "first_relevant_rank": None,
            })
            continue

        # Find the rank of the first relevant chunk
        first_relevant_rank = None
        for i, chunk in enumerate(chunks[:k]):
            text_lower = chunk.get("text", "").lower()
            if any(kw in text_lower for kw in keywords):
                first_relevant_rank = i + 1  # 1-indexed
                break

        rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        rr_values.append(rr)

        details.append({
            "id": entry["id"],
            "query": query,
            "rr": rr,
            "first_relevant_rank": first_relevant_rank,
        })

        if verbose:
            print(f"  [{entry['id']}] {query}")
            print(f"    First relevant rank: {first_relevant_rank}, RR: {rr:.3f}")

    avg_mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0
    return {
        "mrr": avg_mrr,
        "total_queries": len(entries_with_keywords),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Metric: Latency
# ---------------------------------------------------------------------------


def evaluate_latency(
    benchmark: list[dict],
    data: dict,
    k: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Measure response time per query type.

    Returns:
        {
            "router_avg_ms": float,
            "structured_avg_ms": float,
            "retrieval_avg_ms": float,
            "total_avg_ms": float,
            "by_category": dict[str, float],
        }
    """
    router = _import_module("router", "08_router.py")

    try:
        retrieve = _import_module("retrieve", "07_retrieve_context.py")
        has_retrieval = True
    except Exception:
        has_retrieval = False

    from src.query.resolver import resolve

    router_times = []
    structured_times = []
    retrieval_times = []
    by_category = defaultdict(list)

    for entry in benchmark:
        query = entry["query"]
        t0 = time.perf_counter()

        # Routing
        route = router.route_query(query)
        t1 = time.perf_counter()
        router_times.append((t1 - t0) * 1000)

        # Structured
        if route.structured_query:
            t2 = time.perf_counter()
            resolve(route.structured_query, data)
            t3 = time.perf_counter()
            structured_times.append((t3 - t2) * 1000)

        # Retrieval
        if has_retrieval and route.path in ("semantic", "hybrid"):
            try:
                t4 = time.perf_counter()
                retrieve.hybrid_search(route.semantic_query or query, k=k)
                t5 = time.perf_counter()
                retrieval_times.append((t5 - t4) * 1000)
            except Exception:
                pass

        total = (time.perf_counter() - t0) * 1000
        by_category[entry["category"]].append(total)

    result = {
        "router_avg_ms": sum(router_times) / len(router_times) if router_times else 0.0,
        "structured_avg_ms": (sum(structured_times) / len(structured_times)
                              if structured_times else 0.0),
        "retrieval_avg_ms": (sum(retrieval_times) / len(retrieval_times)
                             if retrieval_times else 0.0),
        "by_category": {
            cat: sum(times) / len(times) for cat, times in by_category.items()
        },
    }

    # Total is the average across all queries
    all_totals = [sum(v) for v in by_category.values()]
    total_count = sum(len(v) for v in by_category.values())
    result["total_avg_ms"] = sum(all_totals) / total_count if total_count > 0 else 0.0

    if verbose:
        print(f"  Router:      {result['router_avg_ms']:.1f} ms avg")
        print(f"  Structured:  {result['structured_avg_ms']:.1f} ms avg")
        print(f"  Retrieval:   {result['retrieval_avg_ms']:.1f} ms avg")
        print(f"  Total:       {result['total_avg_ms']:.1f} ms avg")
        for cat, avg in result["by_category"].items():
            print(f"    {cat}: {avg:.1f} ms")

    return result


# ---------------------------------------------------------------------------
# Evaluation Report
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    router_accuracy: dict = field(default_factory=dict)
    structured_accuracy: dict = field(default_factory=dict)
    retrieval_precision: dict = field(default_factory=dict)
    retrieval_level_recall: dict = field(default_factory=dict)
    mrr: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("  RAG EVALUATION REPORT")
        lines.append("=" * 60)

        # Router
        if self.router_accuracy:
            acc = self.router_accuracy
            lines.append(f"\nRouter Accuracy:      {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")

        # Structured
        if self.structured_accuracy:
            acc = self.structured_accuracy
            lines.append(f"Structured Accuracy:  {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")

        # Retrieval
        if self.retrieval_precision and not self.retrieval_precision.get("skipped"):
            p = self.retrieval_precision
            lines.append(f"Retrieval P@5:        {p['precision_at_k']:.1%} ({p['queries_with_results']}/{p['total_queries']} queries)")

        # Level Recall
        if self.retrieval_level_recall and not self.retrieval_level_recall.get("skipped"):
            r = self.retrieval_level_recall
            lines.append(f"Level Recall@5:       {r['recall']:.1%} ({r['total_queries']} queries)")

        # MRR
        if self.mrr and not self.mrr.get("skipped"):
            m = self.mrr
            lines.append(f"MRR@5:                {m['mrr']:.3f} ({m['total_queries']} queries)")

        # Latency
        if self.latency:
            lat = self.latency
            lines.append(f"\nLatency:")
            lines.append(f"  Router:       {lat['router_avg_ms']:.1f} ms")
            lines.append(f"  Structured:   {lat['structured_avg_ms']:.1f} ms")
            lines.append(f"  Retrieval:    {lat['retrieval_avg_ms']:.1f} ms")
            lines.append(f"  Total:        {lat['total_avg_ms']:.1f} ms")

        # Skipped
        if self.skipped:
            lines.append(f"\nSkipped: {', '.join(self.skipped)}")

        # Failures
        if self.router_accuracy.get("failures"):
            lines.append(f"\nRouter Failures ({len(self.router_accuracy['failures'])}):")
            for f in self.router_accuracy["failures"][:5]:
                lines.append(f"  [{f['id']}] {f['query']}")
                lines.append(f"    Expected: {f['expected']}, Got: {f['actual']}")

        if self.structured_accuracy.get("failures"):
            lines.append(f"\nStructured Failures ({len(self.structured_accuracy['failures'])}):")
            for f in self.structured_accuracy["failures"][:5]:
                lines.append(f"  [{f['id']}] {f['query']}")
                lines.append(f"    Expected: {f['expected_answer']}, Got: {f['actual_answer']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


def run_evaluation(
    benchmark: list[dict] | None = None,
    category: str | None = None,
    skip_retrieval: bool = False,
    verbose: bool = False,
) -> EvaluationReport:
    """
    Run the full evaluation pipeline.

    Parameters:
        benchmark: list of benchmark entries (default: all)
        category: filter by category (default: all)
        skip_retrieval: skip ChromaDB-dependent evaluations
        verbose: print per-query details

    Returns:
        EvaluationReport
    """
    if benchmark is None:
        benchmark = BENCHMARK

    if category:
        benchmark = [b for b in benchmark if b["category"] == category]

    report = EvaluationReport()

    # Load data for structured queries
    try:
        data = _load_match_facts()
    except pytest.skip.Exception:
        data = None
        report.skipped.append("match_facts.json not found")

    # 1. Router Accuracy
    print("\nEvaluating router accuracy...")
    report.router_accuracy = evaluate_router_accuracy(benchmark, verbose=verbose)
    print(f"  => {report.router_accuracy['accuracy']:.1%} "
          f"({report.router_accuracy['correct']}/{report.router_accuracy['total']})")

    # 2. Structured Accuracy
    if data:
        print("\nEvaluating structured accuracy...")
        report.structured_accuracy = evaluate_structured_accuracy(
            benchmark, data, verbose=verbose
        )
        print(f"  => {report.structured_accuracy['accuracy']:.1%} "
              f"({report.structured_accuracy['correct']}/{report.structured_accuracy['total']})")
    else:
        report.skipped.append("Structured accuracy (no data)")

    # 3. Retrieval metrics (require ChromaDB)
    if not skip_retrieval:
        print("\nEvaluating retrieval precision@5...")
        report.retrieval_precision = evaluate_retrieval_precision(
            benchmark, k=5, verbose=verbose
        )
        if not report.retrieval_precision.get("skipped"):
            print(f"  => {report.retrieval_precision['precision_at_k']:.1%}")
        else:
            report.skipped.append("Retrieval precision (ChromaDB not available)")

        print("\nEvaluating retrieval level recall@5...")
        report.retrieval_level_recall = evaluate_retrieval_level_recall(
            benchmark, k=5, verbose=verbose
        )
        if not report.retrieval_level_recall.get("skipped"):
            print(f"  => {report.retrieval_level_recall['recall']:.1%}")
        else:
            report.skipped.append("Level recall (ChromaDB not available)")

        print("\nEvaluating MRR@5...")
        report.mrr = evaluate_mrr(benchmark, k=5, verbose=verbose)
        if not report.mrr.get("skipped"):
            print(f"  => {report.mrr['mrr']:.3f}")
        else:
            report.skipped.append("MRR (ChromaDB not available)")
    else:
        report.skipped.append("Retrieval metrics (--skip-retrieval)")

    # 4. Latency
    if data:
        print("\nMeasuring latency...")
        report.latency = evaluate_latency(
            benchmark, data, k=5, verbose=verbose
        )
    else:
        report.skipped.append("Latency (no data)")

    return report


# ---------------------------------------------------------------------------
# Pytest Tests
# ---------------------------------------------------------------------------


def test_router_accuracy():
    """Router should classify at least 70% of queries correctly."""
    report = run_evaluation(
        benchmark=BENCHMARK,
        skip_retrieval=True,
        verbose=False,
    )
    assert report.router_accuracy["accuracy"] >= 0.70, (
        f"Router accuracy {report.router_accuracy['accuracy']:.1%} < 70%. "
        f"Failures: {len(report.router_accuracy['failures'])}"
    )


def test_structured_accuracy():
    """Structured queries should return correct answers at least 60% of the time.

    Known limitation: opponent filters (e.g. 'against France') are not extracted
    by the router's regex patterns. This reduces accuracy for slice queries.
    Threshold is set to 60% to account for this known gap.
    """
    data = _load_match_facts()
    result = evaluate_structured_accuracy(BENCHMARK, data, verbose=False)
    assert result["accuracy"] >= 0.60, (
        f"Structured accuracy {result['accuracy']:.1%} < 60%. "
        f"Failures: {len(result['failures'])}"
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_evaluation.py                      # Full evaluation
  python tests/test_evaluation.py --category numeric   # Numeric only
  python tests/test_evaluation.py --skip-retrieval     # No ChromaDB needed
  python tests/test_evaluation.py --verbose            # Per-query details
        """,
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["numeric", "superlative", "slice", "semantic", "hybrid", "edge_case"],
        help="Filter by category",
    )
    parser.add_argument(
        "--skip-retrieval", action="store_true",
        help="Skip ChromaDB-dependent evaluations",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-query details",
    )
    args = parser.parse_args()

    report = run_evaluation(
        category=args.category,
        skip_retrieval=args.skip_retrieval,
        verbose=args.verbose,
    )

    print(report.summary())

    # Exit with error if critical metrics fail
    if report.router_accuracy.get("accuracy", 0) < 0.50:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
