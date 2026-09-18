from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from experiments.benchmark_retrieval import LexicalIndex, SearchResult, _percentile
from experiments.generate_synthetic_corpus import generate_benchmark
from experiments.hybrid_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    HybridIndex,
    SentenceTransformerEncoder,
    VectorIndex,
    load_or_create_document_embeddings,
)


SearchFunction = Callable[[dict[str, Any]], list[SearchResult]]


def evaluate_search(
    benchmark: dict[str, Any], mode: str, search: SearchFunction, top_k: int
) -> dict[str, Any]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    cohort_recalls: dict[str, list[float]] = defaultdict(list)
    leaked_records = 0
    returned_records = 0

    for query in benchmark["queries"]:
        started = time.perf_counter()
        results = search(query)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        result_ids = [result.record_id for result in results]
        expected_ids = set(query["expected_record_ids"])
        recall = len(expected_ids.intersection(result_ids)) / len(expected_ids)
        recalls.append(recall)
        cohort_recalls[query["cohort"]].append(recall)
        first_rank = next(
            (rank for rank, record_id in enumerate(result_ids, start=1) if record_id in expected_ids),
            None,
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1 / first_rank)
        leaked_records += sum(result.event_id != query["event_id"] for result in results)
        returned_records += len(results)

    return {
        "mode": mode,
        "queries": len(benchmark["queries"]),
        "top_k": top_k,
        "recall_at_k": statistics.fmean(recalls),
        "mrr": statistics.fmean(reciprocal_ranks),
        "full_recall_rate": sum(recall == 1.0 for recall in recalls) / len(recalls),
        "cross_event_leakage_rate": leaked_records / returned_records,
        "latency_ms_p50": _percentile(latencies_ms, 0.50),
        "latency_ms_p95": _percentile(latencies_ms, 0.95),
        "cohort_recall_at_k": {
            cohort: statistics.fmean(values)
            for cohort, values in sorted(cohort_recalls.items())
        },
    }


def run_hybrid_comparison(
    benchmark: dict[str, Any],
    vector_index: VectorIndex,
    *,
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    records = benchmark["records"]
    lexical_index = LexicalIndex(records)
    hybrid_index = HybridIndex(records, vector_index)

    def lexical_search(query: dict[str, Any]) -> list[SearchResult]:
        return lexical_index.search(
            query["text"],
            top_k=top_k,
            event_id=query["event_id"],
            kind=query["kind"],
        )

    def vector_search(query: dict[str, Any]) -> list[SearchResult]:
        return vector_index.search(
            query["text"],
            top_k=top_k,
            event_id=query["event_id"],
            kind=query["kind"],
        )

    def hybrid_search(query: dict[str, Any]) -> list[SearchResult]:
        return hybrid_index.search(
            query["text"],
            top_k=top_k,
            event_id=query["event_id"],
            kind=query["kind"],
        )

    return {
        "lexical": evaluate_search(
            benchmark, "lexical_routed", lexical_search, top_k
        ),
        "vector": evaluate_search(
            benchmark, "vector_routed", vector_search, top_k
        ),
        "hybrid": evaluate_search(
            benchmark, "hybrid_routed", hybrid_search, top_k
        ),
    }


def print_results(results: dict[str, dict[str, Any]], build_seconds: float) -> None:
    print(f"Document embedding/index build: {build_seconds:.2f}s")
    print("Event-routed retrieval comparison")
    print("-" * 86)
    print(
        f"{'Mode':<16} {'Recall@5':>10} {'MRR':>8} {'Full recall':>13} "
        f"{'Event leakage':>15} {'p50 ms':>9} {'p95 ms':>9}"
    )
    for name in ("lexical", "vector", "hybrid"):
        result = results[name]
        print(
            f"{result['mode']:<16} "
            f"{result['recall_at_k']:>10.3f} "
            f"{result['mrr']:>8.3f} "
            f"{result['full_recall_rate'] * 100:>12.1f}% "
            f"{result['cross_event_leakage_rate'] * 100:>14.1f}% "
            f"{result['latency_ms_p50']:>9.2f} "
            f"{result['latency_ms_p95']:>9.2f}"
        )
    print("\nRecall@5 by query cohort")
    for cohort in results["lexical"]["cohort_recall_at_k"]:
        scores = ", ".join(
            f"{name}={results[name]['cohort_recall_at_k'][cohort]:.3f}"
            for name in ("lexical", "vector", "hybrid")
        )
        print(f"- {cohort}: {scores}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare event-routed lexical, vector and hybrid retrieval."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/generated/synthetic_retrieval_benchmark.json"),
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("experiments/generated/document_embeddings.npz"),
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    benchmark = (
        json.loads(args.input.read_text(encoding="utf-8"))
        if args.input.exists()
        else generate_benchmark()
    )
    encoder = SentenceTransformerEncoder(args.model)
    started = time.perf_counter()
    embeddings, cache_used = load_or_create_document_embeddings(
        benchmark["records"],
        encoder,
        args.embedding_cache,
        rebuild=args.rebuild_cache,
    )
    build_seconds = time.perf_counter() - started
    print(f"Embedding cache: {'used' if cache_used else 'created'}")
    vector_index = VectorIndex(benchmark["records"], embeddings, encoder)
    results = run_hybrid_comparison(
        benchmark,
        vector_index,
        top_k=args.top_k,
    )
    print_results(results, build_seconds)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "embedding_cache_used": cache_used,
                    "build_seconds": build_seconds,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
