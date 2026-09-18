from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.benchmark_hybrid import evaluate_search, split_benchmark_by_event
from experiments.generate_synthetic_corpus import generate_benchmark
from experiments.hybrid_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    AdaptiveHybridIndex,
    SentenceTransformerEncoder,
    VectorIndex,
    load_or_create_document_embeddings,
)
from experiments.routing import QueryRouter, RouteDecision


def evaluate_router(
    benchmark: dict[str, Any], decisions: dict[str, RouteDecision]
) -> dict[str, Any]:
    queries = benchmark["queries"]
    event_correct = sum(
        decisions[query["query_id"]].event_id == query["event_id"]
        for query in queries
    )
    topic_correct = sum(
        decisions[query["query_id"]].topic == query["topic"]
        for query in queries
    )
    joint_correct = sum(
        decisions[query["query_id"]].event_id == query["event_id"]
        and decisions[query["query_id"]].topic == query["topic"]
        for query in queries
    )
    return {
        "queries": len(queries),
        "event_accuracy": event_correct / len(queries),
        "topic_accuracy": topic_correct / len(queries),
        "joint_accuracy": joint_correct / len(queries),
        "event_sources": dict(
            sorted(Counter(decisions[q["query_id"]].event_source for q in queries).items())
        ),
        "topic_sources": dict(
            sorted(Counter(decisions[q["query_id"]].topic_source for q in queries).items())
        ),
    }


def run_routing_benchmark(
    benchmark: dict[str, Any],
    vector_index: VectorIndex,
    router: QueryRouter,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    adaptive = AdaptiveHybridIndex(benchmark["records"], vector_index)
    decisions = {
        query["query_id"]: router.route(query["text"])
        for query in benchmark["queries"]
    }

    def oracle_event(query: dict[str, Any]):
        return adaptive.search(
            query["text"],
            top_k=top_k,
            event_id=query["event_id"],
            kind=query["kind"],
        )

    def inferred_event(query: dict[str, Any]):
        route = decisions[query["query_id"]]
        return adaptive.search(
            query["text"],
            top_k=top_k,
            event_id=route.event_id,
            kind=query["kind"],
        )

    def inferred_event_topic(query: dict[str, Any]):
        route = decisions[query["query_id"]]
        return adaptive.search(
            query["text"],
            top_k=top_k,
            event_id=route.event_id,
            kind=query["kind"],
            topic=route.topic,
        )

    event_candidate_counts: list[int] = []
    topic_candidate_counts: list[int] = []
    for query in benchmark["queries"]:
        route = decisions[query["query_id"]]
        event_candidates = [
            record
            for record in benchmark["records"]
            if record["event_id"] == route.event_id
            and record["kind"] == query["kind"]
        ]
        topic_candidates = [
            record
            for record in event_candidates
            if record["topic"] == route.topic
        ]
        event_candidate_counts.append(len(event_candidates))
        topic_candidate_counts.append(len(topic_candidates))

    return {
        "routing": evaluate_router(benchmark, decisions),
        "retrieval": {
            "oracle_event": evaluate_search(
                benchmark, "oracle_event", oracle_event, top_k
            ),
            "inferred_event": evaluate_search(
                benchmark, "inferred_event", inferred_event, top_k
            ),
            "inferred_event_topic": evaluate_search(
                benchmark,
                "inferred_event_topic",
                inferred_event_topic,
                top_k,
            ),
        },
        "candidates": {
            "mean_event_partition": statistics.fmean(event_candidate_counts),
            "mean_event_topic_partition": statistics.fmean(topic_candidate_counts),
            "reduction_rate": 1
            - statistics.fmean(topic_candidate_counts)
            / statistics.fmean(event_candidate_counts),
        },
    }


def print_results(results: dict[str, Any], held_out: dict[str, Any]) -> None:
    routing = results["routing"]
    print("Automatic event/topic routing benchmark")
    print("-" * 78)
    print(f"Event routing accuracy: {routing['event_accuracy']:.3f}")
    print(f"Topic routing accuracy: {routing['topic_accuracy']:.3f}")
    print(f"Joint routing accuracy: {routing['joint_accuracy']:.3f}")
    print(f"Event route sources: {routing['event_sources']}")
    print(f"Topic route sources: {routing['topic_sources']}")

    print("\nRetrieval after routing")
    print(f"{'Mode':<24} {'Recall@5':>10} {'MRR':>8} {'Leakage':>10} {'p50 ms':>9}")
    for name in ("oracle_event", "inferred_event", "inferred_event_topic"):
        result = results["retrieval"][name]
        print(
            f"{result['mode']:<24} {result['recall_at_k']:>10.3f} "
            f"{result['mrr']:>8.3f} "
            f"{result['cross_event_leakage_rate'] * 100:>9.1f}% "
            f"{result['latency_ms_p50']:>9.2f}"
        )

    candidates = results["candidates"]
    print("\nCandidate-set reduction")
    print(f"Event partition mean: {candidates['mean_event_partition']:.1f}")
    print(
        "Event + topic partition mean: "
        f"{candidates['mean_event_topic_partition']:.1f}"
    )
    print(f"Reduction: {candidates['reduction_rate'] * 100:.1f}%")
    print("\nHeld-out event queries")
    print(
        f"Joint routing accuracy: {held_out['routing']['joint_accuracy']:.3f}; "
        "event+topic Recall@5: "
        f"{held_out['retrieval']['inferred_event_topic']['recall_at_k']:.3f}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate automatic event/topic routing before retrieval."
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
        benchmark["records"], encoder, args.embedding_cache
    )
    print(f"Embedding cache: {'used' if cache_used else 'created'}")
    print(f"Document embedding/index build: {time.perf_counter() - started:.2f}s")
    vector_index = VectorIndex(benchmark["records"], embeddings, encoder)
    router = QueryRouter(benchmark["events"], encoder)
    results = run_routing_benchmark(
        benchmark, vector_index, router, top_k=args.top_k
    )
    _, held_out_benchmark = split_benchmark_by_event(benchmark)
    held_out_results = run_routing_benchmark(
        held_out_benchmark, vector_index, router, top_k=args.top_k
    )
    print_results(results, held_out_results)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "embedding_cache_used": cache_used,
                    "results": results,
                    "held_out_results": held_out_results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
