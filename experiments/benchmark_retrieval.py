from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.generate_synthetic_corpus import generate_benchmark


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "helps",
    "in",
    "is",
    "of",
    "on",
    "specifically",
    "the",
    "to",
    "which",
    "with",
}


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def _normalise(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _document_tokens(record: dict[str, Any]) -> list[str]:
    """Build a transparent weighted lexical representation of one record."""
    title = _tokens(record["title"])
    topic = _tokens(record["topic"])
    event = _tokens(record["event_name"])
    description = _tokens(record["description"])
    identifiers = _tokens(f"{record['record_id']} {record['event_id']}")
    people = _tokens(" ".join(record.get("speakers", [])))
    return (
        identifiers * 4
        + title * 4
        + topic * 2
        + event * 2
        + description
        + people
    )


@dataclass(frozen=True)
class SearchResult:
    record_id: str
    event_id: str
    score: float


class LexicalIndex:
    """Small dependency-free BM25 index used as the benchmark baseline."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("records must not be empty")

        self.records = records
        self.term_frequencies = [Counter(_document_tokens(record)) for record in records]
        self.document_lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_document_length = statistics.fmean(self.document_lengths)
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        document_count = len(records)
        self.inverse_document_frequency = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }
        self.indices_by_event: dict[str, list[int]] = defaultdict(list)
        self.indices_by_kind: dict[str, list[int]] = defaultdict(list)
        self.indices_by_topic: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.indices_by_event[record["event_id"]].append(index)
            self.indices_by_kind[record["kind"]].append(index)
            self.indices_by_topic[record["topic"]].append(index)

    def _score(self, query_tokens: list[str], index: int) -> float:
        frequencies = self.term_frequencies[index]
        document_length = self.document_lengths[index]
        k1 = 1.5
        b = 0.75
        score = 0.0
        for term, query_frequency in Counter(query_tokens).items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            denominator = term_frequency + k1 * (
                1 - b + b * document_length / self.average_document_length
            )
            score += (
                self.inverse_document_frequency.get(term, 0.0)
                * (term_frequency * (k1 + 1) / denominator)
                * query_frequency
            )
        return score

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        event_id: str | None = None,
        kind: str | None = None,
        topic: str | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidate_indices: Iterable[int] = range(len(self.records))
        if event_id is not None:
            candidate_indices = self.indices_by_event.get(event_id, [])
        if kind is not None:
            allowed_kind_indices = set(self.indices_by_kind.get(kind, []))
            candidate_indices = (
                index for index in candidate_indices if index in allowed_kind_indices
            )
        if topic is not None:
            allowed_topic_indices = set(self.indices_by_topic.get(topic, []))
            candidate_indices = (
                index for index in candidate_indices if index in allowed_topic_indices
            )

        query_tokens = _tokens(query)
        normalised_query = _normalise(query)
        ranked: list[tuple[float, str, int]] = []
        for index in candidate_indices:
            record = self.records[index]
            score = self._score(query_tokens, index)
            normalised_title = _normalise(record["title"])
            if normalised_title and normalised_title in normalised_query:
                score += 12.0
            if score > 0:
                ranked.append((score, record["record_id"], index))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(
                record_id=self.records[index]["record_id"],
                event_id=self.records[index]["event_id"],
                score=score,
            )
            for score, _, index in ranked[:top_k]
        ]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def evaluate_lexical(
    benchmark: dict[str, Any],
    *,
    routed: bool,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate global or event-routed lexical retrieval on labelled queries."""
    index = LexicalIndex(benchmark["records"])
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    leaked_records = 0
    returned_records = 0
    cohort_recalls: dict[str, list[float]] = defaultdict(list)

    for query in benchmark["queries"]:
        started = time.perf_counter()
        results = index.search(
            query["text"],
            top_k=top_k,
            event_id=query["event_id"] if routed else None,
            kind=query["kind"],
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        result_ids = [result.record_id for result in results]
        expected_ids = set(query["expected_record_ids"])
        recall = len(expected_ids.intersection(result_ids)) / len(expected_ids)
        recalls.append(recall)
        cohort_recalls[query["cohort"]].append(recall)
        first_relevant_rank = next(
            (rank for rank, record_id in enumerate(result_ids, start=1) if record_id in expected_ids),
            None,
        )
        reciprocal_ranks.append(0.0 if first_relevant_rank is None else 1 / first_relevant_rank)
        leaked_records += sum(result.event_id != query["event_id"] for result in results)
        returned_records += len(results)

    return {
        "mode": "event_routed" if routed else "global",
        "queries": len(benchmark["queries"]),
        "top_k": top_k,
        "recall_at_k": statistics.fmean(recalls),
        "mrr": statistics.fmean(reciprocal_ranks),
        "full_recall_rate": sum(recall == 1.0 for recall in recalls) / len(recalls),
        "cross_event_leakage_rate": (
            leaked_records / returned_records if returned_records else 0.0
        ),
        "latency_ms_p50": _percentile(latencies_ms, 0.50),
        "latency_ms_p95": _percentile(latencies_ms, 0.95),
        "cohort_recall_at_k": {
            cohort: statistics.fmean(values)
            for cohort, values in sorted(cohort_recalls.items())
        },
    }


def run_comparison(
    benchmark: dict[str, Any], *, top_k: int = 5
) -> dict[str, dict[str, Any]]:
    return {
        "global": evaluate_lexical(benchmark, routed=False, top_k=top_k),
        "event_routed": evaluate_lexical(benchmark, routed=True, top_k=top_k),
    }


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_comparison(results: dict[str, dict[str, Any]]) -> None:
    print("Lexical retrieval baseline")
    print("-" * 86)
    print(
        f"{'Mode':<15} {'Recall@5':>10} {'MRR':>8} {'Full recall':>13} "
        f"{'Event leakage':>15} {'p50 ms':>9} {'p95 ms':>9}"
    )
    for name in ("global", "event_routed"):
        result = results[name]
        print(
            f"{result['mode']:<15} "
            f"{result['recall_at_k']:>10.3f} "
            f"{result['mrr']:>8.3f} "
            f"{_format_percentage(result['full_recall_rate']):>13} "
            f"{_format_percentage(result['cross_event_leakage_rate']):>15} "
            f"{result['latency_ms_p50']:>9.2f} "
            f"{result['latency_ms_p95']:>9.2f}"
        )
    print("\nRecall@5 by query cohort")
    cohorts = results["global"]["cohort_recall_at_k"]
    for cohort in cohorts:
        global_score = results["global"]["cohort_recall_at_k"][cohort]
        routed_score = results["event_routed"]["cohort_recall_at_k"][cohort]
        print(f"- {cohort}: global={global_score:.3f}, event_routed={routed_score:.3f}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare global and event-routed lexical retrieval."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/generated/synthetic_retrieval_benchmark.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    benchmark = (
        json.loads(args.input.read_text(encoding="utf-8"))
        if args.input.exists()
        else generate_benchmark()
    )
    results = run_comparison(benchmark, top_k=args.top_k)
    print_comparison(results)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
