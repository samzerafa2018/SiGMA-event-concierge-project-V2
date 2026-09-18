import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.benchmark_retrieval import SearchResult
from experiments.benchmark_hybrid import run_hybrid_comparison
from experiments.generate_synthetic_corpus import generate_benchmark
from experiments.hybrid_retrieval import (
    HybridIndex,
    VectorIndex,
    load_or_create_document_embeddings,
    reciprocal_rank_fusion,
)


class ConceptEncoder:
    model_id = "test/concept-encoder"

    def __init__(self) -> None:
        self.document_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if any(term in lowered for term in ("rapid payout", "instant transfer")):
            return [1.0, 0.0, 0.0]
        if any(term in lowered for term in ("identity check", "confirm customer")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def encode_documents(self, texts):
        self.document_calls += 1
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)


def _record(record_id: str, event_id: str, title: str, description: str):
    return {
        "record_id": record_id,
        "event_id": event_id,
        "event_name": event_id,
        "kind": "session",
        "topic": "Payments" if "Payout" in title else "Compliance",
        "title": title,
        "description": description,
        "speakers": [],
    }


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _record("EVT001-S0001", "EVT001", "Rapid Payout", "Instant transfer rails"),
            _record("EVT001-S0002", "EVT001", "Identity Check", "Customer verification"),
            _record("EVT002-S0001", "EVT002", "Rapid Payout", "Instant transfer rails"),
        ]
        self.encoder = ConceptEncoder()
        embeddings = self.encoder.encode_documents(
            [f"{record['title']} {record['description']}" for record in self.records]
        )
        self.vector_index = VectorIndex(self.records, embeddings, self.encoder)

    def test_vector_search_matches_meaning_and_respects_event_route(self) -> None:
        results = self.vector_index.search(
            "How can we make an instant transfer?",
            event_id="EVT001",
            kind="session",
        )

        self.assertEqual(results[0].record_id, "EVT001-S0001")
        self.assertTrue(all(result.event_id == "EVT001" for result in results))

    def test_hybrid_search_preserves_event_route(self) -> None:
        hybrid = HybridIndex(self.records, self.vector_index)
        results = hybrid.search(
            "How can we make an instant transfer?",
            event_id="EVT001",
            kind="session",
            top_k=2,
        )

        self.assertEqual(results[0].record_id, "EVT001-S0001")
        self.assertTrue(all(result.event_id == "EVT001" for result in results))

    def test_reciprocal_rank_fusion_combines_both_rankings(self) -> None:
        lexical = [
            SearchResult("A", "EVT001", 10.0),
            SearchResult("B", "EVT001", 5.0),
        ]
        vector = [
            SearchResult("B", "EVT001", 0.9),
            SearchResult("C", "EVT001", 0.8),
        ]

        fused = reciprocal_rank_fusion(lexical, vector, top_k=3)

        self.assertEqual(fused[0].record_id, "B")
        self.assertEqual({result.record_id for result in fused}, {"A", "B", "C"})

    def test_embedding_cache_is_reused_and_invalidated(self) -> None:
        encoder = ConceptEncoder()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "embeddings.npz"
            first, first_cached = load_or_create_document_embeddings(
                self.records, encoder, cache_path
            )
            second, second_cached = load_or_create_document_embeddings(
                self.records, encoder, cache_path
            )
            changed_records = [dict(record) for record in self.records]
            changed_records[0]["description"] = "Changed content"
            _, changed_cached = load_or_create_document_embeddings(
                changed_records, encoder, cache_path
            )

        self.assertFalse(first_cached)
        self.assertTrue(second_cached)
        self.assertFalse(changed_cached)
        self.assertEqual(encoder.document_calls, 2)
        np.testing.assert_allclose(first, second)

    def test_invalid_embedding_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            VectorIndex(self.records, np.asarray([1.0, 2.0]), self.encoder)
        with self.assertRaisesRegex(ValueError, "one embedding"):
            VectorIndex(self.records, np.ones((2, 3)), self.encoder)

    def test_comparison_reports_all_three_routed_methods(self) -> None:
        benchmark = generate_benchmark(
            event_count=2,
            sessions_per_event=10,
            exhibitors_per_event=5,
            seed=13,
        )
        encoder = ConceptEncoder()
        embeddings = encoder.encode_documents(
            [
                f"{record['title']} {record['description']}"
                for record in benchmark["records"]
            ]
        )
        vector_index = VectorIndex(benchmark["records"], embeddings, encoder)

        results = run_hybrid_comparison(benchmark, vector_index)

        self.assertEqual(set(results), {"lexical", "vector", "hybrid"})
        for result in results.values():
            self.assertEqual(result["queries"], 8)
            self.assertEqual(result["cross_event_leakage_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
