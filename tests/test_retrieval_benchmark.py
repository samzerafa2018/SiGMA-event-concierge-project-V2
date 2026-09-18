import unittest

from experiments.benchmark_retrieval import LexicalIndex, run_comparison
from experiments.generate_synthetic_corpus import generate_benchmark


class RetrievalBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = generate_benchmark(
            event_count=4,
            sessions_per_event=40,
            exhibitors_per_event=10,
            seed=91,
        )

    def test_event_routing_eliminates_cross_event_leakage(self) -> None:
        results = run_comparison(self.benchmark)

        self.assertGreater(results["global"]["cross_event_leakage_rate"], 0)
        self.assertEqual(results["event_routed"]["cross_event_leakage_rate"], 0)

    def test_event_routing_does_not_reduce_recall(self) -> None:
        results = run_comparison(self.benchmark)

        self.assertGreaterEqual(
            results["event_routed"]["recall_at_k"],
            results["global"]["recall_at_k"],
        )

    def test_results_include_quality_latency_and_cohort_metrics(self) -> None:
        results = run_comparison(self.benchmark)
        routed = results["event_routed"]

        self.assertEqual(routed["queries"], 16)
        self.assertEqual(routed["top_k"], 5)
        self.assertIn("mrr", routed)
        self.assertIn("latency_ms_p50", routed)
        self.assertIn("latency_ms_p95", routed)
        self.assertEqual(
            set(routed["cohort_recall_at_k"]),
            {
                "exact_title",
                "semantic_paraphrase",
                "cross_event_disambiguation",
                "exhibitor_semantic",
            },
        )

    def test_search_rejects_invalid_top_k(self) -> None:
        index = LexicalIndex(self.benchmark["records"])

        with self.assertRaisesRegex(ValueError, "top_k"):
            index.search("AI evaluation", top_k=0)


if __name__ == "__main__":
    unittest.main()
