import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from experiments.generate_synthetic_corpus import (
    generate_benchmark,
    validate_benchmark,
    write_benchmark,
)


class SyntheticCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = generate_benchmark(
            event_count=4,
            sessions_per_event=25,
            exhibitors_per_event=10,
            seed=42,
        )

    def test_expected_counts_and_record_kinds(self) -> None:
        records = self.benchmark["records"]
        counts = Counter(record["kind"] for record in records)

        self.assertEqual(len(self.benchmark["events"]), 4)
        self.assertEqual(counts, {"session": 100, "exhibitor": 40})
        self.assertEqual(len(self.benchmark["queries"]), 16)

    def test_generation_is_reproducible_for_a_fixed_seed(self) -> None:
        first = generate_benchmark(
            event_count=2,
            sessions_per_event=10,
            exhibitors_per_event=5,
            seed=7,
        )
        second = generate_benchmark(
            event_count=2,
            sessions_per_event=10,
            exhibitors_per_event=5,
            seed=7,
        )
        different_seed = generate_benchmark(
            event_count=2,
            sessions_per_event=10,
            exhibitors_per_event=5,
            seed=8,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)

    def test_ids_are_unique_and_every_query_target_exists(self) -> None:
        records = self.benchmark["records"]
        record_ids = [record["record_id"] for record in records]
        known_ids = set(record_ids)

        self.assertEqual(len(record_ids), len(known_ids))
        for query in self.benchmark["queries"]:
            self.assertTrue(query["expected_record_ids"])
            self.assertTrue(set(query["expected_record_ids"]).issubset(known_ids))
            self.assertTrue(
                all(
                    record_id.startswith(f"{query['event_id']}-")
                    for record_id in query["expected_record_ids"]
                )
            )

    def test_repeated_titles_create_cross_event_distractors(self) -> None:
        events_by_title: dict[str, set[str]] = defaultdict(set)
        for record in self.benchmark["records"]:
            if record["kind"] == "session":
                events_by_title[record["title"]].add(record["event_id"])

        repeated_titles = [
            title for title, event_ids in events_by_title.items() if len(event_ids) > 1
        ]
        self.assertTrue(repeated_titles)

    def test_all_retrieval_challenge_cohorts_are_present(self) -> None:
        cohorts = Counter(query["cohort"] for query in self.benchmark["queries"])

        self.assertEqual(
            cohorts,
            {
                "exact_title": 4,
                "semantic_paraphrase": 4,
                "cross_event_disambiguation": 4,
                "exhibitor_semantic": 4,
            },
        )

    def test_written_benchmark_round_trips_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "benchmark.json"
            write_benchmark(self.benchmark, output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, self.benchmark)
        validate_benchmark(loaded)

    def test_invalid_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "event_count"):
            generate_benchmark(event_count=1)
        with self.assertRaisesRegex(ValueError, "sessions_per_event"):
            generate_benchmark(sessions_per_event=1)
        with self.assertRaisesRegex(ValueError, "exhibitors_per_event"):
            generate_benchmark(exhibitors_per_event=1)


if __name__ == "__main__":
    unittest.main()
