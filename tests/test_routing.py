import unittest

import numpy as np

from experiments.benchmark_retrieval import LexicalIndex
from experiments.hybrid_retrieval import VectorIndex
from experiments.routing import QueryRouter


class RoutingEncoder:
    model_id = "test/routing-encoder"

    TOPIC_TERMS = (
        ("ai", "agent", "language model", "personalisation", "neural"),
        ("payment", "payout", "bank", "money", "cash"),
        ("regulation", "identity", "compliance", "customer really"),
        ("affiliate", "marketing", "advert", "campaign", "sale"),
        ("esports", "player", "tournament", "game", "supporter"),
    )

    @classmethod
    def _vector(cls, text: str) -> list[float]:
        lowered = text.lower()
        scores = [
            float(sum(term in lowered for term in terms))
            for terms in cls.TOPIC_TERMS
        ]
        if not any(scores):
            scores[0] = 1.0
        return scores

    def encode_documents(self, texts):
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            {
                "event_id": "EVT001",
                "event_name": "SiGMA Malta 2027",
                "city": "Valletta",
                "year": 2027,
            },
            {
                "event_id": "EVT002",
                "event_name": "SiGMA Europe 2027",
                "city": "Rome",
                "year": 2027,
            },
        ]
        self.encoder = RoutingEncoder()
        self.router = QueryRouter(self.events, self.encoder)

    def test_explicit_event_and_exact_focus_are_deterministic(self) -> None:
        route = self.router.route(
            "Which Instant payouts talks are at SiGMA Malta 2027?"
        )

        self.assertEqual(route.event_id, "EVT001")
        self.assertEqual(route.topic, "Payments & Fintech")
        self.assertEqual(route.event_source, "explicit_name")
        self.assertEqual(route.topic_source, "exact_focus")

    def test_selected_event_context_and_semantic_topic_are_supported(self) -> None:
        route = self.router.route(
            "How can software confirm who a customer really is?",
            selected_event_id="EVT002",
        )

        self.assertEqual(route.event_id, "EVT002")
        self.assertEqual(route.topic, "Regulation & Compliance")
        self.assertEqual(route.event_source, "conversation_context")
        self.assertEqual(route.topic_source, "semantic")

    def test_missing_event_context_is_left_unresolved(self) -> None:
        route = self.router.route("Which talks cover safer language models?")

        self.assertIsNone(route.event_id)
        self.assertEqual(route.event_source, "unresolved")

    def test_unknown_selected_event_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown selected event"):
            self.router.route("Show payment talks", selected_event_id="UNKNOWN")

    def test_metadata_topic_filter_is_applied_to_both_indexes(self) -> None:
        records = [
            {
                "record_id": "EVT001-S0001",
                "event_id": "EVT001",
                "event_name": "SiGMA Malta 2027",
                "kind": "session",
                "topic": "Payments & Fintech",
                "title": "Instant payouts",
                "description": "Fast payment products",
                "speakers": [],
            },
            {
                "record_id": "EVT001-S0002",
                "event_id": "EVT001",
                "event_name": "SiGMA Malta 2027",
                "kind": "session",
                "topic": "Regulation & Compliance",
                "title": "Payment advertising rules",
                "description": "Compliance discussion",
                "speakers": [],
            },
        ]
        embeddings = self.encoder.encode_documents(
            [f"{record['title']} {record['description']}" for record in records]
        )
        lexical = LexicalIndex(records)
        vector = VectorIndex(records, embeddings, self.encoder)

        lexical_results = lexical.search(
            "payment",
            event_id="EVT001",
            kind="session",
            topic="Payments & Fintech",
        )
        vector_results = vector.search(
            "payment",
            event_id="EVT001",
            kind="session",
            topic="Payments & Fintech",
        )

        self.assertEqual([result.record_id for result in lexical_results], ["EVT001-S0001"])
        self.assertEqual([result.record_id for result in vector_results], ["EVT001-S0001"])


if __name__ == "__main__":
    unittest.main()
