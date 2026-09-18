import unittest

from concierge.answering import GroundedConcierge
from concierge.llm import MockProvider
from concierge.models import load_agenda
from concierge.retrieval import (
    DeterministicRetriever,
    build_retriever,
    retrieve,
)


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.delegate = DeterministicRetriever()

    def retrieve(self, agenda, query: str, limit: int = 100):
        self.calls.append((query, limit))
        return self.delegate.retrieve(agenda, query, limit=limit)


class RetrieverInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agenda = load_agenda()

    def test_adapter_preserves_existing_results(self) -> None:
        adapter = DeterministicRetriever()
        questions = (
            "When and where is session S017?",
            "What AI-related talks are on Wednesday afternoon?",
            "Which exhibitors are related to AI?",
        )

        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    adapter.retrieve(self.agenda, question),
                    retrieve(self.agenda, question),
                )

    def test_builder_returns_deterministic_default(self) -> None:
        self.assertIsInstance(build_retriever(), DeterministicRetriever)
        self.assertIsInstance(
            build_retriever("  DETERMINISTIC  "),
            DeterministicRetriever,
        )

    def test_builder_rejects_unreleased_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "remains experimental"):
            build_retriever("adaptive")

    def test_grounded_concierge_uses_injected_retriever(self) -> None:
        recording = RecordingRetriever()
        concierge = GroundedConcierge(
            self.agenda,
            MockProvider("This response should not be needed."),
            limit=7,
            retriever=recording,
        )

        answer = concierge.answer("When and where is session S017?")

        self.assertEqual(recording.calls, [("When and where is session S017?", 7)])
        self.assertEqual(answer.sources, ("S017",))
        self.assertEqual(answer.mode, "local")


if __name__ == "__main__":
    unittest.main()
