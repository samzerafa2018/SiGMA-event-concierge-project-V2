import unittest

from concierge.answering import GroundedConcierge
from concierge.llm import LLMError
from concierge.models import load_agenda
from concierge.retrieval import retrieve


class FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        raise LLMError("The provider must not be called.")


class ReportedRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agenda = load_agenda()

    def test_kyc_time_lookup_uses_the_local_fast_path(self) -> None:
        question = "When is the KYC in 2026 session?"
        retrieved_ids = [
            item.id
            for item in retrieve(self.agenda, question)
        ]
        provider = FailingProvider()
        concierge = GroundedConcierge(self.agenda, provider)

        answer = concierge.answer(question)

        self.assertEqual(retrieved_ids, ["S005"])
        self.assertEqual(answer.mode, "local")
        self.assertEqual(answer.sources, ("S005",))
        self.assertIn("[S005]", answer.text)
        self.assertIn("13:00", answer.text)
        self.assertIn("13:45", answer.text)
        self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
