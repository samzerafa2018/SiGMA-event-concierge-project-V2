import unittest

from concierge.answering import GroundedConcierge
from concierge.llm import LLMError, MockProvider
from concierge.models import load_agenda
from concierge.planner import parse_time_windows


class SequenceProvider:
    """Test provider that returns one response per model call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_prompt)
        return self.responses.pop(0)


class FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        raise LLMError("Timed out")


class GroundedAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agenda = load_agenda()

    def test_valid_citation_is_accepted(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider(
                "- Karl Mifsud speaks in Instant Payouts. [S006]\n"
                "- Karl Mifsud also speaks in Payments Panel. [S017]"
            ),
        )

        answer = concierge.answer("Which session features Karl Mifsud?")

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.sources, ("S006", "S017"))

    def test_unknown_citation_uses_fallback(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider(
                "Karl Mifsud speaks in an unlisted session. [S999]"
            ),
        )

        answer = concierge.answer("Which session features Karl Mifsud?")

        self.assertTrue(answer.used_fallback)
        self.assertIn("reliably cited answer", answer.text)

    def test_missing_citation_uses_fallback(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider(
                "Karl Mifsud speaks in the Payments Panel session."
            ),
        )

        answer = concierge.answer("Which session features Karl Mifsud?")

        self.assertTrue(answer.used_fallback)
        self.assertIn("reliably cited answer", answer.text)

    def test_invalid_citations_are_repaired_once_before_fallback(self) -> None:
        provider = SequenceProvider(
            [
                "Karl Mifsud speaks in the Payments Panel session.",
                "Karl Mifsud speaks in Instant Payouts. [S006]\n"
                "He also speaks in Payments Panel. [S017]",
            ]
        )
        concierge = GroundedConcierge(agenda=self.agenda, generator=provider)

        answer = concierge.answer(
            "Recommend the sessions featuring Karl Mifsud."
        )

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.sources, ("S006", "S017"))
        self.assertEqual(len(provider.prompts), 2)
        self.assertIn("failed citation validation", provider.prompts[1])

    def test_list_question_repair_includes_every_retrieved_record(self) -> None:
        provider = SequenceProvider(
            [
                "- Agentic AI is at 13:00. [S014]\n"
                "- The Open-Weight Models workshop is at 14:00. [S015]",
                "- Agentic AI is at 13:00. [S014]\n"
                "- The Open-Weight Models workshop is at 14:00. [S015]\n"
                "- CTOs Off the Record is at 17:00. [S033]",
            ]
        )
        concierge = GroundedConcierge(agenda=self.agenda, generator=provider)

        answer = concierge.answer(
            "Recommend all AI-related talks on Wednesday afternoon."
        )

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.sources, ("S014", "S015", "S033"))
        self.assertEqual(len(provider.prompts), 2)

    def test_list_question_uses_verified_summary_after_failed_repair(self) -> None:
        provider = SequenceProvider([
            "- Agentic AI is at 13:00. [S014]",
            "- Agentic AI is at 13:00. [S014]",
        ])
        concierge = GroundedConcierge(agenda=self.agenda, generator=provider)

        answer = concierge.answer(
            "Recommend all AI-related talks on Wednesday afternoon."
        )

        self.assertTrue(answer.used_fallback)
        self.assertEqual(answer.sources, ("S014", "S015", "S033"))
        self.assertIn("[S033]", answer.text)

    def test_itinerary_returns_verified_plan_when_model_times_out(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=FailingProvider(),
        )

        answer = concierge.answer(
            "Create a one-day agenda for Tuesday focused on AI."
        )

        self.assertTrue(answer.used_fallback)
        self.assertEqual(answer.sources, ("S001", "S002", "S003", "S031"))

    def test_focused_question_uses_local_fast_path(self) -> None:
        provider = FailingProvider()
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=provider,
        )

        answer = concierge.answer(
            "When is session S017?"
        )

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.mode, "local")
        self.assertEqual(answer.sources, ("S017",))
        self.assertIn("[S017]", answer.text)
        self.assertIn("14:00", answer.text)
        self.assertIn("15:00", answer.text)
        self.assertIn("Harbour Room", answer.text)
        self.assertEqual(provider.calls, 0)

    def test_prompt_requests_direct_natural_language(self) -> None:
        provider = SequenceProvider(
            [
                "- Karl Mifsud appears in Instant Payouts. [S006]\n"
                "- Karl Mifsud also appears in Payments Panel. [S017]"
            ]
        )
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=provider,
        )

        answer = concierge.answer(
            "Recommend the sessions featuring Karl Mifsud."
        )

        self.assertFalse(answer.used_fallback)
        self.assertIn("natural, polished English", provider.system_prompts[0])
        self.assertIn("Begin with the direct answer", provider.prompts[0])
        self.assertIn("Do not mention evidence", provider.prompts[0])

    def test_exhibitor_fast_path_keeps_matching_description(self) -> None:
        provider = FailingProvider()
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=provider,
        )

        answer = concierge.answer(
            "Which exhibitor offers embedded AI personalisation APIs?"
        )

        self.assertEqual(answer.mode, "local")
        self.assertEqual(answer.sources, ("E008",))
        self.assertIn("SpinLogic", answer.text)
        self.assertIn("embedded AI personalisation", answer.text)
        self.assertIn("[E008]", answer.text)
        self.assertEqual(provider.calls, 0)

    def test_no_result_message_suggests_a_better_query(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider("This response must not be used."),
        )

        answer = concierge.answer(
            "zzzz_nonexistent_session_or_exhibitor_zzzz"
        )

        self.assertEqual(answer.sources, ())
        self.assertEqual(answer.mode, "local")
        self.assertIn("could not find", answer.text)
        self.assertIn("speaker name", answer.text)

    def test_generic_definition_returns_scope_response(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider("This model response must not be used."),
        )

        answer = concierge.answer("What is crypto?")

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.sources, ())
        self.assertIn("do not provide general definitions", answer.text)

    def test_event_specific_definition_is_not_blocked(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider(
                "Session S017 is Payments Panel: Open Banking Meets iGaming. "
                "[S017]"
            ),
        )

        answer = concierge.answer("What is session S017?")

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.sources, ("S017",))

    def test_itinerary_uses_only_planned_non_overlapping_sessions(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider(
                "- Agentic AI is at 13:00. [S014]\n"
                "- The Open-Weight Models workshop is at 14:00. [S015]\n"
                "- CTOs Off the Record is at 17:00 and invite-only. [S033]"
            ),
        )

        windows = parse_time_windows(
            self.agenda,
            "I can only attend on Wednesday from 13:00 to 18:00.",
        )
        answer = concierge.answer(
            "Build me a 1-day agenda focused on emerging tech.",
            time_windows=windows,
        )

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.sources, ("S014", "S015", "S033"))

    def test_availability_follow_up_keeps_requested_ai_topic(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=FailingProvider(),
        )
        windows = parse_time_windows(
            self.agenda,
            "I can only attend on Wednesday from 13:00 to 18:00.",
        )

        answer = concierge.answer(
            "Which AI sessions can I attend?",
            time_windows=windows,
        )

        self.assertEqual(answer.sources, ("S014", "S015", "S033"))
        self.assertNotIn("[S016]", answer.text)
        self.assertNotIn("[S017]", answer.text)

    def test_reference_follow_up_uses_only_previous_sources(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=FailingProvider(),
        )

        answer = concierge.answer(
            "Build an agenda using only those sessions.",
            source_scope=("S014", "S015", "S033"),
        )

        self.assertEqual(answer.sources, ("S014", "S015", "S033"))
        self.assertNotIn("[S017]", answer.text)

    def test_itinerary_has_safe_fallback_when_model_citations_fail(self) -> None:
        concierge = GroundedConcierge(
            agenda=self.agenda,
            generator=MockProvider("This response has no citations."),
        )

        answer = concierge.answer(
            "Create a one-day agenda for Tuesday focused on AI."
        )

        self.assertTrue(answer.used_fallback)
        self.assertEqual(answer.sources, ("S001", "S002", "S003", "S031"))
        self.assertIn("[S001]", answer.text)
        self.assertIn("[S031]", answer.text)


if __name__ == "__main__":
    unittest.main()
