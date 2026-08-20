import re
import unittest

from concierge.answering import GroundedConcierge
from concierge.models import load_agenda
from concierge.planner import (
    filter_sessions_by_time_windows,
    parse_time_windows,
)


class RecordingProvider:
    def __init__(self) -> None:
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.user_prompt = user_prompt

        source_ids = tuple(
            dict.fromkeys(
                re.findall(r"\[(S\d{3})\]", user_prompt)
            )
        )

        return "\n".join(
            f"- Eligible session {source_id}. [{source_id}]"
            for source_id in source_ids
        )


class ConversationAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agenda = load_agenda()

    def test_follow_up_reuses_only_attend_availability_before_llm(self) -> None:
        windows = parse_time_windows(
            self.agenda,
            "I can only attend on Wednesday from 13:00 to 18:00.",
        )

        self.assertEqual(len(windows), 1)

        window = windows[0]
        self.assertEqual(window.event_date.isoformat(), "2026-11-11")
        self.assertEqual(window.start.strftime("%H:%M"), "13:00")
        self.assertEqual(window.end.strftime("%H:%M"), "18:00")

        eligible_sessions = filter_sessions_by_time_windows(
            self.agenda,
            windows,
        )
        expected_sources = tuple(
            session.id
            for session in eligible_sessions
        )

        provider = RecordingProvider()
        concierge = GroundedConcierge(self.agenda, provider)

        answer = concierge.answer(
            "Only ones I can attend.",
            time_windows=windows,
        )

        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.sources, expected_sources)

        for source_id in expected_sources:
            self.assertIn(f"[{source_id}]", provider.user_prompt)

        self.assertNotIn("[S004]", provider.user_prompt)
        self.assertNotIn("[S008]", provider.user_prompt)


if __name__ == "__main__":
    unittest.main()