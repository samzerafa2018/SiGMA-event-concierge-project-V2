import unittest

from concierge.models import load_agenda
from concierge.retrieval import retrieve


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agenda = load_agenda()

    def test_exact_session_title_ranks_first(self) -> None:
        results = retrieve(
            self.agenda,
            "When is Payments Panel Open Banking Meets iGaming?",
            limit=3,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].id, "S017")

    def test_speaker_query_returns_both_sessions(self) -> None:
        results = retrieve(
            self.agenda,
            "Which sessions feature Karl Mifsud?",
            limit=6,
        )

        result_ids = {item.id for item in results}
        self.assertIn("S006", result_ids)
        self.assertIn("S017", result_ids)
    def test_afternoon_query_excludes_morning_sessions(self) -> None:
        results = retrieve(
            self.agenda,
            "What AI-related talks are on Wednesday afternoon?",
            limit=6,
        )

        result_ids = {item.id for item in results}

        self.assertEqual(result_ids, {"S014", "S015", "S033"})
        self.assertNotIn("S016", result_ids)
        self.assertNotIn("S017", result_ids)
        self.assertNotIn("S024", result_ids)
        self.assertNotIn("S036", result_ids)

    def test_wednesday_ai_query_returns_every_matching_ai_session(self) -> None:
        results = retrieve(
            self.agenda,
            "What AI-related talks are on Wednesday?",
        )

        result_ids = {item.id for item in results}
        expected_ids = {"S012", "S013", "S014", "S015", "S033", "S036"}

        self.assertTrue(expected_ids.issubset(result_ids))

    def test_focused_exhibitor_query_removes_weak_ai_matches(self) -> None:
        results = retrieve(
            self.agenda,
            "Which exhibitor offers embedded AI personalisation APIs?",
        )

        self.assertEqual([item.id for item in results], ["E008"])

    def test_plural_ai_exhibitor_query_preserves_multiple_results(self) -> None:
        results = retrieve(
            self.agenda,
            "Which exhibitors are related to AI?",
        )

        result_ids = {item.id for item in results}
        self.assertTrue({"E008", "E009", "E010", "E011", "E020"}.issubset(result_ids))

    def test_all_events_on_any_requested_day_are_returned(self) -> None:
        for day in ("Tuesday", "Wednesday", "Thursday"):
            with self.subTest(day=day):
                results = retrieve(
                    self.agenda,
                    f"Give me all the events that occur on {day}.",
                )
                expected = [
                    session.id
                    for session in sorted(
                        self.agenda.sessions,
                        key=lambda item: (item.event_date, item.start_time, item.id),
                    )
                    if session.day.startswith(day)
                ]

                self.assertEqual([item.id for item in results], expected)

    def test_day_only_question_returns_the_complete_day(self) -> None:
        results = retrieve(self.agenda, "Tuesday?")
        result_ids = {item.id for item in results}
        expected_ids = {
            session.id
            for session in self.agenda.sessions
            if session.day.startswith("Tuesday")
        }

        self.assertEqual(result_ids, expected_ids)

    def test_day_room_and_track_filters_are_combined(self) -> None:
        room_results = retrieve(
            self.agenda,
            "List every Tuesday session in Vista Hall.",
        )
        track_results = retrieve(
            self.agenda,
            "What is on the Payments & Fintech track on Wednesday?",
        )

        self.assertEqual(
            {item.id for item in room_results},
            {"S003", "S006", "S009"},
        )
        self.assertEqual({item.id for item in track_results}, {"S017"})

    def test_explicit_time_filters_use_session_overlap(self) -> None:
        at_results = retrieve(
            self.agenda,
            "Which sessions are running Wednesday at 14:30?",
        )
        after_results = retrieve(
            self.agenda,
            "Show Thursday sessions after 2 pm.",
        )

        self.assertEqual(
            {item.id for item in at_results},
            {"S015", "S017"},
        )
        self.assertEqual(
            {item.id for item in after_results},
            {"S025", "S029", "S030", "S040"},
        )

    def test_earliest_and_latest_query_returns_schedule_extremes(self) -> None:
        results = retrieve(
            self.agenda,
            "What are the earliest and latest sessions?",
            limit=6,
        )

        result_ids = {item.id for item in results}

        earliest_session = min(
            self.agenda.sessions,
            key=lambda session: (session.day.split()[-1], session.start),
        )
        latest_session = max(
            self.agenda.sessions,
            key=lambda session: (session.day.split()[-1], session.start),
        )

        self.assertIn(earliest_session.id, result_ids)
        self.assertIn(latest_session.id, result_ids)

    def test_no_matching_records_returns_empty_list(self) -> None:
        results = retrieve(
            self.agenda,
            "zzzz_nonexistent_session_or_exhibitor_zzzz",
        )

        self.assertEqual(results, [])
if __name__ == "__main__":
    unittest.main()