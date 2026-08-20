import unittest

from concierge.models import load_agenda
from concierge.planner import (
    filter_sessions_by_time_windows,
    itinerary_request,
    parse_time_windows,
)


class AvailabilityPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agenda = load_agenda()

    def test_available_wednesday_afternoon_returns_all_eligible_sessions(self) -> None:
        question = (
            "I am unavailable on Tuesday from 09:00 to 12:00. "
            "I am available on Wednesday from 13:00 to 18:00. "
            "What sessions can I attend?"
        )

        windows = parse_time_windows(self.agenda, question)
        sessions = filter_sessions_by_time_windows(self.agenda, windows)
        session_ids = {session.id for session in sessions}

        self.assertEqual(len(windows), 2)
        self.assertIn("S014", session_ids)
        self.assertIn("S033", session_ids)

        for session in sessions:
            self.assertIn("Wednesday", session.day)
            self.assertGreaterEqual(session.start, "13:00")
            self.assertLessEqual(session.end, "18:00")

    def test_unavailable_window_excludes_overlapping_sessions(self) -> None:
        question = "I am unavailable on Tuesday from 09:00 to 12:00."

        windows = parse_time_windows(self.agenda, question)
        sessions = filter_sessions_by_time_windows(self.agenda, windows)

        for session in sessions:
            is_blocked_tuesday_session = (
                "Tuesday" in session.day
                and session.start < "12:00"
                and session.end > "09:00"
            )
            self.assertFalse(is_blocked_tuesday_session)

    def test_day_only_availability_uses_that_days_schedule_bounds(self) -> None:
        windows = parse_time_windows(
            self.agenda,
            "I am available on Wednesday.",
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].event_date.isoformat(), "2026-11-11")
        self.assertEqual(windows[0].start.strftime("%H:%M"), "08:00")
        self.assertEqual(windows[0].end.strftime("%H:%M"), "18:00")

    def test_direct_day_and_time_range_becomes_availability(self) -> None:
        windows = parse_time_windows(
            self.agenda,
            "Build me an agenda for Thursday 13:00 to 16:00 focused on AI.",
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].event_date.isoformat(), "2026-11-12")
        self.assertEqual(windows[0].start.strftime("%H:%M"), "13:00")
        self.assertEqual(windows[0].end.strftime("%H:%M"), "16:00")

    def test_emerging_tech_itinerary_is_one_day_and_non_overlapping(self) -> None:
        plan = itinerary_request(
            self.agenda,
            "Build me a 1-day agenda focused on emerging tech.",
        )

        self.assertIsNotNone(plan)

        assert plan is not None

        self.assertGreater(len(plan.sessions), 0)

        for session in plan.sessions:
            self.assertEqual(session.event_date, plan.event_date)

            searchable_text = (
                f"{session.title} {session.track} {session.abstract}"
            ).lower()

            self.assertIn("emerging", searchable_text)
            self.assertIn("tech", searchable_text)

        for previous, current in zip(plan.sessions, plan.sessions[1:]):
            self.assertLessEqual(previous.end_time, current.start_time)

    def test_itinerary_respects_requested_day(self) -> None:
        plan = itinerary_request(
            self.agenda,
            "Build me a 1-day agenda on Wednesday focused on emerging tech.",
        )

        self.assertIsNotNone(plan)

        assert plan is not None

        self.assertEqual(plan.event_date.strftime("%A"), "Wednesday")

    def test_focus_topic_stops_before_trailing_day_clause(self) -> None:
        plan = itinerary_request(
            self.agenda,
            "Build me a one-day itinerary focused on emerging tech for Wednesday.",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.event_date.strftime("%A"), "Wednesday")
        self.assertEqual(
            [session.id for session in plan.sessions],
            ["S012", "S013", "S036", "S014", "S015", "S033"],
        )

    def test_common_itinerary_misspelling_is_recognised(self) -> None:
        plan = itinerary_request(
            self.agenda,
            "Build me a one day itenerary focused on emerging tech for Wednesday.",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            [session.id for session in plan.sessions],
            ["S012", "S013", "S036", "S014", "S015", "S033"],
        )

    def test_general_non_overlapping_agenda_needs_no_focus_topic(self) -> None:
        windows = parse_time_windows(
            self.agenda,
            "I am available Tuesday from 10:00 to 13:00",
        )
        eligible = filter_sessions_by_time_windows(self.agenda, windows)
        scoped = self.agenda.model_copy(update={"sessions": eligible})

        plan = itinerary_request(
            scoped,
            "Build me a non-overlapping agenda from the sessions I can attend.",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            [session.id for session in plan.sessions],
            ["S002", "S003", "S035"],
        )


if __name__ == "__main__":
    unittest.main()