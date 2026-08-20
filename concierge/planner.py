from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, time

from concierge.models import Agenda, Session, TimeWindow


AVAILABILITY_PATTERN = re.compile(
    r"\b(?:(?P<kind>unavailable|available|free)|can\s+(?:only\s+)?attend)\b"
    r"\s+(?:on\s+)?"
    r"(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+(?P<date>\d{4}-\d{2}-\d{2}))?"
    r"(?:\s+from\s+"
    r"(?P<start>\d{1,2}:\d{2})"
    r"\s+to\s+"
    r"(?P<end>\d{1,2}:\d{2}))?\b",
    re.IGNORECASE,
)

DAY_TIME_RANGE_PATTERN = re.compile(
    r"\b(?:on\s+)?"
    r"(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+(?P<date>\d{4}-\d{2}-\d{2}))?"
    r"\s+(?:from\s+)?"
    r"(?P<start>\d{1,2}:\d{2})"
    r"\s+to\s+"
    r"(?P<end>\d{1,2}:\d{2})\b",
    re.IGNORECASE,
)

ITINERARY_PATTERN = re.compile(
    r"(?:"
    r"\b(?:1[-\s]?day|one[-\s]?day)\b.*\b(?:agenda|it[ie]nerary)\b|"
    r"\b(?:build|create|make|plan|give)\b.*\b(?:agenda|it[ie]nerary)\b|"
    r"\b(?:non[-\s]?overlapping|conflict[-\s]?free)\b.*"
    r"\b(?:agenda|it[ie]nerary)\b|"
    r"\b(?:agenda|it[ie]nerary)\b.*"
    r"\b(?:non[-\s]?overlapping|conflict[-\s]?free)\b"
    r")",
    re.IGNORECASE,
)

FOCUS_PATTERN = re.compile(
    r"\bfocused\s+on\s+(?P<topic>.+?)"
    r"(?=\s+(?:for|on)\s+(?:"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{4}-\d{2}-\d{2}"
    r")\b|[?.!,]|$)",
    re.IGNORECASE,
)

DAY_NAME_PATTERN = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
}


@dataclass(frozen=True)
class ItineraryPlan:
    event_date: date
    topic: str
    sessions: tuple[Session, ...]


def _parse_time(value: str) -> time:
    hour_text, minute_text = value.split(":")
    return time(hour=int(hour_text), minute=int(minute_text))


def _event_dates(agenda: Agenda) -> set[date]:
    return {session.event_date for session in agenda.sessions}


def _resolve_date(
    agenda: Agenda,
    day_name: str,
    explicit_date: str | None,
) -> date | None:
    event_dates = _event_dates(agenda)

    if explicit_date is not None:
        try:
            resolved_date = date.fromisoformat(explicit_date)
        except ValueError:
            return None

        if resolved_date not in event_dates:
            return None

        if resolved_date.strftime("%A").lower() != day_name.lower():
            return None

        return resolved_date

    matching_dates = [
        event_date
        for event_date in event_dates
        if event_date.strftime("%A").lower() == day_name.lower()
    ]

    if len(matching_dates) != 1:
        return None

    return matching_dates[0]


def _day_bounds(agenda: Agenda, event_date: date) -> tuple[time, time]:
    """Return the first start and final end time scheduled on an event day."""
    sessions = [
        session for session in agenda.sessions if session.event_date == event_date
    ]

    return (
        min(session.start_time for session in sessions),
        max(session.end_time for session in sessions),
    )


def parse_time_windows(agenda: Agenda, question: str) -> tuple[TimeWindow, ...]:
    """Parse visitor availability and unavailability time windows."""
    windows: list[TimeWindow] = []

    matched_spans: list[tuple[int, int]] = []

    for match in AVAILABILITY_PATTERN.finditer(question):
        event_date = _resolve_date(
            agenda=agenda,
            day_name=match.group("day"),
            explicit_date=match.group("date"),
        )

        if event_date is None:
            continue

        try:
            if match.group("start") is None:
                start, end = _day_bounds(agenda, event_date)
            else:
                start = _parse_time(match.group("start"))
                end = _parse_time(match.group("end"))

            window = TimeWindow(
                # "I can only attend ..." is an availability constraint even
                # though it does not use the literal word "available".
                kind=(
                    "unavailable"
                    if match.group("kind") == "unavailable"
                    else "available"
                ),
                event_date=event_date,
                start=start,
                end=end,
            )
        except ValueError:
            continue

        windows.append(window)
        matched_spans.append(match.span())

    # Support direct agenda requests such as "Wednesday 13:00 to 18:00"
    # even when they do not explicitly say "available".
    for match in DAY_TIME_RANGE_PATTERN.finditer(question):
        if any(
            match.start() < end and match.end() > start
            for start, end in matched_spans
        ):
            continue

        event_date = _resolve_date(
            agenda=agenda,
            day_name=match.group("day"),
            explicit_date=match.group("date"),
        )

        if event_date is None:
            continue

        try:
            windows.append(
                TimeWindow(
                    kind="available",
                    event_date=event_date,
                    start=_parse_time(match.group("start")),
                    end=_parse_time(match.group("end")),
                )
            )
        except ValueError:
            continue

    return tuple(windows)


def filter_sessions_by_time_windows(
    agenda: Agenda,
    windows: tuple[TimeWindow, ...],
) -> list[Session]:
    """Return every session permitted by the visitor's time windows."""
    available_windows = [
        window for window in windows if window.kind == "available"
    ]
    unavailable_windows = [
        window for window in windows if window.kind == "unavailable"
    ]

    eligible_sessions: list[Session] = []

    for session in agenda.sessions:
        if available_windows and not any(
            window.contains_session(session) for window in available_windows
        ):
            continue

        if any(window.overlaps_session(session) for window in unavailable_windows):
            continue

        eligible_sessions.append(session)

    return sorted(
        eligible_sessions,
        key=lambda session: (
            session.event_date,
            session.start_time,
            session.id,
        ),
    )


def extract_itinerary_topic(question: str) -> str | None:
    """Return an itinerary topic, or an empty string for a general plan."""
    if not ITINERARY_PATTERN.search(question):
        return None

    match = FOCUS_PATTERN.search(question)

    if match is None:
        return ""

    topic = match.group("topic").strip().lower()

    return topic or None


def _topic_terms(topic: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", topic.lower())
        if term not in TOPIC_STOPWORDS
    }


def _topic_score(session: Session, topic: str) -> int:
    searchable_text = (
        f"{session.title} {session.track} {session.abstract}"
    ).lower()

    topic_terms = _topic_terms(topic)

    if not topic_terms:
        return 0

    searchable_terms = set(re.findall(r"[a-z0-9]+", searchable_text))
    matched_terms = topic_terms & searchable_terms

    if not matched_terms:
        return 0

    if len(topic_terms) > 1 and not topic_terms.issubset(searchable_terms):
        return 0

    score = len(matched_terms)

    if topic.lower() in searchable_text:
        score += 10

    if topic.lower() in session.track.lower():
        score += 5

    return score


def _requested_itinerary_date(agenda: Agenda, question: str) -> date | None:
    event_dates = _event_dates(agenda)

    iso_match = ISO_DATE_PATTERN.search(question)

    if iso_match is not None:
        try:
            requested_date = date.fromisoformat(iso_match.group(0))
        except ValueError:
            return None

        if requested_date in event_dates:
            return requested_date

        return None

    day_match = DAY_NAME_PATTERN.search(question)

    if day_match is None:
        return None

    requested_day = day_match.group(0).lower()
    matching_dates = [
        event_date
        for event_date in event_dates
        if event_date.strftime("%A").lower() == requested_day
    ]

    if len(matching_dates) == 1:
        return matching_dates[0]

    return None


def _select_non_overlapping_sessions(
    sessions: list[Session],
    topic: str,
) -> list[Session]:
    """Choose a chronological, non-overlapping set of relevant sessions."""
    candidates = sorted(
        sessions,
        key=lambda session: (
            session.start_time,
            session.end_time,
            -_topic_score(session, topic),
            session.id,
        ),
    )

    selected: list[Session] = []

    for session in candidates:
        if not selected:
            selected.append(session)
            continue

        previous_session = selected[-1]

        if session.start_time >= previous_session.end_time:
            selected.append(session)
            continue

        if (
            _topic_score(session, topic) > _topic_score(previous_session, topic)
            and (
                len(selected) == 1
                or session.start_time >= selected[-2].end_time
            )
        ):
            selected[-1] = session

    return selected


def build_one_day_itinerary(
    agenda: Agenda,
    topic: str,
    requested_date: date | None = None,
) -> ItineraryPlan | None:
    """Build the strongest non-overlapping one-day itinerary for a topic."""
    candidate_sessions = [
        session
        for session in agenda.sessions
        if not topic or _topic_score(session, topic) > 0
    ]

    if requested_date is not None:
        candidate_sessions = [
            session
            for session in candidate_sessions
            if session.event_date == requested_date
        ]

    if not candidate_sessions:
        return None

    sessions_by_date: dict[date, list[Session]] = {}

    for session in candidate_sessions:
        sessions_by_date.setdefault(session.event_date, []).append(session)

    plans: list[ItineraryPlan] = []

    for event_date, sessions in sessions_by_date.items():
        selected_sessions = _select_non_overlapping_sessions(sessions, topic)

        if selected_sessions:
            plans.append(
                ItineraryPlan(
                    event_date=event_date,
                    topic=topic,
                    sessions=tuple(selected_sessions),
                )
            )

    if not plans:
        return None

    return max(
        plans,
        key=lambda plan: (
            sum(_topic_score(session, topic) for session in plan.sessions),
            len(plan.sessions),
            -plan.event_date.toordinal(),
        ),
    )


def itinerary_request(
    agenda: Agenda,
    question: str,
) -> ItineraryPlan | None:
    """Build an itinerary when the question contains a supported request."""
    topic = extract_itinerary_topic(question)

    if topic is None:
        return None

    requested_date = _requested_itinerary_date(agenda, question)

    return build_one_day_itinerary(
        agenda=agenda,
        topic=topic,
        requested_date=requested_date,
    )