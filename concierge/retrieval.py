from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from typing import Literal

from concierge.models import Agenda


ItemKind = Literal["session", "exhibitor"]


@dataclass(frozen=True)
class RetrievedItem:
    kind: ItemKind
    id: str
    score: int
    text: str


STOPWORDS = {
    "a",
    "an",
    "and",
    "all",
    "are",
    "at",
    "about",
    "discuss",
    "discusses",
    "feature",
    "features",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "related",
    "run",
    "running",
    "session",
    "sessions",
    "speaker",
    "speakers",
    "speak",
    "speaking",
    "talk",
    "talks",
    "that",
    "exhibitor",
    "exhibitors",
    "event",
    "events",
    "every",
    "everything",
    "give",
    "happen",
    "happening",
    "list",
    "me",
    "occur",
    "occurs",
    "programme",
    "program",
    "scheduled",
    "show",
    "the",
    "to",
    "what",
    "where",
    "which",
    "who",
    "with",
}

DAY_NAMES = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}

QUERY_CONTROL_TERMS = {
    "am",
    "after",
    "afternoon",
    "before",
    "between",
    "evening",
    "morning",
    "night",
    "pm",
    "earliest",
    "first",
    "last",
    "latest",
}


@dataclass(frozen=True)
class SessionConstraints:
    """Structured filters extracted from values present in the agenda."""

    day: str | None = None
    event_date: str | None = None
    start_after: time | None = None
    start_before: time | None = None
    active_at: time | None = None
    room: str | None = None
    track: str | None = None

    @property
    def any(self) -> bool:
        return any(value is not None for value in self.__dict__.values())


def _terms(query: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return set(tokens) - STOPWORDS


def _matches_time_of_day(query: str, start_time: str) -> bool:
    query_lower = query.lower()

    if "morning" in query_lower:
        return start_time < "12:00"

    if "afternoon" in query_lower:
        # In this agenda, the noon block is separate from the afternoon
        # programme, which begins at 13:00.
        return "13:00" <= start_time < "18:00"

    return True


def _requested_day_or_date(query: str) -> tuple[str | None, str | None]:
    """Extract an explicit weekday or ISO date constraint from a question."""
    query_lower = query.lower()
    day = next((name for name in DAY_NAMES if re.search(rf"\b{name}\b", query_lower)), None)
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", query)
    return day, date_match.group(0) if date_match else None


def _content_terms(query: str) -> set[str]:
    """Return terms that describe the requested event content, not constraints."""
    query_without_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", query)
    return {
        term
        for term in _terms(query_without_dates)
        if term not in DAY_NAMES
        and term not in QUERY_CONTROL_TERMS
        and not term.isdigit()
    }


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _parse_clock(hour_text: str, minute_text: str | None, meridiem: str | None) -> time | None:
    hour = int(hour_text)
    minute = int(minute_text or "0")

    if meridiem:
        if not 1 <= hour <= 12:
            return None
        if meridiem.lower() == "pm" and hour != 12:
            hour += 12
        elif meridiem.lower() == "am" and hour == 12:
            hour = 0

    if hour > 23 or minute > 59:
        return None

    return time(hour, minute)


def _clock_matches(query: str) -> list[tuple[str, time]]:
    pattern = re.compile(
        r"\b(?P<prefix>at|after|from|before|until|by)\s+"
        r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)?\b",
        re.IGNORECASE,
    )
    matches: list[tuple[str, time]] = []

    for match in pattern.finditer(query):
        parsed = _parse_clock(
            match.group("hour"), match.group("minute"), match.group("meridiem")
        )
        if parsed is not None:
            matches.append((match.group("prefix").lower(), parsed))

    return matches


def _matching_agenda_value(query: str, values: set[str]) -> str | None:
    normalised_query = f" {_normalise(query)} "
    matches = [
        value
        for value in values
        if f" {_normalise(value)} " in normalised_query
    ]
    return max(matches, key=len) if matches else None


def _requested_track(query: str, agenda: Agenda) -> str | None:
    tracks = {session.track for session in agenda.sessions}
    exact = _matching_agenda_value(query, tracks)
    if exact is not None:
        return exact

    if not re.search(r"\btrack\b", query, re.IGNORECASE):
        return None

    query_terms = _content_terms(query) - {"track"}
    candidates = [
        track
        for track in tracks
        if query_terms & set(_normalise(track).split())
    ]
    return candidates[0] if len(candidates) == 1 else None


def _session_constraints(agenda: Agenda, query: str) -> SessionConstraints:
    day, event_date = _requested_day_or_date(query)
    query_lower = query.lower()
    start_after: time | None = None
    start_before: time | None = None
    active_at: time | None = None

    if "morning" in query_lower:
        start_before = time(12, 0)
    elif "afternoon" in query_lower:
        start_after, start_before = time(13, 0), time(18, 0)
    elif "evening" in query_lower or "night" in query_lower:
        start_after = time(18, 0)

    clocks = _clock_matches(query)
    for prefix, value in clocks:
        if prefix in {"after", "from"}:
            start_after = value
        elif prefix in {"before", "until", "by"}:
            start_before = value
        elif prefix == "at":
            active_at = value

    range_match = re.search(
        r"\b(?:between|from)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
        r"\s+(?:and|to|-)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        query,
        re.IGNORECASE,
    )
    if range_match:
        first_meridiem = range_match.group(3) or range_match.group(6)
        start_after = _parse_clock(range_match.group(1), range_match.group(2), first_meridiem)
        start_before = _parse_clock(range_match.group(4), range_match.group(5), range_match.group(6))

    return SessionConstraints(
        day=day,
        event_date=event_date,
        start_after=start_after,
        start_before=start_before,
        active_at=active_at,
        room=_matching_agenda_value(query, {session.room for session in agenda.sessions}),
        track=_requested_track(query, agenda),
    )


def _matches_constraints(session, constraints: SessionConstraints) -> bool:
    if constraints.day and not session.day.lower().startswith(constraints.day):
        return False
    if constraints.event_date and constraints.event_date not in session.day:
        return False
    if constraints.room and session.room != constraints.room:
        return False
    if constraints.track and session.track != constraints.track:
        return False
    if constraints.active_at and not (
        session.start_time <= constraints.active_at < session.end_time
    ):
        return False
    if constraints.start_after and session.end_time <= constraints.start_after:
        return False
    if constraints.start_before and session.start_time >= constraints.start_before:
        return False
    return True


def _is_session_query(query: str) -> bool:
    return bool(
        re.search(
            r"\b(session|sessions|talk|talks|speaker|speakers|agenda|itinerary|schedule)\b",
            query,
            re.IGNORECASE,
        )
    )


def _is_exhibitor_query(query: str) -> bool:
    return bool(re.search(r"\b(exhibitor|exhibitors|stand|stands)\b", query, re.IGNORECASE))


def _matches_content(content_terms: set[str], searchable_text: str, item_id: str) -> bool:
    """Require a record to match a real topic/name/ID term when one is given."""
    if not content_terms:
        return True

    haystack_terms = set(re.findall(r"[a-z0-9]+", searchable_text.lower()))
    haystack_terms.add(item_id.lower())
    return bool(content_terms & haystack_terms)


def _session_datetime_key(session) -> tuple[str, str]:
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", session.day)

    if date_match is None:
        raise ValueError(
            f"Session {session.id} has no ISO-format date in its day field"
        )

    return date_match.group(0), session.start


def _session_text(session) -> str:
    return (
        f"Session {session.id}\n"
        f"Title: {session.title}\n"
        f"Track: {session.track}\n"
        f"Day: {session.day}\n"
        f"Time: {session.start}-{session.end}\n"
        f"Room: {session.room}\n"
        f"Speakers: {', '.join(session.speakers)}\n"
        f"Abstract: {session.abstract}"
    )


def _exhibitor_text(exhibitor) -> str:
    return (
        f"Exhibitor {exhibitor.id}\n"
        f"Name: {exhibitor.name}\n"
        f"Category: {exhibitor.category}\n"
        f"Stand: {exhibitor.stand}\n"
        f"Description: {exhibitor.description}"
    )


def _score(query: str, searchable_text: str, item_id: str) -> int:
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    terms = set(query_tokens) - STOPWORDS

    haystack_lower = searchable_text.lower()
    haystack_terms = set(re.findall(r"[a-z0-9]+", haystack_lower))
    score = len(terms & haystack_terms)

    meaningful_tokens = [
        token for token in query_tokens if token not in STOPWORDS
    ]

    for index in range(len(meaningful_tokens) - 1):
        phrase = " ".join(meaningful_tokens[index:index + 2])

        if phrase in haystack_lower:
            score += 4

    if item_id.lower() in terms:
        score += 10

    return score


def retrieve(agenda: Agenda, query: str, limit: int = 100) -> list[RetrievedItem]:
    """Return all matching agenda records, ordered by relevance."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    explicit_ids = tuple(
        dict.fromkeys(
            re.findall(
                r"\b[SE]\d{3}\b",
                query.upper(),
            )
        )
    )

    if explicit_ids:
        sessions_by_id = {
            session.id: session
            for session in agenda.sessions
        }
        exhibitors_by_id = {
            exhibitor.id: exhibitor
            for exhibitor in agenda.exhibitors
        }
        explicit_results: list[RetrievedItem] = []

        for item_id in explicit_ids:
            if item_id in sessions_by_id:
                session = sessions_by_id[item_id]
                explicit_results.append(
                    RetrievedItem(
                        kind="session",
                        id=session.id,
                        score=100,
                        text=_session_text(session),
                    )
                )

            elif item_id in exhibitors_by_id:
                exhibitor = exhibitors_by_id[item_id]
                explicit_results.append(
                    RetrievedItem(
                        kind="exhibitor",
                        id=exhibitor.id,
                        score=100,
                        text=_exhibitor_text(exhibitor),
                    )
                )

        return explicit_results[:limit]

    results: list[RetrievedItem] = []
    constraints = _session_constraints(agenda, query)
    content_terms = _content_terms(query)
    session_only = _is_session_query(query) and not _is_exhibitor_query(query)
    exhibitor_only = _is_exhibitor_query(query) and not _is_session_query(query)

    for session in agenda.sessions if not exhibitor_only else []:
        if not _matches_constraints(session, constraints):
            continue

        text = _session_text(session)

        content_text = " ".join(
            [
                session.id,
                session.title,
                session.track,
                session.room,
                " ".join(session.speakers),
                session.abstract,
            ]
        )

        if not _matches_content(content_terms, content_text, session.id):
            continue

        score = _score(query, text, session.id)

        if score or constraints.any:
            results.append(
                RetrievedItem("session", session.id, max(score, 1), text)
            )

    for exhibitor in agenda.exhibitors if not session_only else []:
        text = _exhibitor_text(exhibitor)

        if constraints.any:
            continue

        content_text = " ".join(
            [
                exhibitor.id,
                exhibitor.name,
                exhibitor.category,
                exhibitor.stand,
                exhibitor.description,
            ]
        )

        if not _matches_content(content_terms, content_text, exhibitor.id):
            continue

        score = _score(query, text, exhibitor.id)

        if score:
            results.append(RetrievedItem("exhibitor", exhibitor.id, score, text))

    query_lower = query.lower()
    existing_ids = {item.id for item in results}

    if "earliest" in query_lower or "first session" in query_lower:
        earliest_session = min(agenda.sessions, key=_session_datetime_key)

        if earliest_session.id not in existing_ids:
            results.append(
                RetrievedItem(
                    kind="session",
                    id=earliest_session.id,
                    score=100,
                    text=_session_text(earliest_session),
                )
            )
            existing_ids.add(earliest_session.id)

    if "latest" in query_lower or "last session" in query_lower:
        latest_session = max(agenda.sessions, key=_session_datetime_key)

        if latest_session.id not in existing_ids:
            results.append(
                RetrievedItem(
                    kind="session",
                    id=latest_session.id,
                    score=100,
                    text=_session_text(latest_session),
                )
            )

    sessions_by_id = {session.id: session for session in agenda.sessions}
    complete_constraint_list = constraints.any and (
        not content_terms
        or bool(re.search(r"\b(?:all|every|list)\b", query, re.IGNORECASE))
    )

    if complete_constraint_list:
        return sorted(
            results,
            key=lambda item: (
                0 if item.kind == "session" else 1,
                _session_datetime_key(sessions_by_id[item.id])
                if item.kind == "session"
                else ("", item.id),
            ),
        )[:limit]

    ranked_results = sorted(
        results,
        key=lambda item: (-item.score, item.id),
    )

    expects_multiple = bool(
        re.search(
            r"\b(?:all|every|list|events|sessions|talks|exhibitors)\b",
            query,
            re.IGNORECASE,
        )
    )

    # A score of 4 or more indicates a phrase or multi-term match. For a
    # focused singular question, discard records that matched only one
    # generic term. Broad list questions retain every relevant candidate.
    if (
        not expects_multiple
        and ranked_results
        and ranked_results[0].score >= 4
    ):
        strong_results = [
            item
            for item in ranked_results
            if item.score > 1
        ]

        if strong_results:
            ranked_results = strong_results

    return ranked_results[:limit]


def render_context(items: list[RetrievedItem]) -> str:
    """Format retrieved records for an LLM prompt, preserving source IDs."""
    if not items:
        return "No matching agenda records were found."

    return "\n\n".join(
        f"[{item.id}] ({item.kind}, relevance={item.score})\n{item.text}"
        for item in items
    )