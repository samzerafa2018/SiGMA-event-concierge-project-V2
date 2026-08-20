from __future__ import annotations

import json
import re
from datetime import date, time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def _extract_event_date(day_value: str) -> date:
    match = ISO_DATE_PATTERN.search(day_value)

    if match is None:
        raise ValueError(
            f"Could not find an ISO date in day value: {day_value!r}"
        )

    return date.fromisoformat(match.group(0))


def _duplicate_values(values: list[str]) -> list[str]:
    """Return sorted values that occur more than once."""
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    return sorted(duplicates)


class EventInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    venue: str
    dates: str
    note: str


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^S\d{3}$")
    title: str
    track: str
    day: str
    start: str = Field(pattern=r"^\d{2}:\d{2}$")
    end: str = Field(pattern=r"^\d{2}:\d{2}$")
    room: str
    speakers: list[str]
    abstract: str

    @model_validator(mode="after")
    def validate_schedule(self) -> Session:
        """Validate the session date, clock times, and time order."""
        try:
            _extract_event_date(self.day)
        except ValueError as error:
            raise ValueError(
                "Session day must contain a valid ISO date."
            ) from error

        try:
            start_time = time.fromisoformat(self.start)
            end_time = time.fromisoformat(self.end)
        except ValueError as error:
            raise ValueError(
                "Session start and end must be valid 24-hour times."
            ) from error

        if end_time <= start_time:
            raise ValueError(
                "Session end must be later than its start."
            )

        return self

    @property
    def event_date(self) -> date:
        """Return the ISO date embedded in the agenda day field."""
        return _extract_event_date(self.day)

    @property
    def start_time(self) -> time:
        """Return the session start as a Python time object."""
        return time.fromisoformat(self.start)

    @property
    def end_time(self) -> time:
        """Return the session end as a Python time object."""
        return time.fromisoformat(self.end)


class TimeWindow(BaseModel):
    """A visitor availability or unavailability period for one event day."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["available", "unavailable"]
    event_date: date
    start: time
    end: time

    @model_validator(mode="after")
    def validate_time_order(self) -> TimeWindow:
        if self.end <= self.start:
            raise ValueError("Time window end must be later than its start.")

        return self

    def contains_session(self, session: Session) -> bool:
        """Return True when the full session fits inside this window."""
        return (
            session.event_date == self.event_date
            and session.start_time >= self.start
            and session.end_time <= self.end
        )

    def overlaps_session(self, session: Session) -> bool:
        """Return True when any portion of a session overlaps this window."""
        if session.event_date != self.event_date:
            return False

        return (
            session.start_time < self.end
            and session.end_time > self.start
        )


class Exhibitor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^E\d{3}$")
    name: str
    category: str
    stand: str
    description: str


class Agenda(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: EventInfo
    sessions: list[Session]
    exhibitors: list[Exhibitor]

    @model_validator(mode="after")
    def validate_unique_record_ids(self) -> Agenda:
        """Reject ambiguous duplicate session or exhibitor identifiers."""
        duplicate_session_ids = _duplicate_values(
            [session.id for session in self.sessions]
        )

        if duplicate_session_ids:
            raise ValueError(
                "Duplicate session IDs: "
                + ", ".join(duplicate_session_ids)
            )

        duplicate_exhibitor_ids = _duplicate_values(
            [exhibitor.id for exhibitor in self.exhibitors]
        )

        if duplicate_exhibitor_ids:
            raise ValueError(
                "Duplicate exhibitor IDs: "
                + ", ".join(duplicate_exhibitor_ids)
            )

        return self


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sigma_agenda.json"


def load_agenda(path: Path = DEFAULT_DATA_PATH) -> Agenda:
    """Load and validate the supplied SiGMA event dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Agenda file not found: {path}. "
            "Expected data/sigma_agenda.json in the project root."
        )

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Agenda file contains invalid JSON: {error}") from error

    try:
        return Agenda.model_validate(raw_data)
    except ValidationError as error:
        raise ValueError(f"Agenda data failed validation:\n{error}") from error