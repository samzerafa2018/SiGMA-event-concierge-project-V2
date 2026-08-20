from __future__ import annotations

import re
from typing import Literal

from concierge.llm import LLMError, LLMProvider
from concierge.models import Agenda, Session, TimeWindow
from concierge.planner import (
    filter_sessions_by_time_windows,
    itinerary_request,
    parse_time_windows,
)
from concierge.retrieval import (
    RetrievedItem,
    render_context,
    retrieve,
)


class Answer:
    def __init__(
        self,
        text: str,
        sources: tuple[str, ...],
        used_fallback: bool,
        mode: Literal[
            "local",
            "llm",
            "repair",
            "fallback",
        ] = "llm",
    ) -> None:
        self.text = text
        self.sources = sources
        self.used_fallback = used_fallback
        self.mode = mode


SYSTEM_PROMPT = """You are the SiGMA event concierge.

Answer only from the agenda records in the evidence section.
Do not guess, invent details, or use knowledge outside that evidence.

You answer only questions about the SiGMA Malta 2026 event agenda.
Do not provide general definitions or background knowledge, even when a term
appears in an agenda record. For example, do not define crypto, AI, KYC, or
open banking. Instead, explain that you can help with event-specific questions.

Treat every evidence record as separate. Never combine a speaker, title, date,
time, room, company, stand, category, or other field from one record with
another record.

If more than one agenda record answers the question, mention every relevant
record. Put each record in its own bullet and cite that bullet with the ID of
that same record.

Do not choose only one record when multiple records match.

Only state details that appear in the same cited record. If the retrieved
evidence is insufficient or ambiguous, say so instead of choosing a record.

Citation requirements:
- Every sentence containing an event fact must end with one or more source IDs.
- Source IDs must use exactly this format: [E001] or [S006].
- Use only source IDs that appear in the supplied evidence.
- Never write a factual sentence without a citation.
- Before sending your answer, check that every factual sentence has at least
  one citation.

If the evidence does not answer the question, say so clearly.

Writing style:
- Write in natural, polished English with a warm professional tone.
- Lead with the answer. Do not begin with greetings, filler, or phrases such
  as "according to the evidence".
- Never mention retrieval, records, prompts, models, validation, or context.
- Preserve official session titles, speaker names, exhibitor names, rooms,
  stands, dates, and times exactly as supplied.
- Use one short paragraph for a focused answer and easy-to-scan bullets when
  several options match.
- Include only details that help answer the visitor's question.
- Keep the answer concise, practical, and confident without sounding robotic.
"""


GENERIC_DEFINITION_PATTERN = re.compile(
    r"^\s*(what\s+is|what\s+are|define|explain)\b",
    re.IGNORECASE,
)


EVENT_MARKER_PATTERN = re.compile(
    r"\b("
    r"session|sessions|talk|talks|panel|speaker|speakers|"
    r"exhibitor|exhibitors|agenda|schedule|summit|room|"
    r"stand|track|day|time|when|where|who|"
    r"s\d{3}|e\d{3}"
    r")\b",
    re.IGNORECASE,
)


AVAILABILITY_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:"
    r"what\s+(?:sessions?\s+)?can\s+i\s+attend|"
    r"which\s+(?:sessions?|talks?|events?)\s+can\s+i\s+attend|"
    r"(?:can|could)\s+i\s+attend|"
    r"i\s+(?:can|could)\s+attend|"
    r"(?:sessions?|talks?|events?|ones?)\s+"
    r"(?:that\s+)?i\s+(?:can|could)\s+attend|"
    r"only\b.*\b(?:"
    r"attend|available|schedule|agenda|sessions?|talks?|events?"
    r")|"
    r"my\s+(?:schedule|availability)|"
    r"i(?:'m|\s+am)\s+free|"
    r"available\s+times|"
    r"(?:that\s+)?works?\s+for\s+me|"
    r"fit\s+(?:in|my\s+schedule)"
    r")\b",
    re.IGNORECASE,
)


COMPLETE_RESULT_PATTERN = re.compile(
    r"\b(?:"
    r"all|every|list|events|sessions|talks|exhibitors"
    r")\b",
    re.IGNORECASE,
)


SPEAKER_LIST_PATTERN = re.compile(
    r"\bwho\b.*\b(?:"
    r"speak|speaks|speaker|speakers|speaking|"
    r"present|presents|presenting|"
    r"host|hosts|hosting"
    r")\b",
    re.IGNORECASE,
)


DAY_ONLY_PATTERN = re.compile(
    r"^\s*(?:what(?:'s|\s+is)\s+on\s+)?"
    r"(?:"
    r"monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday"
    r")"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)


DIRECT_FACT_PATTERN = re.compile(
    r"\b(?:"
    r"when|where|who|time|room|location|stand|"
    r"speakers?|speaks?|features?|"
    r"which\s+(?:session|talk|exhibitor)|"
    r"what\s+(?:session|talk)|"
    r"what\s+is\s+(?:session\s+)?[se]\d{3}|"
    r"covers?|discusses?|offers?|showcases?|"
    r"provides?|providing"
    r")\b",
    re.IGNORECASE,
)


SUBJECTIVE_REQUEST_PATTERN = re.compile(
    r"\b(?:"
    r"recommend|suggest|should|best|ideal|"
    r"agenda|itinerary|plan|build|create"
    r")\b",
    re.IGNORECASE,
)


TOPIC_EXPLANATION_PATTERN = re.compile(
    r"\b(?:"
    r"covers?|discusses?|offers?|showcases?|"
    r"provides?|providing|what\s+is"
    r")\b",
    re.IGNORECASE,
)


AVAILABILITY_GENERIC_TERMS = {
    "after",
    "afternoon",
    "am",
    "and",
    "at",
    "attend",
    "availability",
    "available",
    "before",
    "between",
    "can",
    "could",
    "day",
    "during",
    "evening",
    "event",
    "events",
    "for",
    "friday",
    "from",
    "i",
    "in",
    "list",
    "me",
    "monday",
    "morning",
    "my",
    "on",
    "ones",
    "only",
    "please",
    "pm",
    "saturday",
    "session",
    "sessions",
    "show",
    "sunday",
    "talk",
    "talks",
    "that",
    "the",
    "these",
    "those",
    "thursday",
    "time",
    "to",
    "tuesday",
    "wednesday",
    "what",
    "which",
}


def _is_generic_definition_question(
    question: str,
) -> bool:
    return bool(
        GENERIC_DEFINITION_PATTERN.search(question)
        and not EVENT_MARKER_PATTERN.search(question)
    )


def _is_direct_fact_question(question: str) -> bool:
    """Return whether one retrieved record can answer without an LLM."""
    return bool(
        DIRECT_FACT_PATTERN.search(question)
        and not SUBJECTIVE_REQUEST_PATTERN.search(question)
    )


def _availability_follow_up_has_topic(question: str) -> bool:
    """Detect a topic such as AI in an availability follow-up."""
    tokens = re.findall(r"[a-z0-9]+", question.casefold())

    return any(
        token == "ai"
        or (
            token not in AVAILABILITY_GENERIC_TERMS
            and not token.isdigit()
        )
        for token in tokens
    )


def _citation_ids(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            re.findall(r"\[([SE]\d{3})\]", text)
        )
    )


def is_availability_follow_up(
    question: str,
) -> bool:
    """
    Return whether a question refers to the visitor's
    saved personal availability.
    """
    return bool(
        AVAILABILITY_FOLLOW_UP_PATTERN.search(question)
    )


def _is_availability_follow_up(
    question: str,
) -> bool:
    """
    Backward-compatible private alias used by existing
    application tests.
    """
    return is_availability_follow_up(question)


def _requires_complete_result(
    question: str,
) -> bool:
    """
    Identify questions that explicitly request a complete
    event-record list.
    """
    return bool(
        COMPLETE_RESULT_PATTERN.search(question)
        or DAY_ONLY_PATTERN.fullmatch(question)
        or SPEAKER_LIST_PATTERN.search(question)
    )


def _is_valid_response(
    response: str,
    allowed_sources: set[str],
    required_sources: set[str],
    required_fragments: tuple[str, ...] = (),
) -> bool:
    """
    Check that the answer cites supplied evidence, contains
    every required source, and includes requested key fields.
    """
    citations = set(_citation_ids(response))
    response_lower = response.casefold()

    return (
        bool(citations)
        and citations.issubset(allowed_sources)
        and required_sources.issubset(citations)
        and all(
            fragment.casefold() in response_lower
            for fragment in required_fragments
        )
    )


def _required_answer_fragments(
    agenda: Agenda,
    question: str,
    items: list[RetrievedItem],
) -> tuple[str, ...]:
    """
    Require requested time or location fields when retrieval
    identifies one unambiguous record.
    """
    if len(items) != 1:
        return ()

    item = items[0]
    question_lower = question.lower()

    if item.kind == "session":
        session = next(
            session
            for session in agenda.sessions
            if session.id == item.id
        )

        fragments: list[str] = []

        if re.search(
            r"\b(?:when|time)\b",
            question_lower,
        ):
            fragments.extend(
                (
                    session.start,
                    session.end,
                )
            )

        if re.search(
            r"\b(?:where|room|location)\b",
            question_lower,
        ):
            fragments.append(session.room)

        return tuple(fragments)

    exhibitor = next(
        exhibitor
        for exhibitor in agenda.exhibitors
        if exhibitor.id == item.id
    )

    if re.search(
        r"\b(?:where|stand|location)\b",
        question_lower,
    ):
        return (exhibitor.stand,)

    return ()


def _citation_repair_prompt(
    question: str,
    evidence: str,
    response: str,
    allowed_ids: str,
    required_sources: set[str],
    required_fragments: tuple[str, ...],
    itinerary_instruction: str,
) -> str:
    """
    Ask the model once to repair an invalid answer without
    introducing new evidence.
    """
    required_instruction = ""

    if required_sources:
        required_ids = ", ".join(
            f"[{source}]"
            for source in sorted(required_sources)
        )

        required_instruction = (
            "- Include and cite every required source: "
            f"{required_ids}\n"
        )

    field_instruction = ""

    if required_fragments:
        field_values = ", ".join(required_fragments)

        field_instruction = (
            "- Include these requested values from the evidence: "
            f"{field_values}\n"
        )

    return (
        "The draft below failed citation validation. Rewrite it "
        "as a concise final answer using only the supplied evidence. "
        "Do not add facts. Every factual sentence must end with "
        "one or more citations. Use natural, polished English and lead "
        "directly with the answer. Do not mention evidence, records, "
        "retrieval, models, prompts, or validation.\n\n"
        f"Visitor question:\n{question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Invalid draft:\n{response}\n\n"
        "Rules:\n"
        "- Use only the supplied evidence.\n"
        "- Cite every factual sentence exactly as [S001] or [E001].\n"
        f"{itinerary_instruction}"
        f"{required_instruction}"
        f"{field_instruction}"
        f"- The only valid citations are: {allowed_ids}\n"
        "Return only the corrected final answer."
    )


def _session_item(
    session: Session,
) -> RetrievedItem:
    """
    Create evidence for an eligible session when an
    availability follow-up has no useful topic terms.
    """
    return RetrievedItem(
        kind="session",
        id=session.id,
        score=0,
        text=(
            f"Session {session.id}\n"
            f"Title: {session.title}\n"
            f"Track: {session.track}\n"
            f"Day: {session.day}\n"
            f"Time: {session.start}-{session.end}\n"
            f"Room: {session.room}\n"
            f"Speakers: {', '.join(session.speakers)}\n"
            f"Abstract: {session.abstract}"
        ),
    )


def _format_itinerary_answer(
    sessions: tuple[Session, ...],
) -> Answer:
    """
    Return a safe deterministic itinerary when the model
    response fails validation.
    """
    bullets = [
        (
            f"- {session.start}–{session.end}: "
            f"{session.title} ({session.room}). "
            f"[{session.id}]"
        )
        for session in sessions
    ]

    return Answer(
        "Here’s a practical one-day agenda:\n\n"
        + "\n".join(bullets),
        tuple(
            session.id
            for session in sessions
        ),
        True,
        "fallback",
    )


def _format_evidence_answer(
    agenda: Agenda,
    items: list[RetrievedItem],
    *,
    used_fallback: bool = True,
) -> Answer:
    """
    Return a concise cited answer directly from validated
    agenda records.
    """
    records: list[str] = []

    sessions_by_id = {
        session.id: session
        for session in agenda.sessions
    }

    exhibitors_by_id = {
        exhibitor.id: exhibitor
        for exhibitor in agenda.exhibitors
    }

    for item in items:
        if item.kind == "session":
            session = sessions_by_id[item.id]

            records.append(
                f"- {session.title} — {session.day}, "
                f"{session.start}–{session.end}, "
                f"{session.room}. Speakers: "
                f"{', '.join(session.speakers)}. "
                f"[{session.id}]"
            )

        else:
            exhibitor = exhibitors_by_id[item.id]

            records.append(
                f"- {exhibitor.name} — "
                f"{exhibitor.category}, stand "
                f"{exhibitor.stand}. "
                f"{exhibitor.description} "
                f"[{exhibitor.id}]"
            )

    sources = tuple(item.id for item in items)
    answer_mode = "fallback" if used_fallback else "local"

    if len(records) == 1:
        return Answer(
            records[0].removeprefix("- "),
            sources,
            used_fallback,
            answer_mode,
        )

    item_kinds = {item.kind for item in items}

    if item_kinds == {"session"}:
        introduction = (
            "These sessions are the strongest matches:"
            if used_fallback
            else "These sessions match:"
        )
    elif item_kinds == {"exhibitor"}:
        introduction = (
            "These exhibitors are the strongest matches:"
            if used_fallback
            else "These exhibitors match:"
        )
    else:
        introduction = (
            "These agenda options are the strongest matches:"
            if used_fallback
            else "These agenda options match:"
        )

    return Answer(
        introduction + "\n\n" + "\n".join(records),
        sources,
        used_fallback,
        answer_mode,
    )


def _format_direct_answer(
    agenda: Agenda,
    item: RetrievedItem,
    question: str,
) -> Answer:
    """Answer an unambiguous factual lookup without calling the model."""
    question_lower = question.casefold()

    if item.kind == "session":
        session = next(
            session
            for session in agenda.sessions
            if session.id == item.id
        )
        sentences = [
            (
                f"**{session.title}** takes place on {session.day} "
                f"from {session.start}–{session.end} in "
                f"{session.room}. [{session.id}]"
            )
        ]

        if re.search(
            r"\b(?:who|speaker|speakers|speaks|features)\b",
            question_lower,
        ):
            sentences.append(
                f"Speakers: {', '.join(session.speakers)}. "
                f"[{session.id}]"
            )

        if TOPIC_EXPLANATION_PATTERN.search(question):
            sentences.append(
                f"{session.abstract} [{session.id}]"
            )

    else:
        exhibitor = next(
            exhibitor
            for exhibitor in agenda.exhibitors
            if exhibitor.id == item.id
        )
        sentences = [
            (
                f"**{exhibitor.name}** is in the "
                f"{exhibitor.category} category at stand "
                f"{exhibitor.stand}. [{exhibitor.id}]"
            )
        ]

        if TOPIC_EXPLANATION_PATTERN.search(question):
            sentences.append(
                f"{exhibitor.description} [{exhibitor.id}]"
            )

    return Answer(
        " ".join(sentences),
        (item.id,),
        False,
        "local",
    )


class GroundedConcierge:
    def __init__(
        self,
        agenda: Agenda,
        generator: LLMProvider,
        limit: int = 100,
    ) -> None:
        self.agenda = agenda
        self.generator = generator
        self.limit = limit

    def answer(
        self,
        question: str,
        time_windows: (
            tuple[TimeWindow, ...] | None
        ) = None,
        source_scope: tuple[str, ...] | None = None,
    ) -> Answer:
        if _is_generic_definition_question(question):
            return Answer(
                "I can help with the SiGMA Malta 2026 agenda, "
                "but I do not provide general definitions. "
                "Ask about a specific session, speaker, exhibitor, "
                "time, room, or stand.",
                (),
                False,
                "local",
            )

        if time_windows is None:
            time_windows = parse_time_windows(
                self.agenda,
                question,
            )

        scoped_agenda = self.agenda

        if source_scope:
            allowed_source_ids = set(source_scope)
            scoped_agenda = self.agenda.model_copy(
                update={
                    "sessions": [
                        session
                        for session in self.agenda.sessions
                        if session.id in allowed_source_ids
                    ],
                    "exhibitors": [
                        exhibitor
                        for exhibitor in self.agenda.exhibitors
                        if exhibitor.id in allowed_source_ids
                    ],
                }
            )

        eligible_sessions: list[Session] | None = None

        if time_windows:
            eligible_sessions = (
                filter_sessions_by_time_windows(
                    scoped_agenda,
                    time_windows,
                )
            )

            if not eligible_sessions:
                return Answer(
                    "I could not find any agenda sessions that "
                    "fit the stated availability and "
                    "unavailability windows.",
                    (),
                    False,
                    "local",
                )

            # Apply deterministic time constraints before
            # retrieval. The model never receives sessions
            # outside the visitor's availability.
            scoped_agenda = self.agenda.model_copy(
                update={
                    "sessions": eligible_sessions,
                }
            )

        itinerary = itinerary_request(
            scoped_agenda,
            question,
        )

        if itinerary is not None:
            items = [
                _session_item(session)
                for session in itinerary.sessions
            ]

        else:
            items = retrieve(
                scoped_agenda,
                question,
                limit=self.limit,
            )

        if (
            eligible_sessions is not None
            and (
                not items
                or (
                    _is_availability_follow_up(question)
                    and not _availability_follow_up_has_topic(
                        question
                    )
                )
            )
        ):
            items = [
                _session_item(session)
                for session in eligible_sessions
            ]

        if not items:
            return Answer(
                "I could not find matching information "
                "in the event agenda. Try a session title, speaker "
                "name, exhibitor name, topic, day, or time.",
                (),
                False,
                "local",
            )

        if (
            itinerary is None
            and len(items) == 1
            and _is_direct_fact_question(question)
        ):
            return _format_direct_answer(
                self.agenda,
                items[0],
                question,
            )

        requires_complete_result = (
            _requires_complete_result(question)
            or (
                eligible_sessions is not None
                and _is_availability_follow_up(question)
            )
        )

        if (
            itinerary is None
            and requires_complete_result
            and not _is_availability_follow_up(question)
            and not SUBJECTIVE_REQUEST_PATTERN.search(question)
        ):
            return _format_evidence_answer(
                self.agenda,
                items,
                used_fallback=False,
            )

        required_fragments = (
            _required_answer_fragments(
                self.agenda,
                question,
                items,
            )
        )

        allowed_ids = ", ".join(
            f"[{item.id}]"
            for item in items
        )

        itinerary_instruction = ""

        if itinerary is not None:
            itinerary_instruction = (
                "- The evidence is a Python-validated "
                "one-day itinerary. Mention every listed "
                "session in chronological order, and do not "
                "add sessions or change the day or times.\n"
                "- Return only itinerary bullets; do not "
                "speculate about omitted sessions.\n"
            )

        evidence = render_context(items)

        prompt = (
            f"Visitor question:\n{question}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Required response format:\n"
            "- Answer using only the evidence above.\n"
            "- Begin with the direct answer; do not use a greeting "
            "or generic preamble.\n"
            "- Write in natural, polished English with a warm, "
            "professional tone.\n"
            "- Do not mention evidence, records, retrieval, context, "
            "models, prompts, or validation.\n"
            "- Preserve official names, titles, dates, times, rooms, "
            "and stands exactly.\n"
            "- Every factual sentence must end with at "
            "least one citation.\n"
            "- Keep facts from separate source records "
            "separate.\n"
            "- If multiple records answer the question, "
            "use separate bullets.\n"
            "- If asked when, include both the date and "
            "start-end time.\n"
            "- If asked where, include the room or "
            "exhibitor stand.\n"
            f"{itinerary_instruction}"
            f"- The only valid citations are: "
            f"{allowed_ids}\n"
            "- Example: PayVault is at stand A12. [E001]\n"
            "- Do not omit the square brackets around "
            "citations.\n"
            "Write the final answer now."
        )

        allowed_sources = {
            item.id
            for item in items
        }

        required_sources = (
            {
                session.id
                for session in itinerary.sessions
            }
            if itinerary is not None
            else (
                {
                    item.id
                    for item in items
                }
                if requires_complete_result
                else set()
            )
        )

        try:
            response = self.generator.generate(
                SYSTEM_PROMPT,
                prompt,
            ).strip()

        except LLMError:
            if itinerary is not None:
                return _format_itinerary_answer(
                    itinerary.sessions
                )

            return _format_evidence_answer(
                self.agenda,
                items,
            )

        repair_attempted = False

        if not _is_valid_response(
            response,
            allowed_sources,
            required_sources,
            required_fragments,
        ):
            repair_attempted = True
            repair_prompt = _citation_repair_prompt(
                question=question,
                evidence=evidence,
                response=response,
                allowed_ids=allowed_ids,
                required_sources=required_sources,
                required_fragments=required_fragments,
                itinerary_instruction=(
                    itinerary_instruction
                ),
            )

            try:
                response = self.generator.generate(
                    SYSTEM_PROMPT,
                    repair_prompt,
                ).strip()

            except LLMError:
                if itinerary is not None:
                    return _format_itinerary_answer(
                        itinerary.sessions
                    )

                return _format_evidence_answer(
                    self.agenda,
                    items,
                )

        if not _is_valid_response(
            response,
            allowed_sources,
            required_sources,
            required_fragments,
        ):
            if itinerary is not None:
                return _format_itinerary_answer(
                    itinerary.sessions
                )

            if requires_complete_result:
                return _format_evidence_answer(
                    self.agenda,
                    items,
                )

            sources = tuple(
                item.id
                for item in items
            )

            source_text = ", ".join(
                f"[{source}]"
                for source in sources
            )

            return Answer(
                "I found relevant agenda records, but "
                "I could not produce a reliably cited "
                f"answer. Sources: {source_text}",
                sources,
                True,
                "fallback",
            )

        return Answer(
            response,
            _citation_ids(response),
            False,
            "repair" if repair_attempted else "llm",
        )
