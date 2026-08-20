from concierge.models import Agenda, Exhibitor, Session


def format_session(session: Session) -> str:
    """Convert one session into a compact, citation-friendly record."""
    speakers = "; ".join(session.speakers)

    return (
        f"[{session.id}]\n"
        f"Day: {session.day}\n"
        f"Time: {session.start}-{session.end}\n"
        f"Room: {session.room}\n"
        f"Track: {session.track}\n"
        f"Title: {session.title}\n"
        f"Speakers: {speakers}\n"
        f"Abstract: {session.abstract}"
    )


def format_exhibitor(exhibitor: Exhibitor) -> str:
    """Convert one exhibitor into a compact, citation-friendly record."""
    return (
        f"[{exhibitor.id}]\n"
        f"Name: {exhibitor.name}\n"
        f"Category: {exhibitor.category}\n"
        f"Stand: {exhibitor.stand}\n"
        f"Description: {exhibitor.description}"
    )


def build_event_context(agenda: Agenda) -> str:
    """
    Build the complete event context supplied to the LLM.

    Every session and exhibitor is included because this dataset is small.
    Stable IDs allow the model to cite exact source records, e.g. [S012].
    """
    session_records = "\n\n".join(
        format_session(session) for session in agenda.sessions
    )
    exhibitor_records = "\n\n".join(
        format_exhibitor(exhibitor) for exhibitor in agenda.exhibitors
    )

    return (
        f"EVENT\n"
        f"Name: {agenda.event.name}\n"
        f"Venue: {agenda.event.venue}\n"
        f"Dates: {agenda.event.dates}\n"
        f"Note: {agenda.event.note}\n\n"
        f"SESSIONS\n"
        f"{session_records}\n\n"
        f"EXHIBITORS\n"
        f"{exhibitor_records}"
    )