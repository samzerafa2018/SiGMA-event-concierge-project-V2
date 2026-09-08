import os
import re
from html import escape

import streamlit as st
from dotenv import load_dotenv

from concierge.answering import (
    GroundedConcierge,
    is_availability_follow_up,
)
from concierge.llm import (
    GeminiProvider,
    LLMError,
    OllamaProvider,
    RetryingProvider,
)
from concierge.models import Agenda, load_agenda
from concierge.planner import parse_time_windows


load_dotenv()

st.set_page_config(
    page_title="SiGMA Event Concierge",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)


QUESTION_PATTERN = re.compile(
    r"\?|"
    r"\b(?:"
    r"what|which|who|when|where|are|"
    r"show|give|find|list|tell|build|recommend|suggest"
    r")\b",
    re.IGNORECASE,
)


AUXILIARY_QUESTION_PATTERN = re.compile(
    r"\b(?:can|could|would)\s+(?:i|we|you)\b",
    re.IGNORECASE,
)


REFERENCE_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:"
    r"those|these|them|"
    r"the\s+(?:same|listed|mentioned|previous)\s+"
    r"(?:sessions?|talks?|events?|exhibitors?)|"
    r"using\s+only\s+(?:those|these|them)"
    r")\b",
    re.IGNORECASE,
)


THEME_CSS = """
<style>
    :root {
        --sigma-red: #ef233c;
        --sigma-red-dark: #b51028;
        --sigma-ink: #17191e;
        --sigma-bg: #f5f6f8;
        --sigma-panel: #ffffff;
        --sigma-panel-raised: #f0f2f5;
        --sigma-line: rgba(23, 25, 30, 0.13);
        --sigma-muted: #666c77;
        --sigma-white: #ffffff;
    }

    html,
    body,
    #root,
    .stApp,
    [class*="css"] {
        font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif;
        background-color: var(--sigma-bg) !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 78% -10%,
                rgba(239, 35, 60, 0.1),
                transparent 34rem
            ),
            linear-gradient(180deg, #ffffff 0%, #f5f6f8 48%, #ffffff 100%);
        color: var(--sigma-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.86);
        border-bottom: 1px solid var(--sigma-line);
        backdrop-filter: blur(16px);
    }

    [data-testid="stMain"] .block-container {
        max-width: 1040px;
        padding: 2rem 2.2rem 7rem;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--sigma-line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--sigma-line);
    }

    .sigma-wordmark {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin: 0.2rem 0 1.5rem;
    }

    .sigma-wordmark__mark {
        width: 0.78rem;
        height: 0.78rem;
        background: var(--sigma-red);
        transform: rotate(45deg);
        box-shadow: 0 0 18px rgba(239, 35, 60, 0.22);
    }

    .sigma-wordmark__name {
        color: var(--sigma-ink);
        font-size: 1.4rem;
        font-weight: 850;
        letter-spacing: -0.04em;
    }

    .sigma-wordmark__name span {
        color: var(--sigma-red);
        font-weight: 650;
    }

    .sidebar-copy {
        color: var(--sigma-muted);
        font-size: 0.9rem;
        line-height: 1.55;
        margin: -0.35rem 0 1.35rem;
    }

    .sidebar-label,
    .section-label {
        color: var(--sigma-muted);
        font-size: 0.69rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .sidebar-label {
        margin: 0.35rem 0 0.6rem;
    }

    .section-label {
        margin: 1.5rem 0 0.8rem;
    }

    .system-status {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-top: 1rem;
        padding: 0.72rem 0.85rem;
        color: #3f444d;
        background: #f7f8fa;
        border: 1px solid var(--sigma-line);
        border-radius: 0.55rem;
        font-size: 0.78rem;
        font-weight: 650;
    }

    .system-status__dot {
        width: 0.48rem;
        height: 0.48rem;
        border-radius: 50%;
        background: #22a35a;
        box-shadow: 0 0 10px rgba(34, 163, 90, 0.32);
    }

    .sigma-hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.6rem, 4vw, 3.1rem);
        background:
            linear-gradient(118deg, rgba(255, 255, 255, 0.98), rgba(246, 247, 249, 0.96)),
            #ffffff;
        border: 1px solid var(--sigma-line);
        border-radius: 1rem;
        box-shadow: 0 24px 65px rgba(23, 25, 30, 0.09);
    }

    .sigma-hero::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 0.28rem;
        background: var(--sigma-red);
    }

    .sigma-hero::after {
        content: "";
        position: absolute;
        width: 17rem;
        height: 17rem;
        top: -9rem;
        right: -6rem;
        border: 4rem solid rgba(239, 35, 60, 0.08);
        border-radius: 50%;
    }

    .sigma-hero__eyebrow {
        position: relative;
        z-index: 1;
        color: var(--sigma-red-dark);
        font-size: clamp(1.05rem, 2.2vw, 1.45rem);
        font-weight: 800;
        letter-spacing: 0.06em;
        line-height: 1.3;
        text-transform: uppercase;
    }

    .sigma-hero__copy {
        position: relative;
        z-index: 1;
        max-width: 650px;
        margin: 0.9rem 0 0;
        color: #555b65;
        font-size: 1rem;
        line-height: 1.65;
    }

    .sigma-stats {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.55rem;
    }

    .sigma-stat {
        min-width: 8.5rem;
        padding: 0.72rem 0.9rem;
        background: #ffffff;
        border: 1px solid var(--sigma-line);
        border-radius: 0.6rem;
    }

    .sigma-stat strong {
        display: block;
        color: var(--sigma-ink);
        font-size: 1.1rem;
        letter-spacing: -0.02em;
    }

    .sigma-stat span {
        color: var(--sigma-muted);
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .availability-banner {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin: 1rem 0 0;
        padding: 0.85rem 1rem;
        color: #363a42;
        background: #fff3f5;
        border: 1px solid rgba(239, 35, 60, 0.28);
        border-radius: 0.65rem;
        font-size: 0.85rem;
    }

    .availability-banner__label {
        flex: 0 0 auto;
        color: var(--sigma-red-dark);
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .welcome-card {
        margin: 1.2rem 0 0.2rem;
        padding: 1rem 1.1rem;
        color: #555b65;
        background: #ffffff;
        border: 1px solid var(--sigma-line);
        border-radius: 0.65rem;
        line-height: 1.55;
    }

    .verified-note {
        margin: 0.7rem 0;
        padding: 0.72rem 0.85rem;
        color: #24623d;
        background: #effaf3;
        border: 1px solid rgba(34, 163, 90, 0.24);
        border-radius: 0.55rem;
        font-size: 0.82rem;
    }

    [data-testid="stButton"] > button {
        min-height: 2.7rem;
        color: #30343b;
        background: #ffffff;
        border: 1px solid var(--sigma-line);
        border-radius: 0.55rem;
        font-weight: 650;
        transition: border-color 150ms ease, background 150ms ease,
            transform 150ms ease;
    }

    [data-testid="stButton"] > button:hover {
        color: var(--sigma-red-dark);
        background: #fff5f6;
        border-color: rgba(239, 35, 60, 0.72);
        transform: translateY(-1px);
    }

    [data-testid="stButton"] > button:focus:not(:active) {
        color: var(--sigma-red-dark);
        border-color: var(--sigma-red);
        box-shadow: 0 0 0 0.2rem rgba(239, 35, 60, 0.16);
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 0.85rem;
        padding: 1.05rem 1.1rem;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid var(--sigma-line);
        border-radius: 0.8rem;
        box-shadow: 0 8px 28px rgba(23, 25, 30, 0.055);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        line-height: 1.62;
    }

    [data-testid="stChatMessage"] strong {
        color: var(--sigma-ink);
    }

    [data-testid="stChatInput"] {
        background: #ffffff;
        border: 1px solid rgba(23, 25, 30, 0.2);
        border-radius: 0.75rem;
        box-shadow: 0 16px 40px rgba(23, 25, 30, 0.12);
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    .stBottomBlockContainer {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }

    div:has(> [data-testid="stChatInput"]),
    div:has(> div > [data-testid="stChatInput"]),
    div:has(> div > div > [data-testid="stChatInput"]) {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }

    [data-testid="stBottom"]::before {
        background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0),
            #ffffff 72%
        ) !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] > div,
    [data-testid="stChatInput"] textarea {
        color: var(--sigma-ink) !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #7a808b !important;
        opacity: 1;
    }

    [data-testid="stChatInput"] button {
        color: #343840 !important;
        background: #eceef2 !important;
        border-radius: 0.55rem !important;
    }

    [data-testid="stChatInput"] button:hover {
        color: #ffffff !important;
        background: var(--sigma-red) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(239, 35, 60, 0.72);
        box-shadow: 0 0 0 0.18rem rgba(239, 35, 60, 0.12);
    }

    [data-testid="stExpander"] {
        margin-top: 0.8rem;
        background: #ffffff;
        border: 1px solid var(--sigma-line);
        border-radius: 0.6rem;
    }

    [data-testid="stExpander"] summary:hover {
        color: var(--sigma-red-dark);
    }

    .stSpinner > div {
        border-top-color: var(--sigma-red) !important;
    }

    a {
        color: var(--sigma-red-dark) !important;
    }

    @media (max-width: 720px) {
        [data-testid="stMain"] .block-container {
            padding: 1.1rem 1rem 6rem;
        }

        .sigma-hero {
            padding: 1.55rem 1.25rem;
        }

        .sigma-stat {
            flex: 1 1 7.5rem;
            min-width: 0;
        }

        .availability-banner {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.25rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        [data-testid="stButton"] > button {
            transition: none;
        }
    }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_agenda() -> Agenda:
    return load_agenda()


def event_date_range(agenda: Agenda) -> str:
    """Return a compact event date label for the hero panel."""
    dates = sorted({session.event_date for session in agenda.sessions})

    if not dates:
        return agenda.event.dates

    first_date = dates[0]
    last_date = dates[-1]

    if first_date == last_date:
        return first_date.strftime("%d %b %Y").upper()

    if (
        first_date.month == last_date.month
        and first_date.year == last_date.year
    ):
        return (
            f"{first_date.day:02d}–{last_date.day:02d} "
            f"{first_date.strftime('%b %Y').upper()}"
        )

    return (
        f"{first_date.strftime('%d %b').upper()}–"
        f"{last_date.strftime('%d %b %Y').upper()}"
    )


def render_hero(agenda: Agenda) -> None:
    """Render the branded event overview without external image assets."""
    event_name = escape(agenda.event.name)
    venue = escape(agenda.event.venue)
    date_label = escape(event_date_range(agenda))
    session_count = len(agenda.sessions)
    exhibitor_count = len(agenda.exhibitors)

    st.markdown(
        f"""
        <section class="sigma-hero">
            <div class="sigma-hero__eyebrow">
                {event_name} · AI agenda assistant
            </div>
            <p class="sigma-hero__copy">
                Find the right talks, exhibitors and connections—then build a
                conflict-free plan grounded in the official event dataset.
            </p>
            <div class="sigma-stats" aria-label="Event overview">
                <div class="sigma-stat">
                    <strong>{date_label}</strong>
                    <span>Event dates</span>
                </div>
                <div class="sigma-stat">
                    <strong>{venue}</strong>
                    <span>Venue</span>
                </div>
                <div class="sigma-stat">
                    <strong>{session_count}</strong>
                    <span>Sessions</span>
                </div>
                <div class="sigma-stat">
                    <strong>{exhibitor_count}</strong>
                    <span>Exhibitors</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_llm_provider() -> RetryingProvider:
    """Build the configured provider; Gemini is the lightweight default."""
    provider_name = os.getenv(
        "LLM_PROVIDER",
        "gemini",
    ).strip().lower()

    if provider_name == "gemini":
        base_provider = GeminiProvider(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.7-flash",
            ),
            base_url=os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            ),
            timeout_seconds=int(
                os.getenv("GEMINI_TIMEOUT_SECONDS", "45")
            ),
            thinking_level=os.getenv(
                "GEMINI_THINKING_LEVEL",
                "low",
            ),
        )

    elif provider_name == "ollama":
        base_provider = OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ),
            timeout_seconds=int(
                os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")
            ),
        )

    else:
        raise LLMError(
            "LLM_PROVIDER must be either 'gemini' or 'ollama'."
        )

    return RetryingProvider(
        base_provider,
        max_attempts=int(
            os.getenv("LLM_MAX_ATTEMPTS", "2")
        ),
        delay_seconds=float(
            os.getenv("LLM_RETRY_DELAY_SECONDS", "0.5")
        ),
    )


@st.cache_resource
def get_concierge() -> GroundedConcierge:
    return GroundedConcierge(
        agenda=get_agenda(),
        generator=build_llm_provider(),
    )


def source_details(
    agenda: Agenda,
    source_id: str,
) -> str:
    for session in agenda.sessions:
        if session.id == source_id:
            speakers = ", ".join(session.speakers)

            return (
                f"**{session.id} — {session.title}**\n\n"
                f"**Track:** {session.track}  \n"
                f"**When:** {session.day}, "
                f"{session.start}–{session.end}  \n"
                f"**Room:** {session.room}  \n"
                f"**Speakers:** {speakers}  \n"
                f"**About:** {session.abstract}"
            )

    for exhibitor in agenda.exhibitors:
        if exhibitor.id == source_id:
            return (
                f"**{exhibitor.id} — {exhibitor.name}**\n\n"
                f"**Category:** {exhibitor.category}  \n"
                f"**Stand:** {exhibitor.stand}  \n"
                f"**About:** {exhibitor.description}"
            )

    return f"Source record `{source_id}` was not found."


def render_assistant_message(message: dict) -> None:
    st.markdown(message["text"])

    if message.get("used_fallback"):
        st.markdown(
            """
            <div class="verified-note">
                ✓ Verified agenda summary used to keep this answer accurate.
            </div>
            """,
            unsafe_allow_html=True,
        )

    sources = message.get("sources", [])

    if sources:
        with st.expander(
            f"Evidence used ({len(sources)} source(s))"
        ):
            agenda = get_agenda()

            for source_id in sources:
                st.markdown(
                    source_details(agenda, source_id)
                )
                st.divider()


def is_availability_only_message(
    question: str,
    new_windows: tuple,
) -> bool:
    """
    Identify an availability-setting message that does not
    also ask an agenda question.
    """
    return (
        bool(new_windows)
        and not QUESTION_PATTERN.search(question)
        and not AUXILIARY_QUESTION_PATTERN.search(question)
    )


def is_reference_follow_up(question: str) -> bool:
    """Return whether a message refers to the previous answer's sources."""
    return bool(REFERENCE_FOLLOW_UP_PATTERN.search(question))


def previous_answer_sources(messages: list[dict]) -> tuple[str, ...]:
    """Return the newest non-empty set of assistant sources."""
    for message in reversed(messages):
        if (
            message.get("role") == "assistant"
            and message.get("sources")
        ):
            return tuple(message["sources"])

    return ()


def availability_window_label(window) -> str:
    kind = (
        "Available"
        if window.kind == "available"
        else "Unavailable"
    )
    day_label = window.event_date.strftime("%A %d %b")

    return (
        f"{kind}: {day_label}, "
        f"{window.start.strftime('%H:%M')}"
        f"–{window.end.strftime('%H:%M')}"
    )


def availability_summary(windows: tuple) -> str:
    return " | ".join(
        availability_window_label(window)
        for window in windows
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "availability_windows" not in st.session_state:
    st.session_state.availability_windows = ()


EXAMPLE_QUESTIONS = (
    "Which sessions feature Karl Mifsud?",
    "What AI-related talks are on Tuesday?",
    "Which exhibitors are related to AI in gaming?",
    "When is Payments Panel: Open Banking Meets iGaming?",
)


with st.sidebar:
    st.markdown(
        """
        <div class="sigma-wordmark">
            <div class="sigma-wordmark__mark"></div>
            <div class="sigma-wordmark__name">SiGMA<span>/AI</span></div>
        </div>
        <p class="sidebar-copy">
            Your grounded concierge for sessions, speakers, exhibitors,
            rooms, times and personalised itineraries.
        </p>
        <div class="sidebar-label">Quick questions</div>
        """,
        unsafe_allow_html=True,
    )
    selected_question = None

    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):
            selected_question = example

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.availability_windows = ()
        st.rerun()

    if st.session_state.availability_windows:
        st.divider()
        st.markdown(
            '<div class="sidebar-label">Active availability</div>',
            unsafe_allow_html=True,
        )

        for window in st.session_state.availability_windows:
            st.caption(
                availability_window_label(window)
            )

        if st.button(
            "Clear availability",
            use_container_width=True,
        ):
            st.session_state.availability_windows = ()
            st.rerun()

    st.divider()
    st.markdown(
        """
        <div class="system-status">
            <span class="system-status__dot"></span>
            Grounding and citation checks active
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Every answer is limited to retrieved agenda records. "
        "Open Evidence used to inspect the sources."
    )


render_hero(get_agenda())

if st.session_state.availability_windows:
    active_availability = escape(
        availability_summary(
            st.session_state.availability_windows
        )
    )
    st.markdown(
        f"""
        <div class="availability-banner">
            <span class="availability-banner__label">Active availability</span>
            <span>{active_availability}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            Ask naturally—try a topic, speaker, exhibitor or the time you can
            attend. Every result includes inspectable agenda evidence.
        </div>
        <div class="section-label">Popular starting points</div>
        """,
        unsafe_allow_html=True,
    )

    starter_columns = st.columns(2)

    for index, example in enumerate(EXAMPLE_QUESTIONS):
        with starter_columns[index % 2]:
            if st.button(
                example,
                key=f"starter_{index}",
                use_container_width=True,
            ):
                selected_question = example

for message in st.session_state.messages:
    avatar = "🎟️" if message["role"] == "assistant" else "👤"

    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["text"])


typed_question = st.chat_input(
    "Ask about sessions, speakers, exhibitors, "
    "or your event day..."
)

question = selected_question or typed_question


if question and question.strip():
    question = question.strip()

    user_message = {
        "role": "user",
        "text": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🎟️"):
        try:
            new_windows = parse_time_windows(
                get_agenda(),
                question,
            )

            if new_windows:
                st.session_state.availability_windows = (
                    new_windows
                )
                windows_for_answer = new_windows

            elif (
                st.session_state.availability_windows
                and is_availability_follow_up(question)
            ):
                windows_for_answer = (
                    st.session_state.availability_windows
                )

            else:
                windows_for_answer = ()

            if is_availability_only_message(
                question,
                new_windows,
            ):
                assistant_message = {
                    "role": "assistant",
                    "text": (
                        "Availability saved: "
                        f"{availability_summary(new_windows)}. "
                        "Ask which sessions you can attend."
                    ),
                    "sources": [],
                    "used_fallback": False,
                }

            else:
                source_scope = (
                    previous_answer_sources(
                        st.session_state.messages
                    )
                    if is_reference_follow_up(question)
                    else ()
                )

                with st.spinner(
                    "Searching the event agenda..."
                ):
                    answer = get_concierge().answer(
                        question,
                        time_windows=windows_for_answer,
                        source_scope=source_scope,
                    )

                assistant_message = {
                    "role": "assistant",
                    "text": answer.text,
                    "sources": list(answer.sources),
                    "used_fallback": answer.used_fallback,
                }

            render_assistant_message(assistant_message)

            st.session_state.messages.append(
                assistant_message
            )

            if new_windows:
                st.rerun()

        except LLMError as error:
            error_message = str(error)
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": error_message,
                    "sources": [],
                    "used_fallback": False,
                }
            )

        except Exception as error:
            error_message = (
                "I could not load the event data or generate "
                "an answer. Please try again."
            )

            st.error(error_message)
            st.caption(str(error))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": error_message,
                    "sources": [],
                    "used_fallback": False,
                }
            )
