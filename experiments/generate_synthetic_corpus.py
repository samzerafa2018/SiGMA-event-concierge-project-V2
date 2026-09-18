from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260918
DEFAULT_EVENT_COUNT = 20
DEFAULT_SESSIONS_PER_EVENT = 200
DEFAULT_EXHIBITORS_PER_EVENT = 50

EVENT_LOCATIONS = (
    ("Malta", "Valletta"),
    ("Europe", "Rome"),
    ("Eurasia", "Istanbul"),
    ("Africa", "Cape Town"),
    ("Asia", "Manila"),
    ("Americas", "Sao Paulo"),
    ("Balkans", "Belgrade"),
    ("Nordics", "Stockholm"),
    ("Middle East", "Dubai"),
    ("Central Europe", "Prague"),
)

TOPICS: dict[str, dict[str, tuple[str, ...]]] = {
    "AI & Emerging Tech": {
        "focuses": (
            "Agentic assistants",
            "LLM guardrails",
            "Responsible personalisation",
            "Retrieval-augmented generation",
            "Fraud anomaly detection",
            "Open-weight models",
            "AI evaluation",
            "Synthetic data",
        ),
        "paraphrases": (
            "software agents that take actions",
            "safety controls for language models",
            "safer tailored player experiences",
            "grounding answers in trusted documents",
            "spotting unusual fraudulent behaviour",
            "running adaptable models privately",
            "measuring whether AI answers are reliable",
            "artificial records for safe model testing",
        ),
        "held_out_queries": (
            "autonomous tools that can complete tasks for users",
            "preventing generative systems from producing unsafe responses",
            "adapting experiences while avoiding harm to customers",
            "letting a chatbot consult source material before replying",
            "finding suspicious patterns that could signal abuse",
            "self-hosted neural networks whose parameters are available",
            "testing whether machine-generated responses are dependable",
            "fabricated examples that resemble real records without exposing customers",
        ),
    },
    "Payments & Fintech": {
        "focuses": (
            "Instant payouts",
            "Open banking",
            "Stablecoin settlement",
            "Payment orchestration",
            "Cross-border wallets",
            "Chargeback prevention",
            "Embedded finance",
            "Treasury automation",
        ),
        "paraphrases": (
            "reducing withdrawal waiting times",
            "account-to-account payment rails",
            "settling transactions with digital currency",
            "routing transactions across payment providers",
            "moving wallet funds between countries",
            "preventing disputed card transactions",
            "financial services inside other products",
            "automating cash and liquidity operations",
        ),
        "held_out_queries": (
            "getting winnings into a customer's account with minimal delay",
            "paying directly through a customer's bank account",
            "using blockchain-backed money to complete merchant transfers",
            "choosing the best processor for each purchase attempt",
            "holding and transferring funds across national boundaries",
            "reducing losses when cardholders challenge a purchase",
            "adding banking capabilities directly inside a digital service",
            "using software to manage company cash positions",
        ),
    },
    "Regulation & Compliance": {
        "focuses": (
            "KYC automation",
            "AML monitoring",
            "Advertising rules",
            "Player protection",
            "Privacy governance",
            "Risk-based licensing",
            "Audit readiness",
            "Responsible gaming controls",
        ),
        "paraphrases": (
            "automating customer identity checks",
            "detecting suspicious money movements",
            "complying with marketing restrictions",
            "reducing harm to vulnerable customers",
            "managing personal data lawfully",
            "licensing based on operator risk",
            "preparing evidence for compliance reviews",
            "tools that support safer gambling",
        ),
        "held_out_queries": (
            "confirming who a new customer really is with less manual work",
            "spotting financial activity that may involve criminal proceeds",
            "staying within legal limits when promoting betting products",
            "intervening before at-risk customers experience serious harm",
            "handling identifiable customer information under data law",
            "adjusting supervisory requirements to an operator's risk level",
            "organising proof before an external regulatory inspection",
            "product features that encourage healthier gambling behaviour",
        ),
    },
    "Affiliate Marketing": {
        "focuses": (
            "Attribution modelling",
            "Creator partnerships",
            "SEO resilience",
            "Conversion analytics",
            "Content localisation",
            "Brand safety",
            "First-party audiences",
            "Campaign automation",
        ),
        "paraphrases": (
            "crediting the channel that produced a conversion",
            "working with online creators",
            "maintaining search traffic after algorithm changes",
            "understanding why visitors become customers",
            "adapting content for regional audiences",
            "keeping promotions away from harmful content",
            "marketing with directly collected customer data",
            "automating repetitive advertising workflows",
        ),
        "held_out_queries": (
            "working out which touchpoint deserves credit for a sale",
            "commercial campaigns involving social media personalities",
            "protecting organic discovery when ranking systems change",
            "learning what persuades a website visitor to sign up",
            "rewriting promotions so they feel native to each market",
            "preventing adverts from appearing beside unsuitable material",
            "reaching customers using information collected without intermediaries",
            "letting software execute routine marketing tasks automatically",
        ),
    },
    "Esports & Player Experience": {
        "focuses": (
            "Match integrity",
            "Community moderation",
            "Live engagement",
            "Accessibility design",
            "Player retention",
            "Competitive data rights",
            "Safer streaming",
            "Fan personalisation",
        ),
        "paraphrases": (
            "detecting manipulation in competitive matches",
            "managing harmful behaviour in player communities",
            "keeping audiences involved during live play",
            "designing games for players with disabilities",
            "encouraging healthy long-term participation",
            "governing ownership of tournament information",
            "protecting viewers and creators on live platforms",
            "tailoring experiences to individual supporters",
        ),
        "held_out_queries": (
            "identifying cheating or manipulation in professional competition",
            "keeping abusive conduct out of online player groups",
            "maintaining viewer interest throughout a real-time tournament",
            "making games usable by people with different impairments",
            "giving players good reasons to return without exploiting them",
            "deciding who may collect and sell information from tournaments",
            "reducing harmful material during live online broadcasts",
            "showing each supporter content matched to their preferences",
        ),
    },
}

SESSION_FORMATS = (
    "From Pilot to Production",
    "Architecture and Trade-offs",
    "Lessons from Real Deployments",
    "Risk, Measurement and Controls",
    "Practical Workshop",
)

ROOMS = (
    "Main Stage",
    "Harbour Room",
    "Vista Hall",
    "Skyline Suite",
    "Innovation Lab",
    "Forum Theatre",
)

FIRST_NAMES = (
    "Ana",
    "Elena",
    "Karl",
    "Miriam",
    "Nadia",
    "Tomas",
    "Sofia",
    "Marc",
    "Ines",
    "Dmitri",
)

LAST_NAMES = (
    "Borg",
    "Costa",
    "Delia",
    "Grech",
    "Kovac",
    "Laurent",
    "Mifsud",
    "Ribeiro",
    "Tanaka",
    "Volkov",
)

COMPANY_SUFFIXES = (
    "Labs",
    "Systems",
    "Works",
    "Cloud",
    "Logic",
    "Forge",
    "Stack",
    "Dynamics",
)


def _event_details(index: int) -> dict[str, Any]:
    region, city = EVENT_LOCATIONS[(index - 1) % len(EVENT_LOCATIONS)]
    cycle = (index - 1) // len(EVENT_LOCATIONS)
    year = 2027 + cycle
    event_id = f"EVT{index:03d}"
    return {
        "event_id": event_id,
        "event_name": f"SiGMA {region} {year}",
        "city": city,
        "year": year,
        "start_date": date(year, 2 + ((index - 1) % 8), 10),
    }


def _speaker(rng: random.Random, organisation: str) -> str:
    return (
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)} "
        f"({rng.choice(('CTO', 'Founder', 'Research Lead', 'VP Product'))}, "
        f"{organisation})"
    )


def _company_name(topic_index: int, product_index: int) -> str:
    stems = (
        "Atlas",
        "Beacon",
        "Cobalt",
        "Delta",
        "Ember",
        "Flux",
        "Helix",
        "Ion",
        "Kinetic",
        "Lumina",
    )
    stem = stems[(topic_index * 2 + product_index) % len(stems)]
    suffix = COMPANY_SUFFIXES[(topic_index + product_index) % len(COMPANY_SUFFIXES)]
    return f"{stem}{suffix}"


def _session_record(
    event: dict[str, Any],
    session_number: int,
    rng: random.Random,
) -> dict[str, Any]:
    topic_names = tuple(TOPICS)
    topic_index = (session_number - 1) % len(topic_names)
    topic = topic_names[topic_index]
    position_in_topic = (session_number - 1) // len(topic_names)
    focuses = TOPICS[topic]["focuses"]
    focus_index = position_in_topic % len(focuses)
    format_index = (position_in_topic // len(focuses)) % len(SESSION_FORMATS)
    focus = focuses[focus_index]
    session_format = SESSION_FORMATS[format_index]
    title = f"{focus}: {session_format}"
    organisation = _company_name(topic_index, focus_index)
    day_offset = position_in_topic % 3
    start_hour = 9 + ((position_in_topic // 3) % 8)
    start_minute = 30 if position_in_topic % 2 else 0
    end_hour = start_hour + (1 if start_minute == 30 else 0)
    end_minute = 0 if start_minute == 30 else 45

    return {
        "record_id": f"{event['event_id']}-S{session_number:04d}",
        "kind": "session",
        "event_id": event["event_id"],
        "event_name": event["event_name"],
        "city": event["city"],
        "year": event["year"],
        "topic": topic,
        "title": title,
        "description": (
            f"A {session_format.lower()} session about {focus.lower()}, including "
            f"{TOPICS[topic]['paraphrases'][focus_index]}."
        ),
        "speakers": [_speaker(rng, organisation)],
        "date": (event["start_date"] + timedelta(days=day_offset)).isoformat(),
        "start": f"{start_hour:02d}:{start_minute:02d}",
        "end": f"{end_hour:02d}:{end_minute:02d}",
        "room": rng.choice(ROOMS),
    }


def _exhibitor_record(
    event: dict[str, Any],
    exhibitor_number: int,
) -> dict[str, Any]:
    topic_names = tuple(TOPICS)
    topic_index = (exhibitor_number - 1) % len(topic_names)
    topic = topic_names[topic_index]
    position_in_topic = (exhibitor_number - 1) // len(topic_names)
    focuses = TOPICS[topic]["focuses"]
    focus_index = position_in_topic % len(focuses)
    focus = focuses[focus_index]
    company = _company_name(topic_index, position_in_topic)

    return {
        "record_id": f"{event['event_id']}-E{exhibitor_number:04d}",
        "kind": "exhibitor",
        "event_id": event["event_id"],
        "event_name": event["event_name"],
        "city": event["city"],
        "year": event["year"],
        "topic": topic,
        "title": company,
        "description": (
            f"{company} provides {focus.lower()} products for teams interested in "
            f"{TOPICS[topic]['paraphrases'][focus_index]}."
        ),
        "stand": f"{chr(65 + topic_index)}-{position_in_topic + 1:03d}",
    }


def _queries_for_event(
    event: dict[str, Any],
    sessions: list[dict[str, Any]],
    exhibitors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    event_number = int(event["event_id"].removeprefix("EVT"))
    topics = tuple(TOPICS)
    topic = topics[(event_number - 1) % len(topics)]
    topic_sessions = [record for record in sessions if record["topic"] == topic]
    topic_exhibitors = [record for record in exhibitors if record["topic"] == topic]
    focuses = TOPICS[topic]["focuses"]
    available_focus_count = min(
        len(focuses), len(topic_sessions), len(topic_exhibitors)
    )
    focus_index = (event_number * 3 + 1) % available_focus_count
    focus = focuses[focus_index]
    target_session = next(
        record
        for record in topic_sessions
        if record["title"] == f"{focus}: {SESSION_FORMATS[0]}"
    )
    semantic_targets = [
        record["record_id"]
        for record in topic_sessions
        if record["title"].startswith(f"{focus}:")
    ]
    target_exhibitor = next(
        record
        for record in exhibitors
        if record["topic"] == topic
        and f"{focus.lower()} products" in record["description"]
    )
    held_out_intent = TOPICS[topic]["held_out_queries"][focus_index]

    query_specs = (
        (
            "exact_title",
            f"When is {target_session['title']} at {event['event_name']}?",
            [target_session["record_id"]],
            "session",
            None,
        ),
        (
            "semantic_paraphrase",
            (
                f"At {event['event_name']}, which talks cover "
                f"{held_out_intent}?"
            ),
            semantic_targets,
            "session",
            held_out_intent,
        ),
        (
            "cross_event_disambiguation",
            (
                f"Find the {target_session['title']} session specifically at "
                f"{event['event_name']}."
            ),
            [target_session["record_id"]],
            "session",
            None,
        ),
        (
            "exhibitor_semantic",
            (
                f"Which exhibitor at {event['event_name']} helps with "
                f"{held_out_intent}?"
            ),
            [target_exhibitor["record_id"]],
            "exhibitor",
            held_out_intent,
        ),
    )

    for offset, (cohort, text, expected_ids, kind, semantic_intent) in enumerate(
        query_specs, start=1
    ):
        query = {
            "query_id": f"{event['event_id']}-Q{offset:02d}",
            "cohort": cohort,
            "text": text,
            "event_id": event["event_id"],
            "topic": topic,
            "kind": kind,
            "expected_record_ids": expected_ids,
        }
        if semantic_intent is not None:
            query["semantic_intent"] = semantic_intent
        queries.append(query)

    return queries


def generate_benchmark(
    *,
    event_count: int = DEFAULT_EVENT_COUNT,
    sessions_per_event: int = DEFAULT_SESSIONS_PER_EVENT,
    exhibitors_per_event: int = DEFAULT_EXHIBITORS_PER_EVENT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create a deterministic multi-event retrieval benchmark in memory."""
    if event_count < 2:
        raise ValueError("event_count must be at least 2 to test event isolation")
    if sessions_per_event < len(TOPICS):
        raise ValueError(f"sessions_per_event must be at least {len(TOPICS)}")
    if exhibitors_per_event < len(TOPICS):
        raise ValueError(f"exhibitors_per_event must be at least {len(TOPICS)}")

    rng = random.Random(seed)
    events: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []

    for event_index in range(1, event_count + 1):
        event = _event_details(event_index)
        event_public = {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in event.items()
        }
        events.append(event_public)
        sessions = [
            _session_record(event, number, rng)
            for number in range(1, sessions_per_event + 1)
        ]
        exhibitors = [
            _exhibitor_record(event, number)
            for number in range(1, exhibitors_per_event + 1)
        ]
        records.extend(sessions)
        records.extend(exhibitors)
        queries.extend(_queries_for_event(event, sessions, exhibitors))

    benchmark = {
        "metadata": {
            "generator_version": "1.1",
            "seed": seed,
            "event_count": event_count,
            "sessions_per_event": sessions_per_event,
            "exhibitors_per_event": exhibitors_per_event,
            "record_count": len(records),
            "query_count": len(queries),
            "purpose": (
                "Synthetic scalability benchmark. No people, companies or events "
                "in this file are real."
            ),
        },
        "events": events,
        "records": records,
        "queries": queries,
    }
    validate_benchmark(benchmark)
    return benchmark


def validate_benchmark(benchmark: dict[str, Any]) -> None:
    """Reject malformed benchmarks before they are used for comparisons."""
    events = benchmark.get("events", [])
    records = benchmark.get("records", [])
    queries = benchmark.get("queries", [])
    event_ids = [event["event_id"] for event in events]
    record_ids = [record["record_id"] for record in records]

    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Synthetic event IDs must be unique")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Synthetic record IDs must be unique")

    known_events = set(event_ids)
    known_records = set(record_ids)
    records_by_id = {record["record_id"]: record for record in records}
    for record in records:
        if record["event_id"] not in known_events:
            raise ValueError(f"Record references unknown event: {record['record_id']}")
    for query in queries:
        if query["event_id"] not in known_events:
            raise ValueError(f"Query references unknown event: {query['query_id']}")
        missing = set(query["expected_record_ids"]) - known_records
        if missing:
            raise ValueError(
                f"Query {query['query_id']} references unknown records: {sorted(missing)}"
            )
        wrong_event = [
            record_id
            for record_id in query["expected_record_ids"]
            if not record_id.startswith(f"{query['event_id']}-")
        ]
        if wrong_event:
            raise ValueError(
                f"Query {query['query_id']} leaks targets from another event: {wrong_event}"
            )
        if query["cohort"] in {"semantic_paraphrase", "exhibitor_semantic"}:
            semantic_intent = query.get("semantic_intent", "").strip().lower()
            if not semantic_intent:
                raise ValueError(
                    f"Semantic query {query['query_id']} has no held-out intent"
                )
            for record_id in query["expected_record_ids"]:
                target = records_by_id[record_id]
                target_text = f"{target['title']} {target['description']}".lower()
                if semantic_intent in target_text:
                    raise ValueError(
                        f"Semantic intent for {query['query_id']} appears verbatim "
                        f"in target {record_id}"
                    )

    title_event_counts: dict[str, set[str]] = {}
    for record in records:
        if record["kind"] != "session":
            continue
        title_event_counts.setdefault(record["title"], set()).add(record["event_id"])
    if not any(len(events_for_title) > 1 for events_for_title in title_event_counts.values()):
        raise ValueError("Benchmark must include repeated session titles across events")

    cohorts = Counter(query["cohort"] for query in queries)
    required = {
        "exact_title",
        "semantic_paraphrase",
        "cross_event_disambiguation",
        "exhibitor_semantic",
    }
    if set(cohorts) != required:
        raise ValueError(f"Benchmark query cohorts must be exactly {sorted(required)}")


def write_benchmark(benchmark: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible multi-event retrieval benchmark."
    )
    parser.add_argument("--events", type=int, default=DEFAULT_EVENT_COUNT)
    parser.add_argument(
        "--sessions-per-event", type=int, default=DEFAULT_SESSIONS_PER_EVENT
    )
    parser.add_argument(
        "--exhibitors-per-event", type=int, default=DEFAULT_EXHIBITORS_PER_EVENT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/synthetic_retrieval_benchmark.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    benchmark = generate_benchmark(
        event_count=args.events,
        sessions_per_event=args.sessions_per_event,
        exhibitors_per_event=args.exhibitors_per_event,
        seed=args.seed,
    )
    write_benchmark(benchmark, args.output)
    metadata = benchmark["metadata"]
    print(
        f"Wrote {metadata['record_count']} records and "
        f"{metadata['query_count']} queries to {args.output}"
    )


if __name__ == "__main__":
    main()
