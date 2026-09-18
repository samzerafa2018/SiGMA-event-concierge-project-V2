from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.benchmark_retrieval import _normalise
from experiments.generate_synthetic_corpus import TOPICS
from experiments.hybrid_retrieval import TextEncoder, _normalise_rows


@dataclass(frozen=True)
class RouteDecision:
    event_id: str | None
    topic: str
    event_source: str
    topic_source: str
    topic_score: float
    topic_margin: float


def _topic_document(topic: str) -> str:
    details = TOPICS[topic]
    return "\n".join(
        (
            f"Topic: {topic}",
            f"Areas: {', '.join(details['focuses'])}",
            f"Examples: {', '.join(details['paraphrases'])}",
        )
    )


class QueryRouter:
    """Route a question by event metadata and topic meaning."""

    def __init__(self, events: list[dict[str, Any]], encoder: TextEncoder) -> None:
        if not events:
            raise ValueError("events must not be empty")
        self.events = events
        self.encoder = encoder
        self.events_by_id = {event["event_id"]: event for event in events}
        if len(self.events_by_id) != len(events):
            raise ValueError("event IDs must be unique")
        self.topics = list(TOPICS)
        topic_embeddings = np.asarray(
            encoder.encode_documents(
                [_topic_document(topic) for topic in self.topics]
            ),
            dtype=np.float32,
        )
        if len(topic_embeddings) != len(self.topics):
            raise ValueError("Encoder returned an invalid number of topic embeddings")
        self.topic_embeddings = _normalise_rows(topic_embeddings)

    def _route_event(
        self, query: str, selected_event_id: str | None
    ) -> tuple[str | None, str]:
        normalised_query = _normalise(query)
        for event in self.events:
            if _normalise(event["event_id"]) in normalised_query:
                return event["event_id"], "explicit_id"
            if _normalise(event["event_name"]) in normalised_query:
                return event["event_id"], "explicit_name"

        if selected_event_id is not None:
            if selected_event_id not in self.events_by_id:
                raise ValueError(f"Unknown selected event: {selected_event_id}")
            return selected_event_id, "conversation_context"
        return None, "unresolved"

    def _route_topic(self, query: str) -> tuple[str, str, float, float]:
        normalised_query = _normalise(query)
        for topic in self.topics:
            if _normalise(topic) in normalised_query:
                return topic, "explicit_topic", 1.0, 1.0
            for focus in TOPICS[topic]["focuses"]:
                if _normalise(focus) in normalised_query:
                    return topic, "exact_focus", 1.0, 1.0

        query_embedding = np.asarray(
            self.encoder.encode_queries([query]), dtype=np.float32
        )
        query_vector = _normalise_rows(query_embedding)[0]
        similarities = self.topic_embeddings @ query_vector
        best_index = int(np.argmax(similarities))
        ordered = np.sort(similarities)
        margin = float(ordered[-1] - ordered[-2])
        return (
            self.topics[best_index],
            "semantic",
            float(similarities[best_index]),
            margin,
        )

    def route(
        self, query: str, *, selected_event_id: str | None = None
    ) -> RouteDecision:
        event_id, event_source = self._route_event(query, selected_event_id)
        topic, topic_source, topic_score, topic_margin = self._route_topic(query)
        return RouteDecision(
            event_id=event_id,
            topic=topic,
            event_source=event_source,
            topic_source=topic_source,
            topic_score=topic_score,
            topic_margin=topic_margin,
        )


def should_partition_topic(
    decision: RouteDecision, *, min_semantic_margin: float
) -> bool:
    if min_semantic_margin < 0:
        raise ValueError("min_semantic_margin must not be negative")
    if decision.topic_source in {"explicit_topic", "exact_focus"}:
        return True
    return decision.topic_margin >= min_semantic_margin
