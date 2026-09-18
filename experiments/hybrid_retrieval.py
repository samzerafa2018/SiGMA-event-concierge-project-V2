from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from experiments.benchmark_retrieval import LexicalIndex, SearchResult, _normalise


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    model_id: str

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Lazy adapter so normal application installs do not need ML dependencies."""

    def __init__(self, model_id: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Vector experiments require requirements-experiments.txt"
            ) from error

        self.model_id = model_id
        self._model = SentenceTransformer(model_id)

    def _encode(self, texts: Sequence[str], method_name: str) -> np.ndarray:
        method = getattr(self._model, method_name, self._model.encode)
        return np.asarray(
            method(
                list(texts),
                batch_size=64,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "encode_document")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "encode_query")


def record_text(record: dict[str, Any]) -> str:
    fields = [
        f"Type: {record['kind']}",
        f"Topic: {record['topic']}",
        f"Title: {record['title']}",
        f"Description: {record['description']}",
    ]
    if record.get("speakers"):
        fields.append(f"Speakers: {', '.join(record['speakers'])}")
    return "\n".join(fields)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embeddings must not contain zero-length vectors")
    return values / norms


def _embedding_fingerprint(records: list[dict[str, Any]], model_id: str) -> str:
    digest = hashlib.sha256(model_id.encode("utf-8"))
    for record in records:
        digest.update(record["record_id"].encode("utf-8"))
        digest.update(record_text(record).encode("utf-8"))
    return digest.hexdigest()


def load_or_create_document_embeddings(
    records: list[dict[str, Any]],
    encoder: TextEncoder,
    cache_path: Path | None = None,
    *,
    rebuild: bool = False,
) -> tuple[np.ndarray, bool]:
    """Return normalised document vectors and whether a valid cache was used."""
    fingerprint = _embedding_fingerprint(records, encoder.model_id)
    if cache_path is not None and cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_fingerprint = str(cached["fingerprint"].item())
            embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        if cached_fingerprint == fingerprint and len(embeddings) == len(records):
            return _normalise_rows(embeddings), True

    embeddings = np.asarray(
        encoder.encode_documents([record_text(record) for record in records]),
        dtype=np.float32,
    )
    if len(embeddings) != len(records):
        raise ValueError("Encoder returned a different number of document embeddings")
    embeddings = _normalise_rows(embeddings)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            fingerprint=np.asarray(fingerprint),
            embeddings=embeddings,
        )
    return embeddings, False


class VectorIndex:
    def __init__(
        self,
        records: list[dict[str, Any]],
        embeddings: np.ndarray,
        encoder: TextEncoder,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        if embedding_array.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")
        if len(records) != len(embedding_array):
            raise ValueError("Each record must have one embedding")
        self.records = records
        self.embeddings = _normalise_rows(embedding_array)
        self.encoder = encoder
        self.indices_by_event: dict[str, list[int]] = defaultdict(list)
        self.indices_by_kind: dict[str, list[int]] = defaultdict(list)
        self.indices_by_topic: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.indices_by_event[record["event_id"]].append(index)
            self.indices_by_kind[record["kind"]].append(index)
            self.indices_by_topic[record["topic"]].append(index)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        event_id: str | None = None,
        kind: str | None = None,
        topic: str | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidate_indices = list(range(len(self.records)))
        if event_id is not None:
            candidate_indices = list(self.indices_by_event.get(event_id, []))
        if kind is not None:
            allowed = set(self.indices_by_kind.get(kind, []))
            candidate_indices = [index for index in candidate_indices if index in allowed]
        if topic is not None:
            allowed = set(self.indices_by_topic.get(topic, []))
            candidate_indices = [index for index in candidate_indices if index in allowed]
        if not candidate_indices:
            return []

        query_embedding = np.asarray(
            self.encoder.encode_queries([query]), dtype=np.float32
        )
        query_embedding = _normalise_rows(query_embedding)[0]
        similarities = self.embeddings[candidate_indices] @ query_embedding
        ranked = sorted(
            zip(similarities.tolist(), candidate_indices),
            key=lambda item: (-item[0], self.records[item[1]]["record_id"]),
        )
        return [
            SearchResult(
                record_id=self.records[index]["record_id"],
                event_id=self.records[index]["event_id"],
                score=score,
            )
            for score, index in ranked[:top_k]
        ]


def reciprocal_rank_fusion(
    lexical_results: Sequence[SearchResult],
    vector_results: Sequence[SearchResult],
    *,
    top_k: int = 5,
    rank_constant: int = 60,
    lexical_weight: float = 1.15,
    vector_weight: float = 1.0,
) -> list[SearchResult]:
    """Fuse differently scaled result lists without tuning raw-score ranges."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")
    if lexical_weight <= 0 or vector_weight <= 0:
        raise ValueError("fusion weights must be positive")

    scores: defaultdict[str, float] = defaultdict(float)
    results_by_id: dict[str, SearchResult] = {}
    for weight, results in (
        (lexical_weight, lexical_results),
        (vector_weight, vector_results),
    ):
        for rank, result in enumerate(results, start=1):
            scores[result.record_id] += weight / (rank_constant + rank)
            results_by_id[result.record_id] = result

    ranked_ids = sorted(scores, key=lambda record_id: (-scores[record_id], record_id))
    return [
        SearchResult(
            record_id=record_id,
            event_id=results_by_id[record_id].event_id,
            score=scores[record_id],
        )
        for record_id in ranked_ids[:top_k]
    ]


class HybridIndex:
    def __init__(self, records: list[dict[str, Any]], vector_index: VectorIndex) -> None:
        self.lexical_index = LexicalIndex(records)
        self.vector_index = vector_index

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        event_id: str | None = None,
        kind: str | None = None,
        topic: str | None = None,
        candidate_k: int = 30,
    ) -> list[SearchResult]:
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")
        lexical_results = self.lexical_index.search(
            query,
            top_k=candidate_k,
            event_id=event_id,
            kind=kind,
            topic=topic,
        )
        vector_results = self.vector_index.search(
            query,
            top_k=candidate_k,
            event_id=event_id,
            kind=kind,
            topic=topic,
        )
        return reciprocal_rank_fusion(
            lexical_results,
            vector_results,
            top_k=top_k,
        )


class AdaptiveHybridIndex:
    """Choose lexical retrieval for exact signals and vectors otherwise.

    This keeps identifier and exact-title lookups fast and deterministic while
    reserving the embedding model for queries that need semantic matching.
    """

    def __init__(self, records: list[dict[str, Any]], vector_index: VectorIndex) -> None:
        self.records = records
        self.lexical_index = LexicalIndex(records)
        self.vector_index = vector_index

    def _has_exact_signal(
        self,
        query: str,
        *,
        event_id: str | None,
        kind: str | None,
        topic: str | None,
    ) -> bool:
        normalised_query = _normalise(query)
        query_lower = query.lower()
        for record in self.records:
            if event_id is not None and record["event_id"] != event_id:
                continue
            if kind is not None and record["kind"] != kind:
                continue
            if topic is not None and record["topic"] != topic:
                continue
            if record["record_id"].lower() in query_lower:
                return True
            normalised_title = _normalise(record["title"])
            if normalised_title and normalised_title in normalised_query:
                return True
        return False

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        event_id: str | None = None,
        kind: str | None = None,
        topic: str | None = None,
    ) -> list[SearchResult]:
        index = (
            self.lexical_index
            if self._has_exact_signal(
                query,
                event_id=event_id,
                kind=kind,
                topic=topic,
            )
            else self.vector_index
        )
        return index.search(
            query,
            top_k=top_k,
            event_id=event_id,
            kind=kind,
            topic=topic,
        )
