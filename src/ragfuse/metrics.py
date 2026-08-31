"""Retrieval metrics, so fusion weights can be chosen from evidence.

Every function takes a ranked list of identifiers (best first) and the set of
identifiers that are actually relevant, and uses binary relevance -- a document
is either relevant or it is not. That matches how retrieval benchmarks are
usually labelled, and keeps the numbers comparable with published baselines.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log2

__all__ = [
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def _dedupe(retrieved: Sequence[str]) -> list[str]:
    """Drop repeats, keeping first occurrence -- a duplicate is not a new hit."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in retrieved:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    The denominator is ``min(k, len(retrieved))``, so a system that returns
    three results is not penalised for the seven it never claimed to return.
    Returns ``0.0`` when nothing was retrieved.
    """
    _check_k(k)
    relevant_set = set(relevant)
    top = _dedupe(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant_set) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant documents that appear in the top ``k``.

    Returns ``0.0`` when there are no relevant documents, since no system can
    be credited for finding nothing.
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _dedupe(retrieved)[:k]
    return sum(1 for item in top if item in relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """``1 / rank`` of the first relevant document, or ``0.0`` if none appear."""
    relevant_set = set(relevant)
    for position, item in enumerate(_dedupe(retrieved), start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(
    retrieved_per_query: Sequence[Sequence[str]],
    relevant_per_query: Sequence[Iterable[str]],
) -> float:
    """Mean of :func:`reciprocal_rank` over a set of queries.

    Raises:
        ValueError: If the two sequences are different lengths.
    """
    if len(retrieved_per_query) != len(relevant_per_query):
        raise ValueError(
            f"got {len(retrieved_per_query)} result lists for {len(relevant_per_query)} label sets"
        )
    if not retrieved_per_query:
        return 0.0
    return sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_per_query, relevant_per_query, strict=True)
    ) / len(retrieved_per_query)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k``, with binary relevance.

    Gains are discounted by ``1 / log2(rank + 1)`` and normalized against the
    best achievable ordering, so the result sits in ``0..1`` and stays
    comparable across queries with different numbers of relevant documents.
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    top = _dedupe(retrieved)[:k]
    dcg = sum(
        1.0 / log2(position + 1)
        for position, item in enumerate(top, start=1)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
