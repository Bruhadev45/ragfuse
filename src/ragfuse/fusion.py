"""Weighted Reciprocal Rank Fusion.

RRF combines several ranked lists using only the *positions* documents hold in
them, never their raw scores. That is what makes it robust: a BM25 score and a
cosine similarity are not comparable, but a rank is a rank.

The classic formulation (Cormack et al., 2009) is::

    score(d) = sum over retrievers i of  1 / (k + rank_i(d))

This module adds a per-retriever weight::

    score(d) = sum over retrievers i of  w_i / (k + rank_i(d))

which is the piece most implementations leave out. The common workaround is to
pass the same list twice to count it double -- that only expresses integer
weights, and silently breaks as soon as you want 0.6/0.4.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isnan
from typing import Any, TypeVar

from ._types import Contribution, FusedHit

__all__ = ["reciprocal_rank_fusion", "rrf"]

T = TypeVar("T")

Rankings = Mapping[str, Sequence[T]] | Sequence[Sequence[T]]
Weights = Mapping[str, float] | Sequence[float]

DEFAULT_K = 60
"""Standard RRF smoothing constant. Larger values flatten the rank discount."""


def _as_named_rankings(rankings: Rankings[T]) -> dict[str, Sequence[T]]:
    """Normalize either input shape into ``{name: ranking}``."""
    if isinstance(rankings, Mapping):
        named = dict(rankings)
    else:
        if isinstance(rankings, (str, bytes)):
            raise TypeError("rankings must be a mapping or a sequence of ranked lists")
        named = {str(i): ranking for i, ranking in enumerate(rankings)}

    if not named:
        raise ValueError("rankings must contain at least one ranked list")

    for name, ranking in named.items():
        if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
            raise TypeError(f"ranking {name!r} must be a sequence of documents")

    return named


def _as_weight_map(
    weights: Weights | None,
    names: Sequence[str],
) -> dict[str, float]:
    """Normalize weights into ``{name: weight}``, defaulting to 1.0 each."""
    if weights is None:
        return dict.fromkeys(names, 1.0)

    if isinstance(weights, Mapping):
        unknown = set(weights) - set(names)
        if unknown:
            raise ValueError(
                f"weights reference unknown rankings: {sorted(unknown)}; "
                f"known rankings are {sorted(names)}"
            )
        resolved = {name: float(weights.get(name, 1.0)) for name in names}
    else:
        if len(weights) != len(names):
            raise ValueError(
                f"got {len(weights)} weights for {len(names)} rankings; "
                "pass a mapping if you only want to weight some of them"
            )
        resolved = {name: float(w) for name, w in zip(names, weights, strict=True)}

    for name, weight in resolved.items():
        # NaN slips past both comparisons below (NaN < 0 and NaN == 0 are
        # false), so reject it explicitly rather than corrupt every score.
        if isnan(weight):
            raise ValueError(f"weight for {name!r} is NaN; weights must be finite and >= 0")
        if weight < 0:
            raise ValueError(f"weight for {name!r} is negative ({weight}); weights must be >= 0")

    if all(weight == 0 for weight in resolved.values()):
        raise ValueError("at least one weight must be greater than zero")

    return resolved


def _document_key(document: Any, source: str, key: Callable[[Any], str] | None) -> str:
    """Identify ``document`` with ``key``, or as itself when it is a string."""
    if key is not None:
        return key(document)
    if isinstance(document, str):
        return document
    raise TypeError(
        f"document in ranking {source!r} is {type(document).__name__}, not str; "
        "pass key= to tell ragfuse how to identify documents"
    )


def reciprocal_rank_fusion(
    rankings: Rankings[T],
    *,
    k: int = DEFAULT_K,
    weights: Weights | None = None,
    key: Callable[[T], str] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked lists into one ranking with weighted RRF.

    Args:
        rankings: Either ``{name: ranked_list}`` or a sequence of ranked lists
            (which are then named ``"0"``, ``"1"``, ...). Each list must be
            ordered best-first.
        k: Smoothing constant. Must be positive. Larger values reduce the gap
            between adjacent ranks, making the fusion less sensitive to the
            exact ordering within each list.
        weights: Per-retriever weights, as a mapping or a sequence aligned with
            ``rankings``. Defaults to 1.0 for every retriever. Weights are used
            as given -- they need not sum to 1.
        key: Extracts a stable identifier from a document. Required unless the
            documents are already strings.
        top_k: Truncate the fused ranking to this many hits. ``None`` returns
            everything.

    Returns:
        Fused hits ordered by descending score. Ties break on ``key`` so the
        output is deterministic across runs.

    Raises:
        ValueError: If ``k`` is not positive, ``top_k`` is negative, weights are
            negative or all zero, or no rankings are supplied.
        TypeError: If documents are not strings and no ``key`` is given.

    Example:
        >>> bm25 = ["sec-149", "sec-135", "sec-188"]
        >>> dense = ["sec-135", "sec-149", "sec-203"]
        >>> hits = reciprocal_rank_fusion(
        ...     {"bm25": bm25, "dense": dense},
        ...     weights={"bm25": 0.6, "dense": 0.4},
        ...     top_k=2,
        ... )
        >>> [hit.key for hit in hits]
        ['sec-149', 'sec-135']
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    named = _as_named_rankings(rankings)
    weight_map = _as_weight_map(weights, list(named))

    totals: dict[str, float] = {}
    contributions: dict[str, list[Contribution]] = {}
    best_item: dict[str, tuple[float, Any]] = {}

    for source, ranking in named.items():
        weight = weight_map[source]
        seen_in_this_ranking: set[str] = set()

        for document in ranking:
            doc_key = _document_key(document, source, key)
            # A document repeated inside one list must not be counted twice;
            # its best (earliest) rank is the one that carries information.
            # Rank is the position among *unique* documents, so a skipped
            # duplicate does not push every later document down a slot --
            # that matters when key= collapses chunks to their parent doc.
            if doc_key in seen_in_this_ranking:
                continue
            rank = len(seen_in_this_ranking)
            seen_in_this_ranking.add(doc_key)

            score = weight / (k + rank + 1)
            totals[doc_key] = totals.get(doc_key, 0.0) + score
            contributions.setdefault(doc_key, []).append(
                Contribution(source=source, rank=rank, weight=weight, score=score)
            )
            if doc_key not in best_item or score > best_item[doc_key][0]:
                best_item[doc_key] = (score, document)

    return _assemble_hits(totals, contributions, best_item, top_k)


def _relative(score: float, top: float, lowest: float) -> float:
    """Place ``score`` in ``0..1`` relative to the top fused score.

    Non-negative scores (every RRF score) are divided by the top one. Score
    fusion can produce negative totals (z-scores, raw logits), where a ratio
    stops meaning anything, so those rankings are min-max scaled instead.
    """
    if lowest >= 0:
        return score / top if top > 0 else 0.0
    if top == lowest:
        return 1.0
    return (score - lowest) / (top - lowest)


def _assemble_hits(
    totals: Mapping[str, float],
    contributions: Mapping[str, Sequence[Contribution]],
    best_item: Mapping[str, tuple[float, Any]],
    top_k: int | None,
) -> list[FusedHit]:
    """Order fused totals into hits, ties broken on key, truncated to ``top_k``."""
    if not totals:
        return []

    ordered_keys = sorted(totals, key=lambda doc_key: (-totals[doc_key], doc_key))
    top_score = totals[ordered_keys[0]]
    lowest = totals[ordered_keys[-1]]

    hits = [
        FusedHit(
            key=doc_key,
            score=totals[doc_key],
            normalized_score=_relative(totals[doc_key], top_score, lowest),
            contributions=tuple(sorted(contributions[doc_key], key=lambda c: (-c.score, c.source))),
            item=best_item[doc_key][1],
        )
        for doc_key in ordered_keys
    ]

    return hits if top_k is None else hits[:top_k]


rrf = reciprocal_rank_fusion
"""Short alias for :func:`reciprocal_rank_fusion`."""
