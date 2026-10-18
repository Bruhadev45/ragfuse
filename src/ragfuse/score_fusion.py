"""Score-based fusion: CombSUM and CombMNZ (Fox & Shaw, 1994).

Where RRF throws the retrievers' scores away and fuses on rank, these keep
them: each retriever's scores are normalised onto a common scale, weighted, and
summed::

    CombSUM(d) = sum over retrievers i of  w_i * norm_i(d)
    CombMNZ(d) = (number of retrievers that found d) * CombSUM(d)

Scores carry more information than ranks (a near-tie stays a near-tie), but
only if the normalisation makes them comparable, which is exactly the
assumption RRF avoids. ``normalize`` picks how:

- ``"minmax"`` (the default, after Lee 1997): ``(s - min) / (max - min)`` per
  list, so every retriever spans ``0..1``. A list whose scores are all equal,
  including a single hit, normalises to ``1.0``.
- ``"zscore"``: ``(s - mean) / std`` per list (population std). A constant list
  normalises to ``0.0``. Fused totals can be negative.
- ``None``: scores are used as given, for retrievers already on one scale.

Inputs are ranked lists of ``(document, score)`` pairs, best-first, in the same
shapes and with the same ``weights=``/``key=`` contract as
:func:`ragfuse.reciprocal_rank_fusion`, and the output is the same
:class:`ragfuse.FusedHit`, so the three fusions are drop-in replacements for
one another.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import fsum, isfinite, sqrt
from typing import Any, Literal, TypeVar

from ._types import Contribution, FusedHit
from .fusion import Weights, _as_named_rankings, _as_weight_map, _assemble_hits, _document_key

__all__ = ["NORMALIZATIONS", "Normalization", "combmnz", "combsum"]

T = TypeVar("T")

ScoredRankings = Mapping[str, Sequence[tuple[T, float]]] | Sequence[Sequence[tuple[T, float]]]
Normalization = Literal["minmax", "zscore"] | None

NORMALIZATIONS: tuple[Normalization, ...] = ("minmax", "zscore", None)
"""The values ``normalize=`` accepts."""


def _minmax(scores: Sequence[float]) -> list[float]:
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


def _zscore(scores: Sequence[float]) -> list[float]:
    mean = fsum(scores) / len(scores)
    std = sqrt(fsum((s - mean) ** 2 for s in scores) / len(scores))
    if std == 0:
        return [0.0] * len(scores)
    return [(s - mean) / std for s in scores]


def _normalise(scores: Sequence[float], normalize: Normalization) -> list[float]:
    if not scores or normalize is None:
        return list(scores)
    return _minmax(scores) if normalize == "minmax" else _zscore(scores)


def _unique_scored(
    ranking: Sequence[Any], source: str, key: Callable[[Any], str] | None
) -> list[tuple[str, Any, float]]:
    """Validate one scored list into ``(key, document, score)``, first occurrence kept."""
    seen: set[str] = set()
    unique: list[tuple[str, Any, float]] = []
    for entry in ranking:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise TypeError(
                f"ranking {source!r} must hold (document, score) pairs, got {entry!r}; "
                "use reciprocal_rank_fusion for rank-only lists"
            )
        document, raw = entry
        try:
            score = float(raw)
        except (TypeError, ValueError):
            raise TypeError(
                f"score for a document in ranking {source!r} is {raw!r}, not a number"
            ) from None
        if not isfinite(score):
            raise ValueError(
                f"score for a document in ranking {source!r} is {score}; scores must be finite"
            )
        doc_key = _document_key(document, source, key)
        # As in RRF, a repeat inside one list counts once, at its first position.
        if doc_key not in seen:
            seen.add(doc_key)
            unique.append((doc_key, document, score))
    return unique


def _score_fusion(
    rankings: ScoredRankings[T],
    *,
    normalize: Normalization,
    weights: Weights | None,
    key: Callable[[T], str] | None,
    top_k: int | None,
    multiply_by_hits: bool,
) -> list[FusedHit]:
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    named = _as_named_rankings(rankings)
    weight_map = _as_weight_map(weights, list(named))

    totals: dict[str, float] = {}
    hit_counts: dict[str, int] = {}
    contributions: dict[str, list[Contribution]] = {}
    best_item: dict[str, tuple[float, Any]] = {}

    for source, ranking in named.items():
        weight = weight_map[source]
        unique = _unique_scored(ranking, source, key)
        normalised = _normalise([score for _, _, score in unique], normalize)

        for rank, ((doc_key, document, _), value) in enumerate(
            zip(unique, normalised, strict=True)
        ):
            score = weight * value
            totals[doc_key] = totals.get(doc_key, 0.0) + score
            # A zero-weight retriever is silenced, so it must not count as a
            # vote in CombMNZ either.
            hit_counts[doc_key] = hit_counts.get(doc_key, 0) + (1 if weight > 0 else 0)
            contributions.setdefault(doc_key, []).append(
                Contribution(source=source, rank=rank, weight=weight, score=score)
            )
            if doc_key not in best_item or score > best_item[doc_key][0]:
                best_item[doc_key] = (score, document)

    if multiply_by_hits:
        totals = {doc_key: total * hit_counts[doc_key] for doc_key, total in totals.items()}

    return _assemble_hits(totals, contributions, best_item, top_k)


def combsum(
    rankings: ScoredRankings[T],
    *,
    normalize: Normalization = "minmax",
    weights: Weights | None = None,
    key: Callable[[T], str] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Fuse scored lists by summing each document's weighted, normalised scores.

    Args:
        rankings: ``{name: [(document, score), ...]}`` or a sequence of such
            lists (named ``"0"``, ``"1"``, ...), each ordered best-first.
        normalize: ``"minmax"``, ``"zscore"`` or ``None``; see the module docs.
            Applied to each list separately, before weighting.
        weights: Per-retriever weights, exactly as in RRF.
        key: Extracts a stable identifier from a document, exactly as in RRF.
        top_k: Truncate the fused ranking to this many hits.

    Returns:
        Fused hits by descending score, ties broken on ``key``. Each
        contribution's ``score`` is the weighted normalised score that
        retriever added, and ``rank`` its position in that retriever's list.

    Raises:
        ValueError: On an unknown ``normalize``, a negative ``top_k``, a NaN or
            infinite score, or the weight errors RRF raises.
        TypeError: If a list entry is not a ``(document, score)`` pair, a score
            is not a number, or documents are not strings and no ``key`` is
            given.

    Example:
        >>> rankings = {
        ...     "bm25": [("x", 12.0), ("y", 6.0), ("z", 0.0)],
        ...     "dense": [("w", 0.9), ("y", 0.3), ("v", -0.1)],
        ... }
        >>> [(hit.key, round(hit.score, 2)) for hit in combsum(rankings, top_k=3)]
        [('w', 1.0), ('x', 1.0), ('y', 0.9)]
    """
    return _score_fusion(
        rankings,
        normalize=normalize,
        weights=weights,
        key=key,
        top_k=top_k,
        multiply_by_hits=False,
    )


def combmnz(
    rankings: ScoredRankings[T],
    *,
    normalize: Normalization = "minmax",
    weights: Weights | None = None,
    key: Callable[[T], str] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    """CombSUM multiplied by the number of retrievers that found the document.

    The multiplier rewards agreement, much as RRF does implicitly: a document
    two retrievers both return outranks one a single retriever scored
    slightly higher. Retrievers with weight 0 do not count towards it.
    Arguments, return value and errors are those of :func:`combsum`; the
    hit's ``score`` is the multiplied total, so it no longer equals the sum of
    its contributions.

    With ``normalize="zscore"`` (or raw negative scores) a negative sum is
    multiplied too, pushing a below-average document that several retrievers
    returned further down; prefer ``"minmax"`` unless that is what you want.

    Example:
        >>> rankings = {
        ...     "bm25": [("x", 12.0), ("y", 6.0), ("z", 0.0)],
        ...     "dense": [("w", 0.9), ("y", 0.3), ("v", -0.1)],
        ... }
        >>> [(hit.key, round(hit.score, 2)) for hit in combmnz(rankings, top_k=3)]
        [('y', 1.8), ('w', 1.0), ('x', 1.0)]
    """
    return _score_fusion(
        rankings,
        normalize=normalize,
        weights=weights,
        key=key,
        top_k=top_k,
        multiply_by_hits=True,
    )
