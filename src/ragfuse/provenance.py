"""Fusion and reranking with a checked, complete audit trail.

:func:`fuse` runs one of the existing fusions (:func:`~ragfuse.rrf`,
:func:`~ragfuse.combsum` or :func:`~ragfuse.combmnz`) and returns a
:class:`FusionTrace`. For every fused candidate it records each retriever's
rank, raw score, the value fusion used, the weight and what it added,
including the retrievers that did not return it. :func:`rerank` reorders those
candidates by scores you supply, and only those candidates.

- **Stable ids.** Document ids and retriever names must be exact ``str``
  values (not subclasses), non-empty, without surrounding whitespace or control
  characters. ``key=`` is called once per entry. Retrievers are fused in name
  order and weights must be a mapping, so the mapping's order cannot change
  the trace.
- **Checked scores.** Every raw score must be a finite real number, not a
  ``bool`` or a numeric string; it is recorded as a float. Each list must be
  best-first: no score may exceed the one before it, so negate distances.
- **Closed candidate set.** With ``eligible=``, a retriever that returns any
  other id fails the whole fusion with :class:`IneligibleCandidateError`, even
  if that entry is malformed in other ways too. :func:`rerank` must score
  exactly the trace's candidates; a score for anything else raises the same
  error, so a reranker can reorder or cut the shortlist but not add to it.
  ``FusionTrace`` is a plain public dataclass that anyone can build, so pass
  ``rerank`` the same ``eligible=`` to re-check the trace itself;
  ``eligible_only`` is a record, not proof. The error is a ``ValueError``:
  catch it before any broad ``except ValueError``. These messages give counts
  and positions, never the ids.

Inputs are read in full: cap list lengths, retriever counts and score maps
in the service that calls these. Ids are checked; documents are not. A
candidate's ``item`` is whatever the retriever returned for that id, so fetch
the text you show or quote from your own store by id. Rerank scores order one
shortlist and nothing more; each candidate keeps its fusion scores beside them.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import inf, isfinite
from typing import Any, Literal, TypeVar

from ._types import FusedHit
from .fusion import DEFAULT_K, Weights, _as_float, _as_weight_map, reciprocal_rank_fusion
from .score_fusion import Normalization, _normalise, combmnz, combsum

__all__ = ["Candidate", "FusionTrace", "IneligibleCandidateError", "SourceTrace", "fuse", "rerank"]

T = TypeVar("T")
Method = Literal["rrf", "combsum", "combmnz"]
_METHODS: tuple[Method, ...] = ("rrf", "combsum", "combmnz")
_Entry = tuple[str, Any, float]  # (id, document, raw score)
_MAX_K = 2**53  # beyond this, k + rank + 1 is no longer exact as a float


class IneligibleCandidateError(ValueError):
    """A retriever or a reranker named a document outside the allowed candidates."""


@dataclass(frozen=True, slots=True)
class SourceTrace:
    """What one retriever did for one candidate.

    Attributes:
        source: The retriever's name.
        rank: Zero-indexed position among that list's unique documents, or
            ``None`` if the retriever did not return the candidate.
        raw_score: The retriever's score as a float, or ``None`` if absent.
        value: What fusion used before weighting: ``1 / (k + rank + 1)`` for
            RRF, the normalised score for CombSUM and CombMNZ. ``None`` if absent.
        weight: The retriever's resolved weight.
        contribution: What it added to the fused total, as the fusion computed
            it: ``weight / (k + rank + 1)`` for RRF, which can differ from
            ``weight * value`` in the last bit, and ``weight * value`` for
            score fusion. ``0.0`` if absent. CombMNZ multiplies the sum of
            contributions by the number of retrievers with a weight above zero
            that returned the candidate.
    """

    source: str
    rank: int | None
    raw_score: float | None
    value: float | None
    weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class Candidate:
    """One fused candidate with its provenance.

    Attributes:
        key: The document id.
        rank: Zero-indexed position in this trace, after reranking if any.
        fused_rank: Zero-indexed position the fusion gave it.
        fused_score: The fused score, as :attr:`FusedHit.score`.
        normalized_score: :attr:`FusedHit.normalized_score`, from the fusion.
        rerank_score: The score :func:`rerank` ordered it by, else ``None``.
        retrievers: One :class:`SourceTrace` per retriever, in name order,
            absent retrievers included.
        item: The retriever's document for this id, as :attr:`FusedHit.item`.
            It is not validated, and it is left out of ``repr``, equality and
            the hash.
    """

    key: str
    rank: int
    fused_rank: int
    fused_score: float
    normalized_score: float
    rerank_score: float | None
    retrievers: tuple[SourceTrace, ...]
    item: Any = field(default=None, repr=False, hash=False, compare=False)


@dataclass(frozen=True, slots=True)
class FusionTrace:
    """A fused (and possibly reranked) candidate list and how it was made.

    Attributes:
        method: ``"rrf"``, ``"combsum"`` or ``"combmnz"``.
        k: RRF's smoothing constant, or ``None`` for score fusion.
        normalize: Score fusion's normalisation, or ``None`` for RRF.
        weights: Every retriever's resolved weight, in name order.
        eligible_only: Whether ``fuse`` enforced an ``eligible=`` set. A
            record, not proof: a trace can be built by hand.
        reranked: Whether :func:`rerank` produced this trace.
        candidates: The candidates, best first.
    """

    method: Method
    k: int | None
    normalize: Normalization
    weights: tuple[tuple[str, float], ...]
    eligible_only: bool
    reranked: bool
    candidates: tuple[Candidate, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """The candidates' ids, best first."""
        return tuple(candidate.key for candidate in self.candidates)


def _check_type(value: object, what: str, hint: str = "") -> str:
    if type(value) is not str:
        raise TypeError(f"{what} must be an exact str, got {type(value).__name__!r}{hint}")
    return value


def _check_form(value: str, what: str) -> str:
    if not value or value != value.strip() or not value.isprintable():
        raise ValueError(f"{what} must be non-empty, without surrounding whitespace or controls")
    return value


def _checked_name(name: object) -> str:
    return _check_form(_check_type(name, "retriever name"), "retriever name")


def _check_top_k(top_k: object) -> None:
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError(f"top_k must be a non-negative int or None, got {top_k!r}")


def _check_score(raw: object, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
        raise TypeError(f"{where}: score must be a real number, got {type(raw).__name__!r}")
    score = _as_float(raw)
    if not isfinite(score):
        raise ValueError(f"{where}: score must be finite, got {score}")
    return score


def _allowed(eligible: Collection[str] | None) -> frozenset[str] | None:
    if eligible is None:
        return None
    if isinstance(eligible, (str, bytes)) or not isinstance(eligible, Collection):
        raise TypeError("eligible must be a collection of document ids, not one string")
    ids = list(eligible)
    if any(type(doc_id) is not str for doc_id in ids):
        raise TypeError("eligible document ids must be str")
    return frozenset(ids)


def _checked_list(
    source: str,
    ranking: object,
    key: Callable[[Any], str] | None,
    allowed: frozenset[str] | None,
) -> list[_Entry]:
    """Validate one retriever's list; keep each id's first occurrence."""
    if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
        raise TypeError(f"ranking {source!r} must be a sequence of (document, score) pairs")
    hint = "; pass key= to tell ragfuse how to identify documents" if key is None else ""
    unique: list[_Entry] = []
    seen: set[str] = set()
    previous = inf
    for position, entry in enumerate(ranking):
        where = f"ranking {source!r} position {position}"
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise TypeError(f"{where}: expected a (document, score) pair")
        document, raw = entry
        doc_id = _check_type(document if key is None else key(document), f"{where}: id", hint)
        # Eligibility first, so a leaked id is reported as one even when its
        # score or format is broken too.
        if allowed is not None and doc_id not in allowed:
            raise IneligibleCandidateError(f"{where} is not an eligible document")
        _check_form(doc_id, f"{where}: id")
        score = _check_score(raw, where)
        if score > previous:
            raise ValueError(f"{where} scores above the one before it; order lists best-first")
        previous = score
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append((doc_id, document, score))
    return unique


def _values(
    unique: Sequence[_Entry], method: Method, k: int, normalize: Normalization
) -> dict[str, tuple[float, float]]:
    """``{id: (raw score, value fusion used)}`` for one retriever's unique list."""
    raw = [score for _, _, score in unique]
    if method == "rrf":
        used = [1.0 / (k + rank + 1) for rank in range(len(unique))]
    else:
        used = _normalise(raw, normalize)
    return {doc_id: (score, value) for (doc_id, _, score), value in zip(unique, used, strict=True)}


def _candidate(
    rank: int,
    hit: FusedHit,
    values: Mapping[str, Mapping[str, tuple[float, float]]],
    weights: Mapping[str, float],
) -> Candidate:
    added = {c.source: c for c in hit.contributions}
    retrievers: list[SourceTrace] = []
    for source, weight in weights.items():
        if source in added:
            raw, value = values[source][hit.key]
            contribution = added[source]
            retrievers.append(
                SourceTrace(source, contribution.rank, raw, value, weight, contribution.score)
            )
        else:
            retrievers.append(SourceTrace(source, None, None, None, weight, 0.0))
    return Candidate(
        key=hit.key,
        rank=rank,
        fused_rank=rank,
        fused_score=hit.score,
        normalized_score=hit.normalized_score,
        rerank_score=None,
        retrievers=tuple(retrievers),
        item=hit.item[1],
    )


def _first(pair: tuple[str, Any]) -> str:
    return pair[0]


def fuse(
    rankings: Mapping[str, Sequence[tuple[T, float]]],
    *,
    method: Method = "rrf",
    weights: Weights | None = None,
    k: int = DEFAULT_K,
    normalize: Normalization = "minmax",
    key: Callable[[T], str] | None = None,
    eligible: Collection[str] | None = None,
    top_k: int | None = None,
) -> FusionTrace:
    """Fuse scored lists with an existing fusion and trace every contribution.

    Args:
        rankings: ``{retriever name: [(document, score), ...]}``, each list
            best-first. RRF records the scores but fuses on rank.
        method: ``"rrf"`` (default), ``"combsum"`` or ``"combmnz"``.
        weights: ``{retriever name: weight}``, as in :func:`~ragfuse.rrf`; a
            name left out gets 1.0. A sequence is refused, since it would pair
            weights with the mapping's order.
        k: RRF's smoothing constant, a positive int up to 2**53; unused, and
            recorded as ``None``, for score fusion.
        normalize: Score fusion's per-list normalisation; unused, and recorded
            as ``None``, for RRF.
        key: Extracts a document's id; required unless documents are ids.
        eligible: If given, the only ids a retriever may return.
        top_k: Keep this many fused candidates, e.g. a reranking shortlist.

    Raises:
        IneligibleCandidateError: If a retriever returns an id outside ``eligible``.
        ValueError: On a malformed id or name, a non-finite score, a list
            that is not best-first, an invalid ``k`` or ``top_k``, an overflow,
            or any error the underlying fusion raises for ``normalize`` or
            ``weights``.
        TypeError: On a non-mapping ``rankings`` or ``weights``, malformed
            entries, non-numeric scores or ids that are not exact strings.

    Example:
        >>> trace = fuse(
        ...     {"bm25": [("s-1", 7.2), ("s-2", 3.1)], "dense": [("s-2", 0.8)]},
        ...     eligible={"s-1", "s-2", "s-3"},
        ... )
        >>> trace.keys
        ('s-2', 's-1')
        >>> [(t.source, t.rank, t.raw_score) for t in trace.candidates[1].retrievers]
        [('bm25', 0, 7.2), ('dense', None, None)]
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not isinstance(rankings, Mapping):
        raise TypeError("rankings must map each retriever's name to its (document, score) list")
    if not rankings:
        raise ValueError("rankings must contain at least one ranked list")
    if weights is not None and not isinstance(weights, Mapping):
        raise TypeError("weights must map retriever names to weights, not follow their order")
    if method == "rrf" and (type(k) is not int or not 0 < k <= _MAX_K):
        raise ValueError("k must be a positive int no larger than 2**53")
    _check_top_k(top_k)
    names = sorted(_checked_name(name) for name in rankings)
    weight_map = _as_weight_map(weights, names)
    allowed = _allowed(eligible)
    lists = {name: _checked_list(name, rankings[name], key, allowed) for name in names}

    # Fuse on (id, document) pairs so key= is never called twice.
    pairs = {name: [((i, d), s) for i, d, s in lists[name]] for name in names}
    if method == "rrf":
        documents = {name: [pair for pair, _ in pairs[name]] for name in names}
        hits = reciprocal_rank_fusion(documents, k=k, weights=weight_map, key=_first, top_k=top_k)
    else:
        fusion = combsum if method == "combsum" else combmnz
        try:
            hits = fusion(pairs, normalize=normalize, weights=weight_map, key=_first, top_k=top_k)
        except OverflowError as exc:  # z-scoring scores near the float limit
            raise ValueError("the scores overflowed during normalisation") from exc

    values = {name: _values(lists[name], method, k, normalize) for name in names}
    return FusionTrace(
        method=method,
        k=k if method == "rrf" else None,
        normalize=None if method == "rrf" else normalize,
        weights=tuple(weight_map.items()),
        eligible_only=allowed is not None,
        reranked=False,
        candidates=tuple(_candidate(i, hit, values, weight_map) for i, hit in enumerate(hits)),
    )


def rerank(
    trace: FusionTrace,
    scores: Mapping[str, float],
    *,
    eligible: Collection[str] | None = None,
    top_k: int | None = None,
) -> FusionTrace:
    """Reorder a trace's candidates by ``scores``; never add a candidate.

    Ties keep the fused order. Each candidate keeps ``fused_rank``,
    ``fused_score`` and ``normalized_score`` and gains ``rerank_score``.

    Args:
        trace: What :func:`fuse` returned, not yet reranked.
        scores: ``{candidate id: score}``, higher is better, covering every
            candidate and nothing else. It is read once.
        eligible: If given, every candidate of ``trace`` must be in it, which
            re-checks a trace that did not come from ``fuse(eligible=...)``.
        top_k: Keep this many after reordering.

    Raises:
        IneligibleCandidateError: If ``scores`` names anything that is not a
            candidate of ``trace``, or ``trace`` holds a candidate outside
            ``eligible``.
        ValueError: If a candidate is unscored, a score is not finite, the
            trace is already reranked or repeats a candidate, or ``top_k`` is
            not a non-negative int.
        TypeError: On a wrong argument type or a non-numeric score.

    Example:
        >>> trace = fuse({"bm25": [("s-1", 7.2), ("s-2", 3.1)]})
        >>> rerank(trace, {"s-1": -1.5, "s-2": 4.0}).keys
        ('s-2', 's-1')
    """
    if type(trace) is not FusionTrace:
        raise TypeError("rerank takes the FusionTrace that fuse returned")
    if trace.reranked:
        raise ValueError("this trace is already reranked; rerank the one fuse returned")
    if not isinstance(scores, Mapping):
        raise TypeError("scores must map candidate ids to scores")
    _check_top_k(top_k)
    allowed = _allowed(eligible)
    if allowed is not None:
        stray = sum(1 for c in trace.candidates if type(c.key) is not str or c.key not in allowed)
        if stray:
            raise IneligibleCandidateError(
                f"the trace holds {stray} candidate(s) outside eligible; build it with fuse"
            )
    try:  # one read, so the checks below all see the same scores
        snapshot = dict(scores)
    except Exception:  # its own error could name an id; do not pass it on
        raise TypeError("scores could not be read as a mapping") from None
    pool = {candidate.key for candidate in trace.candidates}
    if len(pool) != len(trace.candidates):
        raise ValueError("the trace holds the same candidate twice; build it with fuse")
    outside = sum(1 for doc_id in snapshot if type(doc_id) is not str or doc_id not in pool)
    if outside:
        raise IneligibleCandidateError(
            f"the reranker scored {outside} document(s) that are not candidates of this "
            "fusion; reranking can reorder or cut the candidates, never add to them"
        )
    missing = sum(1 for c in trace.candidates if c.key not in snapshot)
    if missing:
        raise ValueError(f"the reranker left {missing} candidate(s) unscored")
    checked = {
        c.key: _check_score(snapshot[c.key], f"rerank score at fused rank {c.fused_rank}")
        for c in trace.candidates
    }
    order = sorted(trace.candidates, key=lambda c: (-checked[c.key], c.fused_rank))
    kept = order if top_k is None else order[:top_k]
    return replace(
        trace,
        reranked=True,
        candidates=tuple(
            replace(c, rank=i, rerank_score=checked[c.key]) for i, c in enumerate(kept)
        ),
    )
