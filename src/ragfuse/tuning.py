"""Pick fusion weights from labelled data instead of by feel.

Weights are the one knob RRF gives you, and guessing at them is how a hybrid
retriever ends up worse than either of its halves. This module sweeps the
weight simplex and reports what each combination actually scores.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import comb

from .fusion import DEFAULT_K, reciprocal_rank_fusion
from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

__all__ = ["METRICS", "WeightSearchResult", "evaluate_weights", "tune_weights"]

Metric = Callable[[Sequence[str], Iterable[str], int], float]

METRICS: dict[str, Metric] = {
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": lambda retrieved, relevant, k: reciprocal_rank(retrieved[:k], relevant),
}
"""Metrics addressable by name. All are ``(retrieved, relevant, k) -> float``."""


@dataclass(frozen=True, slots=True)
class WeightSearchResult:
    """Outcome of a weight sweep.

    Attributes:
        weights: The best weight assignment found.
        score: Mean metric value at ``weights``.
        metric: Name of the metric that was optimised.
        top_k: Cutoff the metric was computed at.
        table: Every combination tried, best first, as ``(weights, score)``.
        baseline_score: Score of the exact equal-weight assignment, evaluated
            directly on the data (not looked up in the grid, which may not
            contain it at every resolution/source-count combination).
    """

    weights: dict[str, float]
    score: float
    metric: str
    top_k: int
    table: tuple[tuple[dict[str, float], float], ...]
    baseline_score: float

    def improvement(self) -> float:
        """How much tuning gained over equal weights, in metric points."""
        return self.score - self.baseline_score


def _resolve_metric(metric: str | Metric) -> tuple[str, Metric]:
    if callable(metric):
        return getattr(metric, "__name__", "custom"), metric
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    return metric, METRICS[metric]


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield every way to write ``total`` as an ordered sum of ``parts`` >= 0."""
    if parts == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for tail in _compositions(total - head, parts - 1):
            yield (head, *tail)


def _simplex_grid(n_sources: int, resolution: int) -> list[tuple[float, ...]]:
    """All weight vectors on a lattice over the simplex, summing to 1.

    Generated as integer compositions, so the cost is the number of valid
    points -- C(resolution + n - 1, n - 1) -- rather than the full
    ``(resolution + 1) ** n`` Cartesian product, which becomes millions of
    discarded tuples by six retrievers.
    """
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}")
    n_points = comb(resolution + n_sources - 1, n_sources - 1)
    if n_points > 200_000:
        raise ValueError(
            f"resolution={resolution} over {n_sources} rankings means "
            f"{n_points:,} weight combinations; lower the resolution or "
            "tune fewer retrievers at once"
        )
    return [
        tuple(c / resolution for c in counts) for counts in _compositions(resolution, n_sources)
    ]


def evaluate_weights(
    rankings_per_query: Sequence[Mapping[str, Sequence[str]]],
    relevant_per_query: Sequence[Iterable[str]],
    weights: Mapping[str, float],
    *,
    k: int = DEFAULT_K,
    top_k: int = 5,
    metric: str | Metric = "ndcg",
) -> float:
    """Mean metric value when fusing every query with ``weights``."""
    if len(rankings_per_query) != len(relevant_per_query):
        raise ValueError(
            f"got {len(rankings_per_query)} query rankings for {len(relevant_per_query)} label sets"
        )
    if not rankings_per_query:
        return 0.0

    _, metric_fn = _resolve_metric(metric)
    total = 0.0
    for rankings, relevant in zip(rankings_per_query, relevant_per_query, strict=True):
        fused = reciprocal_rank_fusion(rankings, k=k, weights=weights, top_k=top_k)
        total += metric_fn([hit.key for hit in fused], relevant, top_k)
    return total / len(rankings_per_query)


def tune_weights(
    rankings_per_query: Sequence[Mapping[str, Sequence[str]]],
    relevant_per_query: Sequence[Iterable[str]],
    *,
    k: int = DEFAULT_K,
    top_k: int = 5,
    metric: str | Metric = "ndcg",
    resolution: int = 10,
) -> WeightSearchResult:
    """Sweep the weight simplex and return the best assignment.

    Args:
        rankings_per_query: One ``{retriever: ranked_ids}`` mapping per query.
            Every query must name the same retrievers.
        relevant_per_query: The relevant ids for each query, in the same order.
        k: RRF smoothing constant, held fixed across the sweep.
        top_k: Cutoff at which the metric is computed.
        metric: A name from :data:`METRICS`, or a callable
            ``(retrieved, relevant, k) -> float``.
        resolution: Lattice steps per axis. ``10`` gives weights in units of
            0.1; raise it for a finer sweep at quadratic-or-worse cost.

    Returns:
        The best weights, their score, and the full table of what was tried.

    Raises:
        ValueError: If the inputs are empty, mismatched in length, disagree on
            which retrievers exist, or the metric name is unknown.

    Example:
        >>> rankings = [{"bm25": ["a", "b"], "dense": ["b", "a"]}]
        >>> result = tune_weights(rankings, [{"b"}], top_k=2, resolution=2)
        >>> round(result.score, 3) >= 0.0
        True
    """
    if not rankings_per_query:
        raise ValueError("rankings_per_query is empty; nothing to tune on")
    if len(rankings_per_query) != len(relevant_per_query):
        raise ValueError(
            f"got {len(rankings_per_query)} query rankings for {len(relevant_per_query)} label sets"
        )

    sources = tuple(rankings_per_query[0])
    if not sources:
        raise ValueError("the first query has no retrievers to fuse")
    for index, rankings in enumerate(rankings_per_query):
        if tuple(rankings) != sources:
            raise ValueError(
                f"query {index} names retrievers {tuple(rankings)}, "
                f"but query 0 names {sources}; they must match"
            )

    metric_name, _ = _resolve_metric(metric)

    scored: list[tuple[dict[str, float], float]] = []
    # Every vector on the simplex sums to 1, so no all-zero assignment can
    # occur here and reciprocal_rank_fusion's "some weight must be > 0" rule
    # is satisfied by construction.
    for vector in _simplex_grid(len(sources), resolution):
        weights = dict(zip(sources, vector, strict=True))
        score = evaluate_weights(
            rankings_per_query,
            relevant_per_query,
            weights,
            k=k,
            top_k=top_k,
            metric=metric,
        )
        scored.append((weights, score))

    # Ties break toward the more balanced assignment, which generalises better
    # than a knife-edge weighting that happens to suit this particular sample.
    scored.sort(key=lambda entry: (-entry[1], max(entry[0].values())))
    best_weights, best_score = scored[0]

    equal = dict.fromkeys(sources, 1.0 / len(sources))
    baseline = evaluate_weights(
        rankings_per_query, relevant_per_query, equal, k=k, top_k=top_k, metric=metric
    )

    return WeightSearchResult(
        weights=best_weights,
        score=best_score,
        metric=metric_name,
        top_k=top_k,
        table=tuple(scored),
        baseline_score=baseline,
    )
