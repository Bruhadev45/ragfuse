"""Weighted Reciprocal Rank Fusion for hybrid retrieval.

Fuse the ranked lists from several retrievers into one, with real per-retriever
weights, provenance on every hit, and no dependencies.

    >>> from ragfuse import rrf
    >>> hits = rrf(
    ...     {"bm25": ["sec-149", "sec-135"], "dense": ["sec-135", "sec-149"]},
    ...     weights={"bm25": 0.6, "dense": 0.4},
    ... )
    >>> hits[0].key
    'sec-149'
"""

from __future__ import annotations

from ._types import Contribution, FusedHit
from .fusion import DEFAULT_K, reciprocal_rank_fusion, rrf
from .metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .tuning import METRICS, WeightSearchResult, evaluate_weights, tune_weights

__version__ = "0.1.1"

__all__ = [
    "DEFAULT_K",
    "METRICS",
    "Contribution",
    "FusedHit",
    "WeightSearchResult",
    "__version__",
    "evaluate_weights",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "reciprocal_rank_fusion",
    "rrf",
    "tune_weights",
]
