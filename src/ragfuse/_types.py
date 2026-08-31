"""Immutable result types returned by the fusion functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Contribution", "FusedHit"]


@dataclass(frozen=True, slots=True)
class Contribution:
    """What a single retriever contributed to one fused hit.

    Attributes:
        source: Name of the retriever this contribution came from.
        rank: Zero-indexed position the document held in that retriever's list.
        weight: Weight applied to that retriever during fusion.
        score: The weighted RRF term, ``weight / (k + rank + 1)``.
    """

    source: str
    rank: int
    weight: float
    score: float


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One document in a fused ranking, with its provenance intact.

    Attributes:
        key: Stable identifier the document was fused on.
        score: Sum of every weighted RRF term for this document.
        normalized_score: ``score`` divided by the top hit's score, in ``0..1``.
        contributions: Per-retriever detail, ordered by descending contribution.
        item: The original object, taken from the retriever that contributed most.
    """

    key: str
    score: float
    normalized_score: float
    contributions: tuple[Contribution, ...]
    item: Any = None

    @property
    def sources(self) -> tuple[str, ...]:
        """Names of the retrievers that returned this document."""
        return tuple(c.source for c in self.contributions)

    def rank_in(self, source: str) -> int | None:
        """Return this document's rank in ``source``, or ``None`` if absent."""
        for contribution in self.contributions:
            if contribution.source == source:
                return contribution.rank
        return None
