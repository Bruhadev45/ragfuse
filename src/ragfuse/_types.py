"""Immutable result types returned by the fusion functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Contribution", "FusedHit"]


@dataclass(frozen=True, slots=True)
class Contribution:
    """What a single retriever contributed to one fused hit.

    Attributes:
        source: Name of the retriever this contribution came from.
        rank: Zero-indexed position the document held in that retriever's list.
        weight: Weight applied to that retriever during fusion.
        score: The weighted term this retriever added: ``weight / (k + rank + 1)``
            for RRF, ``weight`` times the normalised retriever score for
            CombSUM and CombMNZ.
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
        score: The fused score: the sum of every contribution (times the
            number of retrievers that found the document, for CombMNZ).
        normalized_score: ``score`` relative to the top hit, in ``0..1``:
            divided by the top score, or min-max scaled across the hits when
            score fusion left some totals negative.
        contributions: Per-retriever detail, ordered by descending contribution.
        item: The original object, taken from the retriever that contributed most.
    """

    key: str
    score: float
    normalized_score: float
    contributions: tuple[Contribution, ...]
    # Excluded from the hash: item may be any object the caller's retrievers
    # returned (often a dict), and an unhashable item must not make the whole
    # hit unusable in sets and dict keys. Equality still compares it.
    item: Any = field(default=None, hash=False)

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
