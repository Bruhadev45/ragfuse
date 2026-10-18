"""Tests for score-based fusion: CombSUM and CombMNZ."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragfuse import (
    DEFAULT_K,
    NORMALIZATIONS,
    FusedHit,
    combmnz,
    combsum,
    reciprocal_rank_fusion,
)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str


def _as_rrf_scores(ranking: list[str], k: int = DEFAULT_K) -> list[tuple[str, float]]:
    """Give each document the score RRF would: ``1 / (k + rank + 1)``."""
    return [(doc, 1 / (k + rank + 1)) for rank, doc in enumerate(ranking)]


def _same_hits(left: list[FusedHit], right: list[FusedHit]) -> None:
    assert [h.key for h in left] == [h.key for h in right]
    for a, b in zip(left, right, strict=True):
        assert a.score == pytest.approx(b.score)
        assert a.normalized_score == pytest.approx(b.normalized_score)
        assert a.sources == b.sources
        for ca, cb in zip(a.contributions, b.contributions, strict=True):
            assert (ca.source, ca.rank, ca.weight) == (cb.source, cb.rank, cb.weight)
            assert ca.score == pytest.approx(cb.score)


BM25 = ["sec-149", "sec-135", "sec-188", "sec-302"]
DENSE = ["sec-135", "sec-149", "sec-203"]


class TestParityWithRRF:
    """Fed RRF's own scores un-normalised, CombSUM *is* weighted RRF."""

    @pytest.mark.parametrize(
        "weights", [None, {"bm25": 0.6, "dense": 0.4}, {"bm25": 2.0}, {"dense": 0.0}]
    )
    def test_combsum_on_rrf_scores_matches_rrf(self, weights: dict[str, float] | None) -> None:
        expected = reciprocal_rank_fusion({"bm25": BM25, "dense": DENSE}, weights=weights)
        scored = {"bm25": _as_rrf_scores(BM25), "dense": _as_rrf_scores(DENSE)}
        _same_hits(combsum(scored, normalize=None, weights=weights), expected)

    def test_one_list_combmnz_matches_combsum_and_rrf(self) -> None:
        scored = {"only": _as_rrf_scores(BM25)}
        expected = reciprocal_rank_fusion({"only": BM25})
        _same_hits(combsum(scored, normalize=None), expected)
        _same_hits(combmnz(scored, normalize=None), expected)

    def test_sequence_input_is_auto_named_like_rrf(self) -> None:
        hits = combsum([[("a", 1.0)], [("b", 1.0)]])
        assert {hit.sources[0] for hit in hits} == {"0", "1"}


class TestCombSUM:
    def test_minmax_puts_every_list_on_zero_to_one(self) -> None:
        hits = combsum({"bm25": [("a", 12.0), ("b", 9.0), ("c", 3.0)]})
        assert [(h.key, h.score) for h in hits] == [
            ("a", 1.0),
            ("b", pytest.approx(2 / 3)),
            ("c", 0.0),
        ]

    def test_scores_beat_ranks_on_a_near_tie(self) -> None:
        """RRF reads 0.80 vs 0.79 as a full rank apart; score fusion does not."""
        scored = {
            "bm25": [("a", 10.0), ("b", 0.0)],
            "dense": [("b", 0.80), ("a", 0.79), ("c", 0.0)],
        }
        ranks = reciprocal_rank_fusion(
            {name: [d for d, _ in pairs] for name, pairs in scored.items()}
        )
        assert ranks[0].score == ranks[1].score  # a and b tie on rank alone
        assert combsum(scored)[0].key == "a"
        assert combsum(scored)[0].score == pytest.approx(1 + 0.79 / 0.80)

    def test_weights_scale_each_contribution(self) -> None:
        hits = combsum(
            {"bm25": [("a", 2.0), ("b", 1.0)], "dense": [("b", 5.0), ("a", 4.0)]},
            weights={"bm25": 0.25, "dense": 0.75},
        )
        assert [h.key for h in hits] == ["b", "a"]
        assert hits[0].score == pytest.approx(0.75)
        assert {c.source: c.score for c in hits[1].contributions} == {"bm25": 0.25, "dense": 0.0}

    def test_contributions_carry_rank_and_weight(self) -> None:
        hit = combsum({"bm25": [("a", 3.0), ("b", 1.0), ("c", 0.0)], "dense": [("b", 1.0)]})[0]
        assert hit.key == "b"
        assert hit.rank_in("bm25") == 1
        assert hit.rank_in("dense") == 0
        assert hit.contributions[0].source == "dense"

    def test_key_extracts_identifiers_and_keeps_the_best_item(self) -> None:
        strong = Doc("sec-149", "from dense")
        hits = combsum(
            {
                "bm25": [(Doc("sec-149", "from bm25"), 1.0), (Doc("x", ""), 2.0)],
                "dense": [(strong, 1.0)],
            },
            key=lambda doc: doc.doc_id,
        )
        top = next(h for h in hits if h.key == "sec-149")
        assert top.item is strong

    def test_repeat_within_one_list_counts_once_at_its_first_position(self) -> None:
        hits = combsum({"r": [("a", 3.0), ("b", 2.0), ("a", 9.0), ("c", 1.0)]}, normalize=None)
        assert [(h.key, h.score) for h in hits] == [("a", 3.0), ("b", 2.0), ("c", 1.0)]
        assert hits[2].rank_in("r") == 2

    def test_top_k_truncates_and_ties_break_on_key(self) -> None:
        hits = combsum({"a": [("y", 1.0)], "b": [("x", 1.0)], "c": [("z", 0.5)]}, top_k=2)
        assert [h.key for h in hits] == ["x", "y"]

    def test_empty_lists_produce_no_hits(self) -> None:
        assert combsum({"a": [], "b": []}) == []
        assert combmnz({"a": []}) == []


class TestCombMNZ:
    def test_agreement_multiplies_the_sum(self) -> None:
        scored = {
            "bm25": [("solo", 1.0), ("both", 0.5), ("z", 0.0)],
            "dense": [("both", 1.0), ("w", 0.0)],
        }
        sums = {h.key: h.score for h in combsum(scored)}
        mnzs = {h.key: h.score for h in combmnz(scored)}
        assert mnzs["both"] == pytest.approx(2 * sums["both"])
        assert mnzs["solo"] == pytest.approx(sums["solo"])

    def test_agreement_can_overturn_combsum(self) -> None:
        scored = {
            "bm25": [("x", 12.0), ("y", 6.0), ("z", 0.0)],
            "dense": [("w", 0.9), ("y", 0.3), ("v", -0.1)],
        }
        assert combsum(scored)[0].key != "y"
        assert combmnz(scored)[0].key == "y"

    def test_zero_weight_retriever_does_not_vote(self) -> None:
        scored = {"bm25": [("a", 1.0), ("b", 0.0)], "dense": [("b", 1.0), ("a", 0.5)]}
        muted = combmnz(scored, weights={"dense": 0.0})
        alone = combmnz({"bm25": [("a", 1.0), ("b", 0.0)]})
        assert [(h.key, h.score) for h in muted] == [(h.key, h.score) for h in alone]
        assert muted[0].sources == ("bm25", "dense")


class TestNormalisationEdgeCases:
    def test_constant_scores_minmax_to_one(self) -> None:
        hits = combsum({"r": [("a", 0.7), ("b", 0.7)]})
        assert [h.score for h in hits] == [1.0, 1.0]

    def test_single_hit_minmax_to_one(self) -> None:
        assert combsum({"r": [("a", -3.0)]})[0].score == 1.0

    def test_constant_scores_zscore_to_zero(self) -> None:
        hits = combsum({"r": [("a", 4.0), ("b", 4.0)]}, normalize="zscore")
        assert [h.score for h in hits] == [0.0, 0.0]

    def test_single_hit_zscore_to_zero(self) -> None:
        assert combsum({"r": [("a", 4.0)]}, normalize="zscore")[0].score == 0.0

    def test_zscore_centres_and_scales_each_list(self) -> None:
        hits = combsum({"r": [("a", 3.0), ("b", 1.0)]}, normalize="zscore")
        assert [(h.key, h.score) for h in hits] == [("a", 1.0), ("b", -1.0)]

    def test_zscore_makes_lists_on_different_scales_comparable(self) -> None:
        hits = combsum(
            {"bm25": [("a", 30.0), ("b", 10.0)], "dense": [("b", 0.9), ("a", 0.7)]},
            normalize="zscore",
        )
        assert [h.score for h in hits] == [pytest.approx(0.0, abs=1e-12)] * 2

    def test_normalisation_ignores_a_repeat_entry(self) -> None:
        # The repeated "a" at 100.0 is dropped before scaling, so it cannot
        # stretch the range and squash "b".
        hits = combsum({"r": [("a", 2.0), ("b", 1.5), ("a", 100.0), ("c", 1.0)]})
        assert {h.key: h.score for h in hits} == {"a": 1.0, "b": 0.5, "c": 0.0}

    def test_none_keeps_raw_scores(self) -> None:
        hits = combsum({"a": [("x", 30.0)], "b": [("x", 0.5)]}, normalize=None)
        assert hits[0].score == 30.5

    def test_negative_totals_min_max_into_normalized_score(self) -> None:
        hits = combsum({"r": [("a", 3.0), ("b", 2.0), ("c", 1.0)]}, normalize="zscore")
        assert [h.normalized_score for h in hits] == [1.0, pytest.approx(0.5), 0.0]

    def test_all_equal_negative_totals_normalise_to_one(self) -> None:
        hits = combsum({"r": [("a", -2.0), ("b", -2.0)]}, normalize=None)
        assert [h.normalized_score for h in hits] == [1.0, 1.0]

    def test_non_negative_totals_keep_the_ratio_to_the_top(self) -> None:
        hits = combsum({"r": [("a", 4.0), ("b", 1.0)]}, normalize=None)
        assert [h.normalized_score for h in hits] == [1.0, 0.25]

    def test_every_option_is_listed(self) -> None:
        assert NORMALIZATIONS == ("minmax", "zscore", None)


class TestValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores_are_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            combsum({"r": [("a", 1.0), ("b", bad)]})

    def test_non_numeric_score_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="not a number"):
            combsum({"r": [("a", "high")]})  # type: ignore[list-item]

    @pytest.mark.parametrize("entry", ["a", ("a",), ("a", 1.0, 2.0), 3])
    def test_entries_must_be_pairs(self, entry: object) -> None:
        with pytest.raises(TypeError, match=r"\(document, score\) pairs"):
            combsum({"r": [entry]})  # type: ignore[list-item]

    def test_unknown_normalisation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="normalize must be one of"):
            combsum({"r": [("a", 1.0)]}, normalize="softmax")  # type: ignore[arg-type]

    def test_negative_top_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            combmnz({"r": [("a", 1.0)]}, top_k=-1)

    def test_objects_without_key_raise_a_helpful_error(self) -> None:
        with pytest.raises(TypeError, match="pass key="):
            combsum({"r": [(Doc("a", ""), 1.0)]})

    def test_weight_rules_are_shared_with_rrf(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            combsum({"r": [("a", 1.0)]}, weights={"r": -1.0})
        with pytest.raises(ValueError, match="unknown rankings"):
            combmnz({"r": [("a", 1.0)]}, weights={"nope": 1.0})
