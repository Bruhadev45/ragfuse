"""Tests for the retrieval metrics."""

from __future__ import annotations

from math import log2

import pytest

from ragfuse import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "x"], {"a"}, k=2) == 0.5

    def test_short_list_is_not_penalised_for_missing_slots(self) -> None:
        assert precision_at_k(["a"], {"a"}, k=5) == 1.0

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_only_counts_the_top_k(self) -> None:
        assert precision_at_k(["x", "a"], {"a"}, k=1) == 0.0


class TestRecallAtK:
    def test_finds_everything(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_finds_half(self) -> None:
        assert recall_at_k(["a", "x"], {"a", "b"}, k=5) == 0.5

    def test_cutoff_excludes_late_hits(self) -> None:
        assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_no_relevant_documents_scores_zero(self) -> None:
        assert recall_at_k(["a"], set(), k=5) == 0.0

    def test_duplicates_do_not_inflate(self) -> None:
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("retrieved", "expected"),
        [(["a", "x", "y"], 1.0), (["x", "a", "y"], 0.5), (["x", "y", "a"], 1 / 3)],
    )
    def test_position_of_first_hit(self, retrieved: list[str], expected: float) -> None:
        assert reciprocal_rank(retrieved, {"a"}) == pytest.approx(expected)

    def test_no_hit_scores_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_list_scores_zero(self) -> None:
        assert reciprocal_rank([], {"a"}) == 0.0


class TestMeanReciprocalRank:
    def test_averages_across_queries(self) -> None:
        result = mean_reciprocal_rank([["a"], ["x", "b"]], [{"a"}, {"b"}])
        assert result == pytest.approx((1.0 + 0.5) / 2)

    def test_empty_input(self) -> None:
        assert mean_reciprocal_rank([], []) == 0.0

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="label sets"):
            mean_reciprocal_rank([["a"], ["b"]], [{"a"}])


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_nothing_relevant_scores_zero(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_no_labels_scores_zero(self) -> None:
        assert ndcg_at_k(["a"], set(), k=2) == 0.0

    def test_earlier_hit_scores_higher(self) -> None:
        assert ndcg_at_k(["a", "x"], {"a"}, k=2) > ndcg_at_k(["x", "a"], {"a"}, k=2)

    def test_matches_hand_computed_value(self) -> None:
        # One relevant document at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2).
        expected = (1 / log2(3)) / (1 / log2(2))
        assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected)

    def test_stays_within_unit_range(self) -> None:
        value = ndcg_at_k(["x", "a", "b"], {"a", "b", "c"}, k=3)
        assert 0.0 <= value <= 1.0


class TestSharedValidation:
    @pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, ndcg_at_k])
    @pytest.mark.parametrize("bad_k", [0, -1])
    def test_non_positive_k_is_rejected(self, metric, bad_k: int) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            metric(["a"], {"a"}, k=bad_k)

    @pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, ndcg_at_k])
    def test_relevant_accepts_any_iterable(self, metric) -> None:
        assert metric(["a"], ["a"], k=1) == metric(["a"], {"a"}, k=1)
