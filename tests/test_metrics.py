"""Tests for the retrieval metrics."""

from __future__ import annotations

import random
from math import log2

import pytest

from ragfuse import (
    PRECISION_MODES,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

UNIVERSE = [f"doc-{index}" for index in range(8)]


def _random_cases(count: int = 50) -> list[tuple[list[str], set[str], int]]:
    """Deterministic (retrieved, relevant, k) triples for the property tests.

    Deliberately generates lists shorter than, equal to and longer than ``k``,
    plus duplicates and empty label sets, since that is exactly where the two
    precision denominators part ways.
    """
    rng = random.Random(20260912)
    cases: list[tuple[list[str], set[str], int]] = []
    for _ in range(count):
        retrieved = [rng.choice(UNIVERSE) for _ in range(rng.randint(0, 10))]
        relevant = set(rng.sample(UNIVERSE, rng.randint(0, len(UNIVERSE))))
        cases.append((retrieved, relevant, rng.randint(1, 10)))
    return cases


CASES = _random_cases()
FILLED_CASES = [case for case in CASES if len(set(case[0])) >= case[2]]
LABELLED_CASES = [case for case in CASES if case[1]]
assert FILLED_CASES and LABELLED_CASES, "generated cases must exercise both properties"


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

    def test_duplicates_are_one_hit(self) -> None:
        assert precision_at_k(["a", "a", "x"], {"a"}, k=3) == 0.5


class TestPrecisionAtKTrecMode:
    def test_short_list_is_penalised_for_missing_slots(self) -> None:
        # The divergence in one line: same input, 0.5 by the default denominator.
        assert precision_at_k(["a", "b"], {"a"}, k=5, mode="trec") == 0.2

    def test_full_list_matches_the_default_mode(self) -> None:
        retrieved = ["a", "x", "b", "y", "z"]
        assert precision_at_k(retrieved, {"a", "b"}, k=5, mode="trec") == pytest.approx(
            precision_at_k(retrieved, {"a", "b"}, k=5)
        )

    def test_empty_retrieved_scores_zero(self) -> None:
        assert precision_at_k([], {"a"}, k=5, mode="trec") == 0.0

    def test_duplicates_do_not_fill_slots(self) -> None:
        # Three results, one distinct document, one hit: 1/5, not 3/5.
        assert precision_at_k(["a", "a", "a"], {"a"}, k=5, mode="trec") == 0.2

    def test_explicit_default_mode_is_the_default(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}, k=5, mode="retrieved") == precision_at_k(
            ["a", "b"], {"a"}, k=5
        )

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown mode 'strict'"):
            precision_at_k(["a"], {"a"}, k=1, mode="strict")

    def test_error_names_the_valid_modes(self) -> None:
        with pytest.raises(ValueError, match=r"\['retrieved', 'trec'\]"):
            precision_at_k(["a"], {"a"}, k=1, mode="")

    def test_modes_constant_matches_what_is_accepted(self) -> None:
        assert set(PRECISION_MODES) == {"retrieved", "trec"}
        for mode in PRECISION_MODES:
            assert precision_at_k(["a"], {"a"}, k=1, mode=mode) == 1.0


class TestPrecisionAtKProperties:
    """Invariants that must hold for every input, checked over generated cases."""

    @pytest.mark.parametrize(("retrieved", "relevant", "k"), CASES)
    def test_both_modes_stay_within_unit_range(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> None:
        for mode in ("retrieved", "trec"):
            value = precision_at_k(retrieved, relevant, k, mode=mode)
            assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize(("retrieved", "relevant", "k"), CASES)
    def test_trec_never_scores_higher(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> None:
        trec = precision_at_k(retrieved, relevant, k, mode="trec")
        default = precision_at_k(retrieved, relevant, k)
        assert trec < default or trec == pytest.approx(default)

    @pytest.mark.parametrize(("retrieved", "relevant", "k"), FILLED_CASES)
    def test_modes_agree_when_the_list_fills_the_cutoff(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> None:
        assert precision_at_k(retrieved, relevant, k, mode="trec") == pytest.approx(
            precision_at_k(retrieved, relevant, k)
        )

    @pytest.mark.parametrize(("retrieved", "relevant", "k"), CASES)
    def test_trec_rescales_the_default_by_the_filled_fraction(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> None:
        filled = min(k, len(set(retrieved)))
        expected = precision_at_k(retrieved, relevant, k) * filled / k
        assert precision_at_k(retrieved, relevant, k, mode="trec") == pytest.approx(expected)

    @pytest.mark.parametrize(("retrieved", "relevant", "k"), CASES)
    def test_trec_numerator_is_a_whole_number_of_hits(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> None:
        hits = precision_at_k(retrieved, relevant, k, mode="trec") * k
        top = list(dict.fromkeys(retrieved))[:k]
        assert hits == pytest.approx(sum(1 for item in top if item in relevant))

    @pytest.mark.parametrize(("retrieved", "relevant", "k"), LABELLED_CASES)
    def test_trec_agrees_with_recall_on_the_hit_count(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> None:
        # Only for labelled queries: recall_at_k is defined as 0.0 with no labels.
        from_precision = precision_at_k(retrieved, relevant, k, mode="trec") * k
        from_recall = recall_at_k(retrieved, relevant, k) * len(relevant)
        assert from_precision == pytest.approx(from_recall)

    @pytest.mark.parametrize(("retrieved", "relevant", "k"), CASES)
    def test_repeating_the_results_changes_nothing(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> None:
        for mode in ("retrieved", "trec"):
            assert precision_at_k(retrieved * 2, relevant, k, mode=mode) == pytest.approx(
                precision_at_k(retrieved, relevant, k, mode=mode)
            )


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
