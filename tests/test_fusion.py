"""Tests for weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragfuse import Contribution, FusedHit, reciprocal_rank_fusion, rrf


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str


class TestBasicFusion:
    def test_single_ranking_preserves_order(self) -> None:
        hits = reciprocal_rank_fusion({"only": ["a", "b", "c"]})
        assert [hit.key for hit in hits] == ["a", "b", "c"]

    def test_scores_follow_the_rrf_formula(self) -> None:
        hits = reciprocal_rank_fusion({"only": ["a", "b"]}, k=60)
        assert hits[0].score == pytest.approx(1 / 61)
        assert hits[1].score == pytest.approx(1 / 62)

    def test_document_in_both_lists_outranks_one_in_either(self) -> None:
        hits = reciprocal_rank_fusion(
            {"bm25": ["shared", "only_bm25"], "dense": ["only_dense", "shared"]}
        )
        assert hits[0].key == "shared"

    def test_sequence_input_is_auto_named(self) -> None:
        hits = reciprocal_rank_fusion([["a"], ["b"]])
        assert {hit.sources[0] for hit in hits} == {"0", "1"}

    def test_top_k_truncates(self) -> None:
        hits = reciprocal_rank_fusion({"r": ["a", "b", "c", "d"]}, top_k=2)
        assert len(hits) == 2

    def test_top_k_zero_returns_nothing(self) -> None:
        assert reciprocal_rank_fusion({"r": ["a", "b"]}, top_k=0) == []

    def test_empty_rankings_produce_no_hits(self) -> None:
        assert reciprocal_rank_fusion({"a": [], "b": []}) == []

    def test_rrf_is_an_alias(self) -> None:
        assert rrf is reciprocal_rank_fusion


class TestWeighting:
    """The behaviour the duplicate-the-list workaround cannot express."""

    def test_weights_change_the_winner(self) -> None:
        """Between two docs each found by one retriever, the weight decides."""
        rankings = {"bm25": ["from_bm25"], "dense": ["from_dense"]}

        bm25_favoured = reciprocal_rank_fusion(rankings, weights={"bm25": 0.9, "dense": 0.1})
        dense_favoured = reciprocal_rank_fusion(rankings, weights={"bm25": 0.1, "dense": 0.9})

        assert bm25_favoured[0].key == "from_bm25"
        assert dense_favoured[0].key == "from_dense"

    def test_agreement_can_outweigh_a_heavy_weight(self) -> None:
        """A document both retrievers return beats one only the favoured
        retriever returns -- this is the whole point of fusing rather than
        just trusting the stronger retriever."""
        hits = reciprocal_rank_fusion(
            {"bm25": ["only_bm25", "agreed"], "dense": ["only_dense", "agreed"]},
            weights={"bm25": 0.9, "dense": 0.1},
        )
        assert hits[0].key == "agreed"

    def test_fractional_weights_are_not_rounded_to_integers(self) -> None:
        """0.6/0.4 must differ from 1/1 -- the double-count hack cannot do this."""
        rankings = {"bm25": ["a", "b"], "dense": ["b", "a"]}

        equal = reciprocal_rank_fusion(rankings, weights={"bm25": 1.0, "dense": 1.0})
        tilted = reciprocal_rank_fusion(rankings, weights={"bm25": 0.6, "dense": 0.4})

        assert equal[0].score == pytest.approx(equal[1].score)
        assert tilted[0].key == "a"
        assert tilted[0].score > tilted[1].score

    def test_duplicating_a_list_equals_weight_two(self) -> None:
        """Weight 2.0 reproduces the old trick: same order, scores equal up to float rounding."""
        hack = reciprocal_rank_fusion(
            {"bm25": ["a", "b"], "bm25_again": ["a", "b"], "dense": ["b", "a"]}
        )
        weighted = reciprocal_rank_fusion(
            {"bm25": ["a", "b"], "dense": ["b", "a"]},
            weights={"bm25": 2.0, "dense": 1.0},
        )
        assert [h.key for h in hack] == [h.key for h in weighted]
        assert hack[0].score == pytest.approx(weighted[0].score)

    def test_zero_weight_silences_a_retriever(self) -> None:
        hits = reciprocal_rank_fusion(
            {"good": ["a"], "ignored": ["b"]}, weights={"good": 1.0, "ignored": 0.0}
        )
        assert hits[0].key == "a"
        assert hits[1].score == 0.0

    def test_sequence_weights_align_with_sequence_rankings(self) -> None:
        hits = reciprocal_rank_fusion([["a"], ["b"]], weights=[0.9, 0.1])
        assert hits[0].key == "a"

    def test_partial_weight_mapping_defaults_the_rest_to_one(self) -> None:
        hits = reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, weights={"a": 5.0})
        assert hits[0].key == "x"
        assert hits[1].contributions[0].weight == 1.0


class TestProvenance:
    def test_contributions_record_every_source(self) -> None:
        hits = reciprocal_rank_fusion({"bm25": ["a"], "dense": ["a"]})
        assert set(hits[0].sources) == {"bm25", "dense"}

    def test_rank_in_reports_position_per_source(self) -> None:
        hits = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "a"]})
        hit_a = next(hit for hit in hits if hit.key == "a")
        assert hit_a.rank_in("bm25") == 0
        assert hit_a.rank_in("dense") == 1
        assert hit_a.rank_in("nonexistent") is None

    def test_contribution_score_matches_formula(self) -> None:
        hits = reciprocal_rank_fusion({"r": ["a"]}, k=10, weights={"r": 0.5})
        assert hits[0].contributions[0] == Contribution(
            source="r", rank=0, weight=0.5, score=pytest.approx(0.5 / 11)
        )

    def test_normalized_score_tops_out_at_one(self) -> None:
        hits = reciprocal_rank_fusion({"r": ["a", "b", "c"]})
        assert hits[0].normalized_score == pytest.approx(1.0)
        assert all(0.0 <= hit.normalized_score <= 1.0 for hit in hits)

    def test_item_comes_from_the_strongest_contributor(self) -> None:
        weak = Doc("d1", "from the weak retriever")
        strong = Doc("d1", "from the strong retriever")
        hits = reciprocal_rank_fusion(
            {"weak": [weak], "strong": [strong]},
            weights={"weak": 0.1, "strong": 0.9},
            key=lambda d: d.doc_id,
        )
        assert hits[0].item is strong

    def test_hits_with_dict_items_are_hashable(self) -> None:
        """Regression: item may be any object (often a dict); it must not
        make the hit unusable in sets and dict keys."""
        hits = reciprocal_rank_fusion({"r": [{"id": "a"}, {"id": "b"}]}, key=lambda d: d["id"])
        assert len({hits[0], hits[1], hits[0]}) == 2
        assert {hits[0]: "cached"}[hits[0]] == "cached"

    def test_hits_are_immutable(self) -> None:
        hit = reciprocal_rank_fusion({"r": ["a"]})[0]
        with pytest.raises((AttributeError, TypeError)):
            hit.score = 99.0  # type: ignore[misc]


class TestCustomDocuments:
    def test_key_extracts_identifiers_from_objects(self) -> None:
        hits = reciprocal_rank_fusion(
            {"r": [Doc("a", "alpha"), Doc("b", "beta")]}, key=lambda d: d.doc_id
        )
        assert [hit.key for hit in hits] == ["a", "b"]

    def test_objects_without_key_raise_a_helpful_error(self) -> None:
        with pytest.raises(TypeError, match="pass key="):
            reciprocal_rank_fusion({"r": [Doc("a", "alpha")]})

    def test_dict_documents_work_via_key(self) -> None:
        hits = reciprocal_rank_fusion({"r": [{"id": "a"}, {"id": "b"}]}, key=lambda d: d["id"])
        assert [hit.key for hit in hits] == ["a", "b"]


class TestDeterminismAndDuplicates:
    def test_ties_break_on_key_so_output_is_stable(self) -> None:
        first = reciprocal_rank_fusion({"r": ["b"], "s": ["a"]})
        second = reciprocal_rank_fusion({"s": ["a"], "r": ["b"]})
        assert [hit.key for hit in first] == [hit.key for hit in second] == ["a", "b"]

    def test_repeat_within_one_list_counts_once_at_its_best_rank(self) -> None:
        repeated = reciprocal_rank_fusion({"r": ["a", "b", "a"]})
        clean = reciprocal_rank_fusion({"r": ["a", "b"]})
        assert [h.key for h in repeated] == [h.key for h in clean]
        assert repeated[0].score == pytest.approx(clean[0].score)
        assert len(repeated[0].contributions) == 1

    def test_documents_after_a_duplicate_keep_their_unique_rank(self) -> None:
        """Regression: collapsing chunks to a parent doc via key= must not
        push later documents down by the number of skipped duplicates."""
        chunked = reciprocal_rank_fusion(
            {"dense": ["docA#1", "docA#2", "docA#3", "docB#1"]},
            key=lambda c: c.split("#")[0],
        )
        flat = reciprocal_rank_fusion({"dense": ["docA", "docB"]})
        doc_b = next(h for h in chunked if h.key == "docB")
        assert doc_b.contributions[0].rank == 1
        assert doc_b.score == pytest.approx(flat[1].score)

    def test_larger_k_flattens_the_gap_between_ranks(self) -> None:
        def gap(k: int) -> float:
            hits = reciprocal_rank_fusion({"r": ["a", "b"]}, k=k)
            return hits[0].score - hits[1].score

        assert gap(10) > gap(1000)


class TestValidation:
    @pytest.mark.parametrize("bad_k", [0, -1, -60])
    def test_non_positive_k_is_rejected(self, bad_k: int) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            reciprocal_rank_fusion({"r": ["a"]}, k=bad_k)

    def test_negative_top_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            reciprocal_rank_fusion({"r": ["a"]}, top_k=-1)

    def test_no_rankings_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one ranked list"):
            reciprocal_rank_fusion({})

    def test_negative_weight_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            reciprocal_rank_fusion({"r": ["a"]}, weights={"r": -1.0})

    def test_nan_weight_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            reciprocal_rank_fusion({"r": ["a"]}, weights={"r": float("nan")})

    def test_all_zero_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than zero"):
            reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, weights={"a": 0, "b": 0})

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), 10**400, -(10**400)])
    def test_infinite_weight_is_rejected(self, bad: float) -> None:
        # inf / (k + rank + 1) is inf, and inf / inf left normalized_score NaN.
        with pytest.raises(ValueError, match="infinite"):
            reciprocal_rank_fusion({"r": ["a", "b"]}, weights={"r": bad})

    def test_unknown_weight_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown rankings"):
            reciprocal_rank_fusion({"r": ["a"]}, weights={"typo": 1.0})

    def test_wrong_number_of_sequence_weights_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights for"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_a_bare_string_is_not_a_ranking(self) -> None:
        with pytest.raises(TypeError, match="must be a sequence"):
            reciprocal_rank_fusion({"r": "abc"})

    def test_a_bare_string_is_not_a_set_of_rankings(self) -> None:
        with pytest.raises(TypeError, match="mapping or a sequence"):
            reciprocal_rank_fusion("abc")


class TestRealisticScenario:
    """The JurisGPT shape: lexical and BM25 lists over statute sections."""

    def test_hybrid_promotes_what_both_retrievers_agree_on(self) -> None:
        """Neither retriever ranks the right statute first, but both rank it
        second -- fusion lifts it above two distractors that each won only one
        list. This is the case a single retriever cannot get right."""
        relevant = "sec-149"
        bm25 = ["sec-135", relevant, "sec-188"]
        lexical = ["sec-203", relevant, "sec-511"]

        fused = reciprocal_rank_fusion(
            {"bm25": bm25, "lexical": lexical},
            weights={"bm25": 0.6, "lexical": 0.4},
        )

        assert bm25[0] != relevant and lexical[0] != relevant
        assert fused[0].key == relevant
        assert fused[0].sources == ("bm25", "lexical")

    def test_fused_hit_type_is_exported(self) -> None:
        hits = reciprocal_rank_fusion({"r": ["a"]})
        assert isinstance(hits[0], FusedHit)
