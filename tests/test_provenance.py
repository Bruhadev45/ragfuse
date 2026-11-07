"""Tests for fuse/rerank: provenance, checked inputs and a closed candidate set."""

from __future__ import annotations

import itertools
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from ragfuse import (
    FusionTrace,
    IneligibleCandidateError,
    SourceTrace,
    combmnz,
    combsum,
    fuse,
    rerank,
    rrf,
)

# Two retrievers over one tenant's sections, as a legal pipeline would hand them over.
BM25 = [("act-12@v2#s4", 9.0), ("act-12@v2#s7", 6.5), ("reg-3@v1#s2", 2.0)]
DENSE = [("reg-3@v1#s2", 0.91), ("act-12@v2#s4", 0.88), ("act-9@v1#s1", 0.40)]
RANKINGS = {"bm25": BM25, "dense": DENSE}
ELIGIBLE = frozenset({"act-12@v2#s4", "act-12@v2#s7", "reg-3@v1#s2", "act-9@v1#s1"})
OTHER_TENANT = "tenant-b/act-12@v3#s4"


def _existing(method: str, rankings: dict[str, Any], **options: Any) -> list[Any]:
    if method == "rrf":
        return rrf({name: [doc for doc, _ in pairs] for name, pairs in rankings.items()}, **options)
    return (combsum if method == "combsum" else combmnz)(rankings, **options)


class TestFusionMatchesTheExistingApis:
    @pytest.mark.parametrize("method", ["rrf", "combsum", "combmnz"])
    @pytest.mark.parametrize("weights", [None, {"bm25": 0.7, "dense": 0.3}])
    def test_order_and_scores(self, method: str, weights: dict[str, float] | None) -> None:
        trace = fuse(RANKINGS, method=method, weights=weights)  # type: ignore[arg-type]
        hits = _existing(method, RANKINGS, weights=weights)
        assert trace.keys == tuple(hit.key for hit in hits)
        assert [c.fused_score for c in trace.candidates] == [hit.score for hit in hits]
        assert [c.normalized_score for c in trace.candidates] == [
            hit.normalized_score for hit in hits
        ]

    @pytest.mark.parametrize("method", ["rrf", "combsum", "combmnz"])
    def test_present_retrievers_match_the_hit_contributions(self, method: str) -> None:
        trace = fuse(RANKINGS, method=method)  # type: ignore[arg-type]
        for candidate, hit in zip(trace.candidates, _existing(method, RANKINGS), strict=True):
            present = {t.source: t for t in candidate.retrievers if t.rank is not None}
            assert {c.source: (c.rank, c.score) for c in hit.contributions} == {
                name: (t.rank, t.contribution) for name, t in present.items()
            }

    def test_effective_parameters_are_recorded(self) -> None:
        ranked = fuse(RANKINGS, k=10, normalize="zscore")
        scored = fuse(RANKINGS, method="combsum", k=10, normalize="zscore")
        assert (ranked.method, ranked.k, ranked.normalize) == ("rrf", 10, None)
        assert (scored.method, scored.k, scored.normalize) == ("combsum", None, "zscore")
        assert ranked.weights == (("bm25", 1.0), ("dense", 1.0))
        assert not ranked.eligible_only and not ranked.reranked


class TestEveryRetrieverIsAccountedFor:
    def test_rrf_records_raw_scores_values_and_absences(self) -> None:
        trace = fuse(RANKINGS, weights={"bm25": 2.0}, k=60)
        s7 = trace.candidates[trace.keys.index("act-12@v2#s7")]
        assert s7.retrievers == (
            SourceTrace("bm25", 1, 6.5, 1 / 62, 2.0, 2.0 / 62),
            SourceTrace("dense", None, None, None, 1.0, 0.0),
        )

    def test_score_fusion_values_are_the_normalised_scores(self) -> None:
        trace = fuse(RANKINGS, method="combsum", weights={"dense": 0.5})
        s2 = trace.candidates[trace.keys.index("reg-3@v1#s2")]
        bm25, dense = s2.retrievers
        assert (bm25.raw_score, bm25.value, bm25.contribution) == (2.0, 0.0, 0.0)
        assert (dense.raw_score, dense.value, dense.contribution) == (0.91, 1.0, 0.5)

    @pytest.mark.parametrize("method", ["rrf", "combsum"])
    def test_contributions_sum_to_the_fused_score(self, method: str) -> None:
        for candidate in fuse(RANKINGS, method=method).candidates:  # type: ignore[arg-type]
            total = sum(t.contribution for t in candidate.retrievers)
            assert total == pytest.approx(candidate.fused_score)

    def test_combmnz_multiplies_by_the_retrievers_that_found_it(self) -> None:
        for candidate in fuse(RANKINGS, method="combmnz").candidates:
            found = [t for t in candidate.retrievers if t.rank is not None]
            total = sum(t.contribution for t in found)
            assert candidate.fused_score == pytest.approx(total * len(found))

    def test_a_repeat_counts_once_at_its_first_position(self) -> None:
        trace = fuse({"r": [("a", 3.0), ("b", 2.0), ("a", 1.0)]})
        assert [
            (c.key, c.retrievers[0].rank, c.retrievers[0].raw_score) for c in trace.candidates
        ] == [
            ("a", 0, 3.0),
            ("b", 1, 2.0),
        ]

    def test_item_is_the_original_document(self) -> None:
        docs = {
            "bm25": [({"id": "a", "text": "x"}, 1.0)],
            "dense": [({"id": "a", "text": "x"}, 0.5)],
        }
        trace = fuse(docs, key=lambda doc: doc["id"])
        assert trace.candidates[0].item == {"id": "a", "text": "x"}
        hash(trace)  # an unhashable item must not make the trace unhashable

    def test_results_are_frozen(self) -> None:
        trace = fuse(RANKINGS)
        with pytest.raises(FrozenInstanceError):
            trace.candidates[0].rank = 5  # type: ignore[misc]


class TestStableIds:
    def test_mapping_order_does_not_change_the_trace(self) -> None:
        traces = {
            fuse(dict(order), method="combmnz")
            for order in itertools.permutations(
                {"bm25": BM25, "dense": DENSE, "sparse": [("act-9@v1#s1", 3.0)]}.items()
            )
        }
        assert len(traces) == 1

    def test_key_is_called_once_per_entry(self) -> None:
        calls: list[str] = []

        def key(doc: dict[str, str]) -> str:
            calls.append(doc["id"])
            return doc["id"]

        fuse({"r": [({"id": "a"}, 2.0), ({"id": "b"}, 1.0)], "s": [({"id": "a"}, 1.0)]}, key=key)
        assert sorted(calls) == ["a", "a", "b"]

    def test_sequence_weights_follow_the_mapping_order(self) -> None:
        trace = fuse({"zeta": [("a", 1.0)], "alpha": [("b", 1.0)]}, weights=[3.0, 1.0])
        assert trace.weights == (("alpha", 1.0), ("zeta", 3.0))
        assert trace.keys == ("a", "b")

    @pytest.mark.parametrize("bad", ["", " a", "a\n", "a\x00b", chr(0xA0)])
    def test_malformed_document_ids_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="id must be non-empty"):
            fuse({"r": [(bad, 1.0)]})

    def test_non_string_ids_need_a_key(self) -> None:
        with pytest.raises(TypeError, match="pass key="):
            fuse({"r": [(7, 1.0)]})  # type: ignore[list-item]

    def test_a_key_must_return_a_plain_str(self) -> None:
        class Id(str):
            pass

        with pytest.raises(TypeError, match="must be a str, got Id"):
            fuse({"r": [("a", 1.0)]}, key=Id)

    @pytest.mark.parametrize("bad", ["", "bm25 ", "a\tb"])
    def test_malformed_retriever_names_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="retriever name"):
            fuse({bad: [("a", 1.0)]})


class TestCheckedScores:
    @pytest.mark.parametrize("bad", [True, "1.5", None, [1.0]])
    def test_non_numeric_scores_are_rejected(self, bad: object) -> None:
        with pytest.raises(TypeError, match="real number"):
            fuse({"r": [("a", bad)]})  # type: ignore[list-item]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 10**400])
    def test_non_finite_scores_are_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            fuse({"r": [("a", 2.0), ("b", bad)]})

    def test_lists_must_be_best_first(self) -> None:
        with pytest.raises(ValueError, match="position 1 scores above"):
            fuse({"r": [("a", 0.2), ("b", 0.9)]})

    def test_ties_are_best_first(self) -> None:
        assert fuse({"r": [("b", 1.0), ("a", 1.0)]}).keys == ("b", "a")

    @pytest.mark.parametrize("entry", ["a", ("a",), ("a", 1.0, 2.0), 3])
    def test_entries_must_be_pairs(self, entry: object) -> None:
        with pytest.raises(TypeError, match=r"\(document, score\) pair"):
            fuse({"r": [entry]})  # type: ignore[list-item]

    def test_overflowing_weights_and_sums_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            fuse(RANKINGS, weights={"bm25": float("inf")})
        with pytest.raises(ValueError, match="not finite"):
            fuse({"a": [("x", 1e308)], "b": [("x", 1e308)]}, method="combsum", normalize=None)

    def test_zscore_overflow_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="overflowed during normalisation"):
            fuse({"r": [("x", 1e308), ("y", -1e308)]}, method="combsum", normalize="zscore")


class TestArguments:
    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            fuse(RANKINGS, method="borda")  # type: ignore[arg-type]

    def test_rankings_must_be_named(self) -> None:
        with pytest.raises(TypeError, match="map each retriever"):
            fuse([BM25, DENSE])  # type: ignore[arg-type]

    def test_rankings_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one ranked list"):
            fuse({})

    def test_a_ranking_must_be_a_sequence(self) -> None:
        with pytest.raises(TypeError, match="must be a sequence"):
            fuse({"r": "abc"})  # type: ignore[dict-item]

    def test_fusion_errors_pass_through(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            fuse(RANKINGS, k=0)
        with pytest.raises(ValueError, match="normalize must be one of"):
            fuse(RANKINGS, method="combsum", normalize="softmax")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="top_k"):
            fuse(RANKINGS, top_k=-1)

    def test_top_k_cuts_the_shortlist(self) -> None:
        assert fuse(RANKINGS, top_k=2).keys == fuse(RANKINGS).keys[:2]


class TestEligibility:
    def test_eligible_candidates_fuse(self) -> None:
        trace = fuse(RANKINGS, eligible=ELIGIBLE)
        assert trace.eligible_only and set(trace.keys) <= ELIGIBLE

    def test_any_other_id_fails_the_whole_fusion_without_echoing_it(self) -> None:
        leaked = {"bm25": BM25, "dense": [*DENSE, (OTHER_TENANT, 0.1)]}
        with pytest.raises(IneligibleCandidateError, match="'dense' position 3") as raised:
            fuse(leaked, eligible=ELIGIBLE)
        assert OTHER_TENANT not in str(raised.value)

    def test_an_empty_eligible_set_admits_nothing(self) -> None:
        with pytest.raises(IneligibleCandidateError):
            fuse(RANKINGS, eligible=frozenset())

    def test_a_single_string_is_not_an_eligible_set(self) -> None:
        with pytest.raises(TypeError, match="not one string"):
            fuse(RANKINGS, eligible="act-12@v2#s4")

    def test_eligible_ids_must_be_str(self) -> None:
        with pytest.raises(TypeError, match="must be str"):
            fuse(RANKINGS, eligible=[1, 2])  # type: ignore[list-item]


class TestRerank:
    def test_reorders_and_keeps_the_fusion_provenance(self) -> None:
        fused = fuse(RANKINGS, eligible=ELIGIBLE, top_k=3)
        scores = dict(zip(fused.keys, [-2.0, 0.5, 3.0], strict=True))
        reranked = rerank(fused, scores)
        assert reranked.keys == fused.keys[::-1]
        assert reranked.reranked and not fused.reranked
        for rank, candidate in enumerate(reranked.candidates):
            before = fused.candidates[candidate.fused_rank]
            assert candidate.rank == rank
            assert candidate.rerank_score == scores[candidate.key]
            assert (candidate.key, candidate.fused_score, candidate.retrievers) == (
                before.key,
                before.fused_score,
                before.retrievers,
            )

    def test_ties_keep_the_fused_order_and_top_k_cuts(self) -> None:
        fused = fuse(RANKINGS)
        reranked = rerank(fused, dict.fromkeys(fused.keys, 1.0), top_k=2)
        assert reranked.keys == fused.keys[:2]

    def test_an_empty_trace_reranks_to_an_empty_trace(self) -> None:
        assert rerank(fuse(RANKINGS, top_k=0), {}).candidates == ()

    @pytest.mark.parametrize("intruder", [OTHER_TENANT, 7])
    def test_cannot_introduce_a_candidate(self, intruder: object) -> None:
        fused = fuse(RANKINGS, eligible=ELIGIBLE, top_k=2)
        scores: dict[Any, float] = {**dict.fromkeys(fused.keys, 1.0), intruder: 99.0}
        with pytest.raises(IneligibleCandidateError, match="1 document") as raised:
            rerank(fused, scores)
        assert OTHER_TENANT not in str(raised.value)

    def test_cannot_promote_a_candidate_cut_by_top_k(self) -> None:
        fused = fuse(RANKINGS, top_k=2)
        cut = fuse(RANKINGS).keys[2]
        with pytest.raises(IneligibleCandidateError):
            rerank(fused, {**dict.fromkeys(fused.keys, 0.0), cut: 5.0})

    def test_every_candidate_must_be_scored(self) -> None:
        fused = fuse(RANKINGS)
        with pytest.raises(ValueError, match="left 1 candidate"):
            rerank(fused, dict.fromkeys(fused.keys[1:], 1.0))

    @pytest.mark.parametrize(
        ("bad", "error"),
        [
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (True, TypeError),
            ("1", TypeError),
        ],
    )
    def test_scores_are_checked(self, bad: object, error: type[Exception]) -> None:
        fused = fuse(RANKINGS, top_k=1)
        with pytest.raises(error):
            rerank(fused, {fused.keys[0]: bad})  # type: ignore[dict-item]

    def test_only_a_fresh_fusion_trace_reranks(self) -> None:
        fused = fuse(RANKINGS)
        once = rerank(fused, dict.fromkeys(fused.keys, 1.0))
        with pytest.raises(ValueError, match="already reranked"):
            rerank(once, dict.fromkeys(fused.keys, 1.0))
        with pytest.raises(TypeError, match="FusionTrace"):
            rerank(_existing("rrf", RANKINGS), {})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must map"):
            rerank(fused, list(fused.keys))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-1, True, 1.5])
    def test_top_k_must_be_a_non_negative_int(self, bad: object) -> None:
        fused = fuse(RANKINGS)
        with pytest.raises(ValueError, match="top_k"):
            rerank(fused, dict.fromkeys(fused.keys, 1.0), top_k=bad)  # type: ignore[arg-type]

    def test_returns_a_new_trace(self) -> None:
        fused = fuse(RANKINGS)
        reranked = rerank(fused, dict.fromkeys(fused.keys, 1.0))
        assert isinstance(reranked, FusionTrace) and reranked is not fused
        assert all(c.rerank_score is None for c in fused.candidates)
