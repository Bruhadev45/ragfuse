"""Tests for weight tuning."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragfuse import METRICS, WeightSearchResult, evaluate_weights, tune_weights


@pytest.fixture
def bm25_is_better() -> tuple[list[dict[str, list[str]]], list[set[str]]]:
    """Three queries where BM25 ranks the relevant doc first and dense does not."""
    rankings = [
        {"bm25": ["rel", "x", "y"], "dense": ["x", "y", "rel"]},
        {"bm25": ["rel2", "p", "q"], "dense": ["p", "q", "rel2"]},
        {"bm25": ["rel3", "m", "n"], "dense": ["m", "n", "rel3"]},
    ]
    labels = [{"rel"}, {"rel2"}, {"rel3"}]
    return rankings, labels


class TestTuneWeights:
    def test_favours_the_stronger_retriever(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        result = tune_weights(rankings, labels, top_k=3, resolution=10)
        assert result.weights["bm25"] > result.weights["dense"]

    def test_returns_a_result_object(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        result = tune_weights(rankings, labels, top_k=3)
        assert isinstance(result, WeightSearchResult)
        assert result.metric == "ndcg"
        assert result.top_k == 3

    def test_weights_form_a_distribution(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        result = tune_weights(rankings, labels, top_k=3, resolution=10)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_table_is_sorted_best_first(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        result = tune_weights(rankings, labels, top_k=3)
        scores = [score for _, score in result.table]
        assert scores == sorted(scores, reverse=True)
        assert result.score == pytest.approx(scores[0])

    def test_tuning_never_scores_below_equal_weights(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        result = tune_weights(rankings, labels, top_k=3, resolution=10)
        assert result.improvement() >= 0

    def test_resolution_controls_the_grid_size(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        coarse = tune_weights(rankings, labels, top_k=3, resolution=2)
        fine = tune_weights(rankings, labels, top_k=3, resolution=10)
        assert len(fine.table) > len(coarse.table)

    def test_baseline_is_finite_at_every_resolution(self, bm25_is_better) -> None:
        # The equal-weight baseline is evaluated directly on the data, so it
        # exists even when the sweep grid cannot represent 1/n exactly
        # (e.g. two retrievers at an odd resolution).
        rankings, labels = bm25_is_better
        for resolution in (2, 3, 7, 10):
            result = tune_weights(rankings, labels, top_k=3, resolution=resolution)
            assert result.baseline_score == result.baseline_score  # not NaN
            assert 0.0 <= result.baseline_score <= 1.0

    def test_three_retrievers_get_a_finite_improvement(self) -> None:
        # Regression: with 3 retrievers at the default resolution=10, the grid
        # holds no exact (1/3, 1/3, 1/3) point; improvement() used to be NaN.
        rankings = [{"bm25": ["rel", "x"], "dense": ["x", "rel"], "sparse": ["rel", "x"]}]
        result = tune_weights(rankings, [{"rel"}], top_k=2)
        assert result.improvement() == result.improvement()  # not NaN

    def test_ties_prefer_the_more_balanced_weights(self) -> None:
        # Both retrievers return the same list, so every weighting scores alike.
        rankings = [{"a": ["rel", "x"], "b": ["rel", "x"]}]
        result = tune_weights(rankings, [{"rel"}], top_k=2, resolution=10)
        assert result.weights["a"] == pytest.approx(0.5)

    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_every_named_metric_runs(self, bm25_is_better, metric: str) -> None:
        rankings, labels = bm25_is_better
        result = tune_weights(rankings, labels, top_k=3, metric=metric)
        assert result.metric == metric
        assert 0.0 <= result.score <= 1.0

    def test_custom_metric_callable(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better

        def always_half(retrieved, relevant, k):
            return 0.5

        result = tune_weights(rankings, labels, top_k=3, metric=always_half)
        assert result.score == pytest.approx(0.5)
        assert result.metric == "always_half"

    def test_three_retrievers(self) -> None:
        rankings = [{"a": ["rel"], "b": ["x"], "c": ["y"]}]
        result = tune_weights(rankings, [{"rel"}], top_k=1, resolution=4)
        assert set(result.weights) == {"a", "b", "c"}
        assert result.weights["a"] > 0


class TestEvaluateWeights:
    def test_scores_a_specific_assignment(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        score = evaluate_weights(rankings, labels, {"bm25": 1.0, "dense": 0.0}, top_k=3)
        assert score == pytest.approx(1.0)

    def test_empty_input_scores_zero(self) -> None:
        assert evaluate_weights([], [], {"a": 1.0}) == 0.0

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="label sets"):
            evaluate_weights([{"a": ["x"]}], [{"x"}, {"y"}], {"a": 1.0})


class TestValidation:
    def test_empty_rankings_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="nothing to tune on"):
            tune_weights([], [])

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="label sets"):
            tune_weights([{"a": ["x"]}], [{"x"}, {"y"}])

    def test_inconsistent_retriever_names_are_rejected(self) -> None:
        rankings = [{"a": ["x"], "b": ["y"]}, {"a": ["x"], "c": ["z"]}]
        with pytest.raises(ValueError, match="they must match"):
            tune_weights(rankings, [{"x"}, {"x"}])

    def test_query_without_retrievers_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no retrievers"):
            tune_weights([{}], [{"x"}])

    def test_unknown_metric_name_is_rejected(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        with pytest.raises(ValueError, match="unknown metric"):
            tune_weights(rankings, labels, metric="nonsense")

    def test_resolution_below_one_is_rejected(self, bm25_is_better) -> None:
        rankings, labels = bm25_is_better
        with pytest.raises(ValueError, match="resolution must be at least 1"):
            tune_weights(rankings, labels, resolution=0)

    def test_explosive_grids_are_rejected_up_front(self) -> None:
        # 8 retrievers at resolution 30 is ~10.3M combinations; refuse with a
        # message instead of hanging.
        rankings = [{str(i): ["rel"] for i in range(8)}]
        with pytest.raises(ValueError, match="lower the resolution"):
            tune_weights(rankings, [{"rel"}], resolution=30)


@dataclass(frozen=True)
class Doc:
    """A document as a retriever usually hands it back: an object, not an id."""

    id: str
    text: str


class TestKeySupport:
    """``key=`` reaches tuning too, so object documents need no conversion pass."""

    @pytest.fixture
    def object_rankings(self) -> tuple[list[dict[str, list[Doc]]], list[set[str]]]:
        def docs(*ids: str) -> list[Doc]:
            return [Doc(id=doc_id, text=f"body of {doc_id}") for doc_id in ids]

        rankings = [
            {"bm25": docs("rel", "x", "y"), "dense": docs("x", "y", "rel")},
            {"bm25": docs("rel2", "p", "q"), "dense": docs("p", "q", "rel2")},
            {"bm25": docs("rel3", "m", "n"), "dense": docs("m", "n", "rel3")},
        ]
        labels = [{"rel"}, {"rel2"}, {"rel3"}]
        return rankings, labels

    def test_tunes_on_object_documents(self, object_rankings) -> None:
        rankings, labels = object_rankings
        result = tune_weights(rankings, labels, top_k=3, resolution=10, key=lambda doc: doc.id)
        assert result.weights["bm25"] > result.weights["dense"]

    def test_object_tuning_matches_the_id_equivalent(self, object_rankings, bm25_is_better) -> None:
        # Same data, one pass through objects and one through raw ids: the
        # sweep must not notice the difference.
        object_result = tune_weights(
            object_rankings[0], object_rankings[1], top_k=3, resolution=10, key=lambda doc: doc.id
        )
        id_result = tune_weights(*bm25_is_better, top_k=3, resolution=10)
        assert object_result.weights == id_result.weights
        assert object_result.score == pytest.approx(id_result.score)
        assert object_result.baseline_score == pytest.approx(id_result.baseline_score)

    def test_baseline_also_goes_through_key(self, object_rankings) -> None:
        # improvement() compares two evaluations; if only the sweep saw key=,
        # the baseline would raise rather than score.
        rankings, labels = object_rankings
        result = tune_weights(rankings, labels, top_k=3, key=lambda doc: doc.id)
        assert result.baseline_score > 0.0
        assert result.improvement() >= 0.0

    def test_evaluate_weights_accepts_key(self, object_rankings) -> None:
        rankings, labels = object_rankings
        score = evaluate_weights(
            rankings, labels, {"bm25": 1.0, "dense": 0.0}, top_k=3, key=lambda doc: doc.id
        )
        assert score == pytest.approx(1.0)

    def test_key_maps_dict_documents(self) -> None:
        rankings = [{"bm25": [{"id": "rel"}, {"id": "x"}], "dense": [{"id": "rel"}]}]
        score = evaluate_weights(
            rankings, [{"rel"}], {"bm25": 1.0, "dense": 1.0}, top_k=2, key=lambda doc: doc["id"]
        )
        assert score == pytest.approx(1.0)

    def test_key_can_collapse_chunks_onto_their_parent(self) -> None:
        # Two chunks of the same statute are one document once key= collapses
        # them, so the labels are written in parent ids.
        rankings = [
            {
                "bm25": [Doc("sec-149#0", "a"), Doc("sec-149#1", "b"), Doc("sec-1#0", "c")],
                "dense": [Doc("sec-149#2", "d"), Doc("sec-1#0", "c")],
            }
        ]
        score = evaluate_weights(
            rankings,
            [{"sec-149"}],
            {"bm25": 1.0, "dense": 1.0},
            top_k=1,
            key=lambda doc: doc.id.split("#")[0],
        )
        assert score == pytest.approx(1.0)

    def test_object_documents_without_key_are_rejected(self, object_rankings) -> None:
        rankings, labels = object_rankings
        with pytest.raises(TypeError, match="pass key="):
            tune_weights(rankings, labels, top_k=3, resolution=2)

    def test_evaluate_weights_without_key_is_rejected(self, object_rankings) -> None:
        rankings, labels = object_rankings
        with pytest.raises(TypeError, match="pass key="):
            evaluate_weights(rankings, labels, {"bm25": 1.0, "dense": 1.0}, top_k=3)
