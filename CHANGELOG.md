# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-31

First release. Extracted from the retrieval layer of
[JurisGPT](https://github.com/Bruhadev45/Juris-GPT) and generalised.

### Added
- `reciprocal_rank_fusion` (aliased `rrf`) with real per-retriever float
  weights, replacing the common "pass the list twice" integer workaround.
- `FusedHit` / `Contribution` result types carrying full provenance: which
  retriever returned a document, at what rank, and what it contributed.
- `tune_weights` and `evaluate_weights` for choosing weights from labelled
  queries by sweeping the weight simplex.
- Binary-relevance metrics: `recall_at_k`, `precision_at_k`, `ndcg_at_k`,
  `reciprocal_rank`, `mean_reciprocal_rank`.
- Deterministic tie-breaking on document id, so repeated runs agree.
- Type hints throughout, with `py.typed` and a `mypy --strict` clean build.
