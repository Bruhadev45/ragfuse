# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-31

Fixes from a same-day adversarial review of 0.1.0.

### Fixed
- Documents following a skipped duplicate now keep their rank among *unique*
  documents, so collapsing chunks to a parent document via ``key=`` no longer
  deflates every later document's score.
- ``tune_weights`` evaluates the equal-weight baseline directly on the data;
  ``improvement()`` is no longer NaN whenever the sweep grid cannot represent
  1/n exactly (e.g. three retrievers at the default resolution).
- ``FusedHit`` excludes ``item`` from its hash, so hits carrying dict (or any
  unhashable) documents work in sets and as dict keys.
- NaN weights are rejected with a clear error instead of silently corrupting
  every score.
- The weight sweep generates only valid simplex points (integer compositions)
  instead of filtering the full Cartesian product, and refuses grids over
  200,000 combinations with an actionable message.

### Changed
- README: the JurisGPT benchmark table now names the weighting used and
  states that fusion's nDCG@5 is lower than lexical-only on that corpus,
  with a pointer to the paired statistics in rag-eval-lab.

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
