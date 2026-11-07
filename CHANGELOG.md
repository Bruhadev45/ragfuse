# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `combsum` and `combmnz`: score-based fusion beside RRF. Each takes ranked
  `(document, score)` lists, normalises every list with
  `normalize="minmax"` (default), `"zscore"` or `None`, and returns the same
  `FusedHit`s under the same `weights=`/`key=` contract, so the fusions are
  interchangeable. NaN and infinite scores are rejected; zero-weight
  retrievers do not count towards CombMNZ's multiplier. `NORMALIZATIONS` lists
  the accepted values and `Normalization` is their type. README: a "Which
  fusion?" comparison.
- `fuse` and `rerank`: fusion and reranking with a checked audit trail. `fuse`
  runs `rrf`, `combsum` or `combmnz` on named `(document, score)` lists and
  returns a `FusionTrace`. Each `Candidate` lists every retriever as a
  `SourceTrace`, with rank, raw score, the value fusion used, weight and
  contribution, including the retrievers that missed it. Ids and names must
  be plain non-empty strings, scores finite and lists best-first, and the
  result does not depend on the mapping's order. `eligible=` makes any other id
  fail the call. `rerank` reorders the candidates by your scores and raises
  `IneligibleCandidateError` for a score on anything else, so reranking can
  reorder or cut candidates but never add one. Error messages leave out the
  offending ids. README: "Trace fusion and reranking".

### Changed
- `FusedHit.normalized_score` is min-max scaled across the hits when a score
  fusion leaves some totals negative (z-scores, raw logits). RRF scores are
  never negative, so RRF output is unchanged.
- README: install, use-when, limitations, input and output, and the API table
  now come before the analysis. It says that v0.2.0 lacks `combsum` and
  `combmnz`, and it reports that the hybrid MRR gain did not replicate on
  rag-eval-lab's wider 204-query set. The migration and precision examples now
  run as printed. `tests/test_readme.py` runs the examples and checks the
  install tag, the API table, the *(main only)* marks and the in-page links.

### Fixed
- Infinite weights are rejected. They used to give infinite scores and a NaN
  `normalized_score`.
- Fused totals that overflow, or that lie too far apart to normalise, raise a
  `ValueError` instead of coming back as `inf` or `NaN`.

## [0.2.0] - 2026-10-12

### Added
- `tune_weights` and `evaluate_weights` accept `key=`, matching
  `reciprocal_rank_fusion`. Weight sweeps can now run over the document
  objects a retriever returns instead of requiring a projection to ids first;
  the equal-weight baseline goes through the same `key`, so `improvement()`
  stays meaningful.
- `precision_at_k(..., mode="trec")` divides by a fixed `k` instead of
  `min(k, len(retrieved))`, so numbers can be made to line up with `trec_eval`
  and the leaderboards computed with it. The default (`mode="retrieved"`) is
  unchanged; `PRECISION_MODES` lists both, and the README explains when the two
  conventions diverge.

### Changed
- The version now lives only in `ragfuse.__version__`; `pyproject.toml` reads
  it from there, so the two can no longer drift.
- Releases are built by a `release.yml` workflow on `v*` tags: tests, sdist and
  wheel, `twine check --strict`, a smoke install of the wheel that must report
  the tagged version, then a GitHub Release with both files attached and this
  changelog section as its notes.
- README: install from the release tag. The `ragfuse` name on PyPI belongs to
  an unrelated package, so `pip install ragfuse` does not install this library.
  The API table now covers every public name, and a "Related" section links
  atlas-ui's `fromRagfuse` adapter and rag-eval-lab.

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
