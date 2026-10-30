# ragfuse

**Weighted Reciprocal Rank Fusion for hybrid retrieval.** Real per-retriever weights, provenance on every hit, zero dependencies.

[![CI](https://github.com/Bruhadev45/ragfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruhadev45/ragfuse/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Bruhadev45/ragfuse)](https://github.com/Bruhadev45/ragfuse/releases)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ragfuse merges the ranked results several retrievers return for one query (say BM25 and a vector search) into a single ranking. You decide how much each retriever counts. Every merged result records which retrievers returned it, at what rank, and how much each one contributed. It is pure Python with no dependencies, and fully typed.

## Install

```bash
pip install "ragfuse @ git+https://github.com/Bruhadev45/ragfuse@v0.2.0"
```

This needs Python 3.10 or newer and installs nothing else. On an older Python, pip stops with "requires a different Python". The macOS `/usr/bin/python3` checked for this page was 3.9.6, so use a newer interpreter there. The [v0.2.0 release](https://github.com/Bruhadev45/ragfuse/releases/tag/v0.2.0) also carries the wheel and sdist.

Not on PyPI. The `ragfuse` project on PyPI is an unrelated document-loading package (checked 2026-10-30), so installing `ragfuse` by name from PyPI gets something else.

**v0.2.0 does not include `combsum` or `combmnz`.** They are on `main` and not yet released. To use them, install the commit that added them; it is pinned so the install stays reproducible. It still reports version 0.2.0, as `main` will until the next release.

```bash
pip install "ragfuse @ git+https://github.com/Bruhadev45/ragfuse@79b5dff326ea50b291c4eb2fb41b5e545ba476a8"
```

## Quickstart

```python
from ragfuse import rrf

bm25 = ["sec-149", "sec-135", "sec-188"]
dense = ["sec-135", "sec-149", "sec-203"]

hits = rrf(
    {"bm25": bm25, "dense": dense},
    weights={"bm25": 0.6, "dense": 0.4},
    top_k=3,
)

for hit in hits:
    print(f"{hit.key:10} {hit.normalized_score:.3f}  from {hit.sources}")
```

```
sec-149    1.000  from ('bm25', 'dense')
sec-135    0.997  from ('bm25', 'dense')
sec-188    0.585  from ('bm25',)
```

The two retrievers disagree about the order of `sec-149` and `sec-135`. Fusion settles it by agreement: the two documents both retrievers returned rank well above the one only BM25 found.

Your documents are usually objects, not strings. Pass a `key`:

```python
hits = rrf(
    {"bm25": bm25_docs, "dense": dense_docs},
    weights={"bm25": 0.6, "dense": 0.4},
    key=lambda doc: doc.id,
)

hits[0].item  # the original object, from the retriever whose weighted contribution was largest
```

## Use it when

- You already have two or more ranked lists for the same query and want one list. The lists can come from BM25 and a vector index, two embedding models, or several rewrites of the query.
- You want to say how much each retriever counts, as a float, and keep that weighting visible in the code.
- You need to see, for each result, which retriever found it and at what rank. That helps when debugging, building a UI, or writing an evaluation report.
- You have labelled queries and want to choose weights from them, with the metrics to check that choice.

## Limitations

- It fuses lists. It does not retrieve, embed, index or rerank; you bring the ranked results.
- It works in memory, one query at a time, in pure Python. Its speed has not been benchmarked.
- The metrics use binary relevance: a document is either relevant or not. There is no graded nDCG.
- `tune_weights` sweeps a grid of weights and refuses grids over 200,000 combinations. With few labelled queries, the best weights may only fit noise, so check them on queries you did not tune on.
- Fusion does not always help. [Below](#does-weighted-fusion-actually-help), it raised MRR on 120 human-verified legal queries, but the gain did not replicate on a wider query set. Measure on your own data.
- It is not on PyPI, and v0.2.0 lacks `combsum` and `combmnz` (see [Install](#install)).

## Input and output

**In.** Each retriever's results for one query, best first:

| Function | One retriever's list | Example |
|---|---|---|
| `rrf` | documents | `["sec-149", "sec-135"]` |
| `combsum`, `combmnz` | `(document, score)` pairs | `[("sec-149", 12.4), ("sec-135", 9.1)]` |

Pass the lists as `{"name": list}`, and the names show up in the provenance. You can also pass a plain sequence of lists, which are then named `"0"`, `"1"`, …. Documents must be strings, or any objects plus a `key=` that returns a string id. A document repeated in one list counts once, at its first position. `weights=` maps names to non-negative floats, or is a sequence aligned with the lists. A name you leave out gets 1.0, and the weights need not sum to 1.

**Out.** A list of frozen `FusedHit`s, best first, with ties broken on the id:

| Attribute | Meaning |
|---|---|
| `key` | The document id. |
| `item` | The original object, from the retriever that contributed most. |
| `score` | The fused score. |
| `normalized_score` | `score` relative to the top hit, from 0 to 1: divided by the top score, or min-max scaled across the hits if some totals are negative. |
| `sources` | The names of the retrievers that returned it. |
| `contributions` | One `Contribution(source, rank, weight, score)` per retriever. `rank` counts from 0. |
| `rank_in(name)` | Its rank in that retriever's list, or `None`. |

**Errors.** Bad input raises at once, with a message that says what to fix. For example, you get a `ValueError` for no lists at all, negative, NaN or all-zero weights, a weight for a retriever that does not exist, an `rrf` `k` that is not positive, and NaN or infinite scores. You get a `TypeError` for documents that are not strings when no `key=` is given, and for list entries that are not `(document, score)` pairs with a numeric score.

## API

Everything imports from `ragfuse`. Names marked *(main only)* are not in v0.2.0.

| Name | What it does | Details |
|---|---|---|
| `reciprocal_rank_fusion(rankings, *, k, weights, key, top_k)`, alias `rrf` | Fuses ranked lists by rank. | [Quickstart](#quickstart), [Why weighted RRF](#why-weighted-rrf) |
| `combsum(rankings, *, normalize, weights, key, top_k)` · `combmnz(...)` *(main only)* | Fuses `(document, score)` lists by normalised score. | [Which fusion?](#which-fusion) |
| `FusedHit` · `Contribution` | Frozen result types: a fused document, and one retriever's share of its score. | [Every hit explains itself](#every-hit-explains-itself) |
| `tune_weights(rankings, relevant, *, k, top_k, metric, resolution, key)` | Sweeps the weight simplex for the best assignment. | [Choosing weights from evidence](#choosing-weights-from-evidence) |
| `evaluate_weights(rankings, relevant, weights, *, k, top_k, metric, key)` | Scores one specific weight assignment. | [Choosing weights from evidence](#choosing-weights-from-evidence) |
| `WeightSearchResult` | What `tune_weights` returns: `weights`, `score`, `table`, `baseline_score`, `improvement()`. | [Choosing weights from evidence](#choosing-weights-from-evidence) |
| `recall_at_k` · `ndcg_at_k` | Binary-relevance metrics at a cutoff. | [Choosing weights from evidence](#choosing-weights-from-evidence) |
| `precision_at_k(retrieved, relevant, k, *, mode)` | Precision at a cutoff; `mode="trec"` divides by a fixed `k`. | [Two precisions at k](#two-precisions-at-k) |
| `reciprocal_rank` · `mean_reciprocal_rank` | The rank of the first relevant hit. | [Choosing weights from evidence](#choosing-weights-from-evidence) |
| `DEFAULT_K` · `METRICS` · `PRECISION_MODES` | The default `k` (60), the metric names tuning accepts, and the two precision modes. | |
| `NORMALIZATIONS` · `Normalization` *(main only)* | The accepted `normalize=` values, and their type. | [Which fusion?](#which-fusion) |

`k` defaults to 60, the standard smoothing constant; larger values shrink the gap between adjacent ranks.

The rest of this page covers why the weights matter, how to choose between the fusions and set weights, and what fusion did on one real benchmark.

## Why weighted RRF

Reciprocal Rank Fusion combines ranked lists using only the *positions* documents hold in them. That is what makes it work: a BM25 score and a cosine similarity are not comparable, but a rank is a rank.

```
score(d) = Σᵢ  1 / (k + rankᵢ(d))
```

What the textbook formula leaves out is that your retrievers are not equally good. BM25 already accounts for term frequency and document length; a general-purpose embedding model may know nothing about your domain. You want to say *this one counts more*.

The workaround you'll find in most codebases is to pass the same list twice:

```python
fused = rrf([bm25_results, bm25_results, dense_results])  # "weight" BM25 double
```

That expresses integer weights only. It cannot say 0.6 / 0.4, it doesn't survive a third retriever, and the intent disappears into a duplicated argument. This library was extracted from a production RAG system that did exactly this — with `BM25_WEIGHT = 0.4` and `SEMANTIC_WEIGHT = 0.6` sitting in the config, read into the class, and then never used, because the fusion had no way to apply them.

`ragfuse` implements the weighted form:

```
score(d) = Σᵢ  wᵢ / (k + rankᵢ(d))
```

### Migrating off the duplicate-list trick

Weight `2.0` reproduces passing a list twice: the same order, and the same scores up to floating-point rounding in the last digit. A test pins it, so you can switch without moving your numbers. Only the provenance changes. The trick shows BM25 as two sources, and the weight shows it as one.

```python
from ragfuse import rrf

bm25 = ["sec-149", "sec-135", "sec-188"]
dense = ["sec-135", "sec-149", "sec-203"]

rrf([bm25, bm25, dense])  # before
rrf({"bm25": bm25, "dense": dense}, weights={"bm25": 2})  # after: same order and scores
```

## Which fusion?

RRF fuses on rank alone. When your retrievers' scores mean something, CombSUM and CombMNZ fuse on the scores instead, after normalising each list onto a common scale. All three take the same `weights=` and `key=`, and return the same `FusedHit`s, so you can swap one for another and compare them on your own labels.

```python
from ragfuse import combmnz, combsum

scored = combsum(
    {
        "bm25": [("sec-149", 12.4), ("sec-135", 9.1)],
        "dense": [("sec-135", 0.82), ("sec-149", 0.80)],
    },
    weights={"bm25": 0.6, "dense": 0.4},
    normalize="minmax",  # or "zscore", or None for scores already on one scale
)
```

| | `rrf` | `combsum` | `combmnz` |
|---|---|---|---|
| Input per retriever | ranked documents | ranked `(document, score)` pairs | ranked `(document, score)` pairs |
| Uses | rank only | normalised scores, summed | normalised scores, summed, times the number of retrievers that found the document |
| Rewards agreement | implicitly | no (only via the sum) | explicitly |
| Near-ties in a retriever | treated as a full rank apart | stay near-ties | stay near-ties |
| Sensitive to score outliers | no | yes, through `minmax`/`zscore` | yes, and amplified for shared documents |
| Reach for it when | scores are not comparable or not available | scores are calibrated and one retriever's confidence should count | scores are calibrated and agreement should count double |

Normalisation is per list: `"minmax"` maps each list onto `0..1` (a constant list, or a single hit, maps to `1.0`); `"zscore"` centres and scales it (a constant list maps to `0.0`, and totals can go negative). NaN and infinite scores are rejected. Fed RRF's own `1 / (k + rank + 1)` scores with `normalize=None`, `combsum` returns exactly what `rrf` does; a test pins it. Which one wins on your data is an empirical question, so measure it with `ndcg_at_k` and friends before switching.

## Every hit explains itself

When a hybrid retriever surfaces something strange, you need to know which half is responsible. Each hit carries its full provenance. Here `hits` is the Quickstart's result:

```python
hit = hits[0]

hit.score  # 0.0163 — the raw weighted RRF sum
hit.normalized_score  # 1.0    — relative to the top hit
hit.sources  # ('bm25', 'dense')
hit.rank_in("dense")  # 1

for c in hit.contributions:
    print(f"  {c.source:6} rank {c.rank}  weight {c.weight}  →  {c.score:.5f}")
```

```
  bm25   rank 0  weight 0.6  →  0.00984
  dense  rank 1  weight 0.4  →  0.00645
```

Results are immutable, and ties break on document id, so the same inputs always produce the same ordering — which matters more than it sounds when you are trying to reproduce a benchmark run.

## Choosing weights from evidence

Guessing at weights is how a hybrid retriever ends up worse than either of its halves. If you have labelled queries, sweep them:

```python
from ragfuse import tune_weights

result = tune_weights(
    rankings_per_query,  # [{"bm25": [...], "dense": [...]}, ...]
    relevant_per_query,  # [{"sec-149"}, ...]
    metric="ndcg",
    top_k=5,
)

result.weights  # the best assignment found, e.g. {'bm25': 0.7, 'dense': 0.3}
result.score  # its mean nDCG@5
result.improvement()  # how many metric points that gained over equal weights
```

Tuning takes the same `key=` as fusion, so you can sweep over the objects your retrievers already return instead of projecting them to ids first. The labels stay in whatever ids `key` produces:

```python
result = tune_weights(
    rankings_per_query,  # [{"bm25": [Doc(...), ...], "dense": [...]}, ...]
    relevant_per_query,  # [{"sec-149"}, ...]
    key=lambda doc: doc.id,
)
```

`result.table` holds every combination tried, best first — worth plotting before you trust a single number, since a weight that wins by 0.001 on 40 queries has not really won.

Metrics are available on their own too:

```python
from ragfuse import recall_at_k, ndcg_at_k, mean_reciprocal_rank
```

### Two precisions at k

Precision@k has two conventions, and they disagree whenever a retriever returns
fewer than `k` documents — a re-ranked shortlist, a filtered search that ran dry.
`mode` makes the choice explicit instead of silent:

```python
from ragfuse import precision_at_k

retrieved = ["sec-149", "sec-135", "sec-188"]  # three results for k=5
relevant = {"sec-149", "sec-135"}

precision_at_k(retrieved, relevant, k=5)  # 0.667 — divides by min(k, len(retrieved))
precision_at_k(retrieved, relevant, k=5, mode="trec")  # 0.4 — divides by k
```

The default, `mode="retrieved"`, does not penalise a system for slots it never
claimed to fill, which is the fairer comparison between retrievers whose list
lengths differ. `mode="trec"` is what `trec_eval` reports, and what the numbers
on a benchmark leaderboard were computed with: the two unfilled slots count as
misses. The modes agree exactly once at least `k` distinct documents come back,
so this only bites at the margin — but at the margin it moved the example above
by 0.27, which is more than most reported improvements.

## Does weighted fusion actually help?

These numbers come from [JurisGPT](https://github.com/Bruhadev45/Juris-GPT), the citation-grounded legal RAG system this library was extracted from — a 120-query human-verified benchmark (Fleiss' κ = 0.81) over 47,867 Indian statutory documents.

| Configuration | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| Lexical only | 0.825 | 0.888 | 0.785 |
| Dense only (MiniLM) | 0.442 | 0.925 | 0.566 |
| Dense only (InLegalBERT) | 0.483 | 0.898 | 0.780 |
| **Hybrid, fused** | **0.842** | **0.953** | 0.685 |

The fused row uses the production 2:1 BM25:lexical weighting (the very duplicate-list hack this library replaces). Read the whole table, not just the bold cells: fusion wins recall and MRR, and its nDCG@5 is *lower* than lexical-only here. Paired statistics over the same runs ([rag-eval-lab](https://github.com/Bruhadev45/rag-eval-lab)) sharpen it further — the MRR gain is significant (+0.065, p=0.0025); the recall gain (+0.017) is within noise at n=120. What fusion buys on these 120 queries is the first relevant statute placed higher.

That gain does not hold beyond them. On rag-eval-lab's wider 204-query set (the same 120 plus 84 auto-generated queries over other document types), the MRR difference is −0.018 (p = 0.38), and on the 84 held-out queries alone fusion is worse (−0.138, p = 0.0002). The 84 are generated from the corpus rather than labelled by people, so they are weaker evidence. Still, the MRR gain is shown only on the human-verified benchmark, not beyond it.

It also comes with a warning. Adding a general-purpose MS MARCO cross-encoder reranker on top *lowered* nDCG@5 in every configuration tested:

| Configuration | nDCG@5 | with reranker | Δ |
|---|---|---|---|
| Hybrid | 0.685 | 0.597 | **−8.7** |
| Dense (MiniLM) | 0.566 | 0.403 | **−16.3** |

Retrieval quality on specialist corpora is corpus-bound, not retriever-bound. A reranker trained on web search does not transfer to statutory text, and bolting one on can undo what fusion gained. Measure it on your own data before shipping it.

## Design

- **No dependencies.** Pure standard library, so it will not fight your torch or numpy pin.
- **Typed.** Ships `py.typed`; `mypy --strict` clean.
- **Immutable results.** Frozen dataclasses, no accidental mutation of a shared ranking.
- **Deterministic.** Ties break on document id rather than dict ordering.
- **Explicit failures.** Negative weights, unknown retriever names, mismatched label counts, and objects with no `key` all raise on the spot with a message that says what to do.

## Related

- [atlas-ui](https://github.com/Bruhadev45/atlas-ui#retrievaltrace): React components for AI interfaces. Its `fromRagfuse` adapter maps serialised `FusedHit`s (snake_case or camelCase) onto `RetrievalTrace`, which renders each hit with its per-retriever rank and weight.
- [rag-eval-lab](https://github.com/Bruhadev45/rag-eval-lab): the reproducible benchmark behind the numbers above, with paired significance tests over the same 120 queries and the wider 204-query replication.

## Development

```bash
git clone https://github.com/Bruhadev45/ragfuse
cd ragfuse
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest          # tests, doctests, coverage gate at 90%
mypy            # strict
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
