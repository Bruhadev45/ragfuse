# ragfuse

**Weighted Reciprocal Rank Fusion for hybrid retrieval.** Real per-retriever weights, provenance on every hit, zero dependencies.

[![CI](https://github.com/Bruhadev45/ragfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruhadev45/ragfuse/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragfuse.svg)](https://pypi.org/project/ragfuse/)
[![Python](https://img.shields.io/pypi/pyversions/ragfuse.svg)](https://pypi.org/project/ragfuse/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```bash
pip install ragfuse
```

---

## The problem

Reciprocal Rank Fusion combines ranked lists using only the *positions* documents hold in them. That is what makes it work: a BM25 score and a cosine similarity are not comparable, but a rank is a rank.

```
score(d) = Σᵢ  1 / (k + rankᵢ(d))
```

What the textbook formula leaves out is that your retrievers are not equally good. BM25 already accounts for term frequency and document length; a general-purpose embedding model may know nothing about your domain. You want to say *this one counts more*.

The workaround you'll find in most codebases is to pass the same list twice:

```python
fused = rrf(bm25_results, bm25_results, dense_results)  # "weight" BM25 double
```

That expresses integer weights only. It cannot say 0.6 / 0.4, it doesn't survive a third retriever, and the intent disappears into a duplicated argument. This library was extracted from a production RAG system that did exactly this — with `BM25_WEIGHT = 0.4` and `SEMANTIC_WEIGHT = 0.6` sitting in the config, read into the class, and then never used, because the fusion had no way to apply them.

`ragfuse` implements the weighted form:

```
score(d) = Σᵢ  wᵢ / (k + rankᵢ(d))
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

Neither retriever put `sec-149` and `sec-135` in the same order, and fusion resolves it by agreement: the two documents both retrievers returned sit far above the one only BM25 found.

Your documents are usually objects, not strings. Pass a `key`:

```python
hits = rrf(
    {"bm25": bm25_docs, "dense": dense_docs},
    weights={"bm25": 0.6, "dense": 0.4},
    key=lambda doc: doc.id,
)

hits[0].item  # the original object, from the retriever whose weighted contribution was largest
```

## Every hit explains itself

When a hybrid retriever surfaces something strange, you need to know which half is responsible. Each hit carries its full provenance:

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

## Does weighted fusion actually help?

These numbers come from [JurisGPT](https://github.com/Bruhadev45/Juris-GPT), the citation-grounded legal RAG system this library was extracted from — a 120-query human-verified benchmark (Fleiss' κ = 0.81) over 47,867 Indian statutory documents.

| Configuration | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| Lexical only | 0.825 | 0.888 | 0.785 |
| Dense only (MiniLM) | 0.442 | 0.925 | 0.566 |
| Dense only (InLegalBERT) | 0.483 | 0.898 | 0.780 |
| **Hybrid, fused** | **0.842** | **0.953** | 0.685 |

The fused row uses the production 2:1 BM25:lexical weighting (the very duplicate-list hack this library replaces). Read the whole table, not just the bold cells: fusion wins recall and MRR, and its nDCG@5 is *lower* than lexical-only here. Paired statistics over the same runs ([rag-eval-lab](https://github.com/Bruhadev45/rag-eval-lab)) sharpen it further — the MRR gain is significant (+0.065, p=0.0025); the recall gain (+0.017) is within noise at n=120. What fusion reliably buys on this corpus is the first relevant statute placed higher.

It also comes with a warning. Adding a general-purpose MS MARCO cross-encoder reranker on top *lowered* nDCG@5 in every configuration tested:

| Configuration | nDCG@5 | with reranker | Δ |
|---|---|---|---|
| Hybrid | 0.685 | 0.597 | **−8.7** |
| Dense (MiniLM) | 0.566 | 0.403 | **−16.3** |

Retrieval quality on specialist corpora is corpus-bound, not retriever-bound. A reranker trained on web search does not transfer to statutory text, and bolting one on can undo what fusion gained. Measure it on your own data before shipping it.

## API

| Function | Purpose |
|---|---|
| `reciprocal_rank_fusion(rankings, *, k, weights, key, top_k)` | Fuse ranked lists. Aliased as `rrf`. |
| `tune_weights(rankings, relevant, *, metric, top_k, resolution, key)` | Sweep the weight simplex for the best assignment. |
| `evaluate_weights(rankings, relevant, weights, *, metric, top_k, key)` | Score one specific weight assignment. |
| `recall_at_k` · `ndcg_at_k` | Binary-relevance metrics at a cutoff. |
| `precision_at_k(retrieved, relevant, k, *, mode)` | Precision at a cutoff; `mode="trec"` divides by a fixed `k`. |
| `reciprocal_rank` · `mean_reciprocal_rank` | Rank of the first relevant hit. |

`rankings` accepts either `{"name": [...]}` or a plain sequence of lists (named `"0"`, `"1"`, …). `k` defaults to 60, the standard smoothing constant; larger values flatten the discount between adjacent ranks.

### Two precisions at k

Precision@k has two conventions, and they disagree whenever a retriever returns
fewer than `k` documents — a re-ranked shortlist, a filtered search that ran dry.
`mode` makes the choice explicit instead of silent:

```python
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

### Migrating off the duplicate-list trick

Weight `2.0` reproduces passing a list twice, exactly — there's a test that pins this, so you can switch without moving your numbers:

```python
rrf(bm25, bm25, dense)  # before
rrf({"bm25": bm25, "dense": dense}, weights={"bm25": 2})  # after, identical output
```

## Design

- **No dependencies.** Pure standard library, so it will not fight your torch or numpy pin.
- **Typed.** Ships `py.typed`; `mypy --strict` clean.
- **Immutable results.** Frozen dataclasses, no accidental mutation of a shared ranking.
- **Deterministic.** Ties break on document id rather than dict ordering.
- **Explicit failures.** Negative weights, unknown retriever names, mismatched label counts, and objects with no `key` all raise on the spot with a message that says what to do.

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
