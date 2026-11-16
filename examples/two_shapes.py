"""Fuse two retrievers that report results in different shapes, and show why.

``python examples/two_shapes.py`` reads two files of real retriever output for
the same two queries and fuses them with :func:`ragfuse.fuse`:

- ``data/bm25.trec``: a TREC run file, the six-column text format
  (``qid Q0 docno rank score tag``) that trec_eval reads and tools such as
  Anserini and Pyserini write.
- ``data/dense_hits.json``: the ``Hit`` records minirag's ``MiniRAG.search``
  returns, as JSON.

Both came from running minirag on its fixture corpus (see
``data/make_inputs.py``). Its "dense" retriever is hashed TF-IDF, which matches
words, so this is two word matchers disagreeing, not lexical against semantic.
The point is the plumbing: each reader turns its shape into best-first
``(id, score)`` lists and fails loudly on malformed input, and the trace shows
each retriever's rank, raw score and contribution for every fused document.
It shows how fusion settles a disagreement; it measures nothing.

Both readers need only the standard library and cap the bytes they read.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from ragfuse import FusionTrace, fuse

DATA = Path(__file__).resolve().parent / "data"
MAX_BYTES = 1_000_000  # an example's inputs; size a real service's cap to its retrievers
MAX_QID = 64
_RANK = re.compile(r"[0-9]+")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")

Ranked = dict[str, list[tuple[str, float]]]  # query id -> best-first (id, score)


def _read_capped(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError(f"{path.name}: larger than {MAX_BYTES} bytes")
    return raw.decode("utf-8-sig")  # a leading byte-order mark is not part of the first qid


def _check_qid(qid: str, where: str) -> str:
    """Query ids are printed, so they must be short and printable."""
    if not qid or len(qid) > MAX_QID or not qid.isprintable() or qid != qid.strip():
        raise ValueError(f"{where}: query id must be 1-{MAX_QID} printable characters")
    return qid


def _finite(value: float, where: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{where}: score must be finite")
    return value


def _parse_score(text: str, where: str) -> float:
    """An ASCII decimal number; error messages name the line, never echo the field."""
    if not _NUMBER.fullmatch(text):
        raise ValueError(f"{where}: score is not a decimal number")
    return _finite(float(text), where)


def read_trec_run(path: Path) -> Ranked:
    """Parse a TREC run file into best-first ``(docno, score)`` lists per query.

    Each non-blank line is ``qid Q0 docno rank score tag``. Lists are ordered
    by the rank column, which must be a positive integer unique within its
    query. A docno may appear once per query, and one file holds one run tag.
    """
    rows: dict[str, list[tuple[int, str, float]]] = {}
    tags: set[str] = set()
    for number, line in enumerate(_read_capped(path).splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{where}: expected 6 fields, got {len(fields)}")
        qid, _, docno, rank_text, score_text, tag = fields
        _check_qid(qid, where)
        if not _RANK.fullmatch(rank_text) or int(rank_text) < 1:
            raise ValueError(f"{where}: rank is not a positive integer")
        tags.add(tag)
        rows.setdefault(qid, []).append((int(rank_text), docno, _parse_score(score_text, where)))
    if len(tags) > 1:
        raise ValueError(f"{path.name}: one run per file, found {len(tags)} run tags")
    ranked: Ranked = {}
    for qid, entries in rows.items():
        entries.sort()
        ranks = [rank for rank, _, _ in entries]
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"{path.name}: query {qid} repeats a rank")
        if len({docno for _, docno, _ in entries}) != len(entries):
            raise ValueError(f"{path.name}: query {qid} lists a docno twice")
        ranked[qid] = [(docno, score) for _, docno, score in entries]
    return ranked


def read_minirag_hits(path: Path) -> tuple[Ranked, dict[str, str]]:
    """Parse serialised minirag hits into ``(doc_id, score)`` lists and query texts.

    The file holds ``{"results": [{"query_id", "query", "hits": [Hit, ...]}]}``
    where each hit is ``dataclasses.asdict`` of a ``minirag.Hit``. Hits keep
    the order minirag returned them in; :func:`ragfuse.fuse` checks it is
    best-first.
    """
    try:
        data = json.loads(_read_capped(path))
    except RecursionError:
        raise ValueError(f"{path.name}: JSON nested too deeply") from None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ValueError(f"{path.name}: expected an object with a 'results' list")
    ranked: Ranked = {}
    queries: dict[str, str] = {}
    for position, result in enumerate(results):
        where = f"{path.name} results[{position}]"
        if not isinstance(result, dict) or not isinstance(result.get("hits"), list):
            raise ValueError(f"{where}: expected an object with a 'hits' list")
        qid, query = result.get("query_id"), result.get("query")
        if not isinstance(qid, str) or not isinstance(query, str) or qid in ranked:
            raise ValueError(f"{where}: needs a unique string query_id and a string query")
        _check_qid(qid, where)
        hits = [_minirag_hit(hit, f"{where}.hits[{i}]") for i, hit in enumerate(result["hits"])]
        if len({doc_id for doc_id, _ in hits}) != len(hits):
            raise ValueError(f"{where}: lists a doc_id twice")
        ranked[qid] = hits
        queries[qid] = query
    return ranked, queries


def _minirag_hit(hit: Any, where: str) -> tuple[str, float]:
    if not isinstance(hit, dict) or not isinstance(hit.get("doc_id"), str):
        raise ValueError(f"{where}: expected a hit with a string doc_id")
    score = hit.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise ValueError(f"{where}: score must be a number")
    try:
        value = float(score)
    except OverflowError:  # an integer too large for a float
        raise ValueError(f"{where}: score must be finite") from None
    return hit["doc_id"], _finite(value, where)


DEPTH = 10  # each file keeps a retriever's top 10, so "-" means "not in its top 10"


def fuse_query(bm25: list[tuple[str, float]], dense: list[tuple[str, float]]) -> FusionTrace:
    """Equal-weight RRF over the two lists, keeping every retriever's part."""
    return fuse({"bm25": bm25, "dense": dense}, method="rrf", top_k=DEPTH)


def _cell(rank: int | None, score: float | None) -> str:
    return "-" if rank is None or score is None else f"{rank + 1:>2}  {score:.4f}"


def report(trace: FusionTrace) -> list[str]:
    """A table of the fused candidates, then how the top one's score adds up."""
    if not trace.candidates:
        return ["  (neither retriever returned anything)"]
    lines = [f"  {'fused':>5}  {'document':<14} {'score':>8}   {'bm25':<10}  dense"]
    for candidate in trace.candidates:
        cells = {t.source: _cell(t.rank, t.raw_score) for t in candidate.retrievers}
        row = (
            f"  {candidate.rank + 1:>5}  {candidate.key:<14} {candidate.fused_score:.6f}"
            f"   {cells['bm25']:<10}  {cells['dense']}"
        )
        lines.append(row.rstrip())
    top = trace.candidates[0]
    adds = " + ".join(f"{t.source} {t.contribution:.6f}" for t in top.retrievers)
    lines.append(f"  {top.key}: {adds} = {top.fused_score:.6f}")
    return lines


def main() -> None:
    bm25 = read_trec_run(DATA / "bm25.trec")
    dense, queries = read_minirag_hits(DATA / "dense_hits.json")
    if bm25.keys() != dense.keys():
        sys.exit("the two files cover different queries")
    print(f"rank and raw score per retriever; - = not in that retriever's top {DEPTH}")
    for qid, query in queries.items():
        print(f"\n{qid}: {query!r}")
        print("\n".join(report(fuse_query(bm25[qid], dense[qid]))))


if __name__ == "__main__":
    main()
