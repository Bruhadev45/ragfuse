"""Regenerate ``bm25.trec`` and ``dense_hits.json`` by running minirag.

The two input files of ``examples/two_shapes.py`` are real retriever output,
not typed in. This script indexes minirag's hand-written solar-system fixture
corpus and writes, for two queries:

- minirag's BM25 results as a TREC run file, the six-column text format that
  trec_eval reads and tools such as Anserini and Pyserini write;
- minirag's dense results (hashed TF-IDF, which matches words; it is not a
  learned semantic embedding) as the ``Hit`` records ``MiniRAG.search``
  returns, serialised with ``dataclasses.asdict``.

It needs minirag, which ragfuse does not depend on, so it is not run by the
tests. The committed files came from minirag commit 462fa00::

    git clone https://github.com/Bruhadev45/minirag && cd minirag
    git checkout 462fa009ee9bdf1b34f56d6e798373cdbd961aa1
    python3 -m venv .venv && .venv/bin/pip install -e .
    .venv/bin/python <ragfuse>/examples/data/make_inputs.py tests/fixtures/corpus.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from minirag import Document, MiniRAG

OUT = Path(__file__).resolve().parent
QUERIES = {
    "q1": "dwarf planet beyond neptune",
    "q2": "which moon has an ocean under the ice",
}
DEPTH = 10  # results kept per query and retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, help="minirag's tests/fixtures/corpus.json")
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    engine = MiniRAG(max_tokens=60, overlap_tokens=20)  # as minirag's extras do
    engine.index([Document(doc_id=d["doc_id"], text=d["text"]) for d in corpus["documents"]])

    trec_lines = []
    dense = []
    for qid, query in QUERIES.items():
        bm25 = engine.search(query, top_k=DEPTH, mode="bm25")
        trec_lines += [
            f"{qid} Q0 {hit.doc_id} {rank} {hit.score:.6f} minirag-bm25"
            for rank, hit in enumerate(bm25, start=1)
        ]
        hits = engine.search(query, top_k=DEPTH, mode="dense")
        dense.append({"query_id": qid, "query": query, "hits": [asdict(h) for h in hits]})

    (OUT / "bm25.trec").write_text("\n".join(trec_lines) + "\n", encoding="utf-8")
    payload = {"retriever": "minirag dense (hashed TF-IDF)", "results": dense}
    (OUT / "dense_hits.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
