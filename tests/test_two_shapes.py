"""examples/two_shapes.py: its two readers, its trace, and the README's copy of its output."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from test_readme import blocks, section

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "two_shapes.py"
TREC = ROOT / "examples" / "data" / "bm25.trec"
HITS = ROOT / "examples" / "data" / "dense_hits.json"


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("two_shapes", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = load_example()


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_output_matches_the_readme() -> None:
    run = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=True,
    )
    shown = [body for lang, body in blocks(section("## Fuse two tools' results")) if not lang]
    assert shown, "the README section needs an output block"
    actual = run.stdout.splitlines()
    expected = shown[0].splitlines()
    assert actual[: len(expected)] == expected
    # The README shows q1 and says q2 "goes the same way": the rest is q2's table.
    rest = actual[len(expected) :]
    assert rest[:2] == ["", "q2: 'which moon has an ocean under the ice'"]
    assert len(rest) == len(expected) - 1  # no header line, otherwise the same shape
    assert rest[-1] == "  moon: bm25 0.016393 + dense 0.016129 = 0.032522"


def test_committed_files_hold_the_same_queries_ten_deep() -> None:
    bm25 = ex.read_trec_run(TREC)
    dense, queries = ex.read_minirag_hits(HITS)
    assert bm25.keys() == dense.keys() == queries.keys() == {"q1", "q2"}
    assert all(len(ranked) == ex.DEPTH for ranked in (*bm25.values(), *dense.values()))
    assert bm25["q1"][0] == ("pluto", 5.859364)
    assert dense["q1"][0][0] == "asteroid_belt"


def test_trec_lists_follow_the_rank_column_not_line_order(tmp_path: Path) -> None:
    lines = TREC.read_text(encoding="utf-8").splitlines()
    shuffled = write(tmp_path, "run.trec", "\n\n".join(reversed(lines)) + "\n")
    assert ex.read_trec_run(shuffled) == ex.read_trec_run(TREC)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("q1 Q0 a 1 2.0\n", "line 1: expected 6 fields, got 5"),
        ("q1 Q0 a 0 2.0 t\n", "line 1: rank is not a positive integer$"),
        ("q1 Q0 a one 2.0 t\n", "rank is not a positive integer$"),
        ("q1 Q0 a \u00b2 2.0 t\n", "rank is not a positive integer$"),  # superscript two
        ("q1 Q0 a \u0661 2.0 t\n", "rank is not a positive integer$"),  # Arabic-Indic one
        ("q1 Q0 a 1 high t\n", "line 1: score is not a decimal number$"),
        ("q1 Q0 a 1 nan t\n", "score is not a decimal number$"),
        ("q1 Q0 a 1 inf t\n", "score is not a decimal number$"),
        ("q1 Q0 a 1 1_0 t\n", "score is not a decimal number$"),
        ("q1 Q0 a 1 1e999 t\n", "score must be finite$"),
        ("q\x1b[2J Q0 a 1 2.0 t\n", "query id must be 1-64 printable characters$"),
        (f"{'q' * 65} Q0 a 1 2.0 t\n", "query id must be 1-64 printable characters$"),
        ("q1 Q0 a 1 2.0 t\nq1 Q0 b 1 1.0 t\n", "query q1 repeats a rank"),
        ("q1 Q0 a 1 2.0 t\nq1 Q0 a 2 1.0 t\n", "query q1 lists a docno twice"),
        ("q1 Q0 a 1 2.0 t\nq2 Q0 a 1 1.0 u\n", "one run per file, found 2 run tags"),
    ],
)
def test_trec_reader_rejects_malformed_runs(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ex.read_trec_run(write(tmp_path, "run.trec", text))


def test_trec_reader_drops_a_byte_order_mark_and_reads_crlf(tmp_path: Path) -> None:
    path = tmp_path / "run.trec"
    path.write_bytes(b"\xef\xbb\xbfq1 Q0 a 1 2.0 t\r\nq1 Q0 b 2 -1.5e-1 t\r\n")
    assert ex.read_trec_run(path) == {"q1": [("a", 2.0), ("b", -0.15)]}


def test_readers_cap_what_they_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="larger than 10 bytes"):
        ex.read_trec_run(write(tmp_path, "run.trec", "q1 Q0 a 1 2.0 t\n"))
    with pytest.raises(ValueError, match="larger than 10 bytes"):
        ex.read_minirag_hits(write(tmp_path, "hits.json", '{"results": []}'))


def hits_file(tmp_path: Path, results: object) -> Path:
    return write(tmp_path, "hits.json", json.dumps({"results": results}))


@pytest.mark.parametrize(
    ("results", "message"),
    [
        (["q1"], r"results\[0\]: expected an object with a 'hits' list"),
        ([{"query_id": "q1", "query": "x"}], "expected an object with a 'hits' list"),
        ([{"query_id": 1, "query": "x", "hits": []}], "unique string query_id"),
        (
            [{"query_id": "q1", "query": "x", "hits": []}] * 2,
            r"results\[1\]: needs a unique string query_id",
        ),
        ([{"query_id": "q1", "query": "x", "hits": [{"score": 1.0}]}], "a string doc_id"),
        (
            [{"query_id": "q1", "query": "x", "hits": [{"doc_id": "a", "score": True}]}],
            r"hits\[0\]: score must be a number",
        ),
        (
            [{"query_id": "q1", "query": "x", "hits": [{"doc_id": "a", "score": "0.5"}]}],
            "score must be a number",
        ),
        (
            [{"query_id": "q1", "query": "x", "hits": [{"doc_id": "a", "score": 10**400}]}],
            "score must be finite",
        ),
        (
            [{"query_id": "q1", "query": "x", "hits": [{"doc_id": "a", "score": 1.0}] * 2}],
            "lists a doc_id twice",
        ),
        ([{"query_id": "q\n1", "query": "x", "hits": []}], "query id must be 1-64 printable"),
    ],
)
def test_minirag_reader_rejects_malformed_hits(
    tmp_path: Path, results: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ex.read_minirag_hits(hits_file(tmp_path, results))


@pytest.mark.parametrize("text", ['[{"results": []}]', '{"result": []}', '{"results": {}}'])
def test_minirag_reader_needs_a_results_list(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError, match="expected an object with a 'results' list"):
        ex.read_minirag_hits(write(tmp_path, "hits.json", text))


def test_minirag_reader_rejects_non_finite_scores(tmp_path: Path) -> None:
    hit = '{"doc_id": "a", "score": NaN}'
    text = f'{{"results": [{{"query_id": "q1", "query": "x", "hits": [{hit}]}}]}}'
    with pytest.raises(ValueError, match="score must be finite"):
        ex.read_minirag_hits(write(tmp_path, "hits.json", text))


def test_minirag_reader_turns_deep_nesting_into_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nested too deeply"):
        ex.read_minirag_hits(write(tmp_path, "hits.json", "[" * 100_000))


def test_out_of_order_hits_reach_fuse_and_fail_there() -> None:
    with pytest.raises(ValueError, match="position 1 scores above the one before it"):
        ex.fuse_query([("a", 1.0), ("b", 2.0)], [("a", 0.5)])


def test_trace_keeps_both_retrievers_parts() -> None:
    bm25 = ex.read_trec_run(TREC)
    dense, _ = ex.read_minirag_hits(HITS)
    trace = ex.fuse_query(bm25["q1"], dense["q1"])

    assert [c.key for c in trace.candidates[:3]] == ["pluto", "asteroid_belt", "kuiper_belt"]
    pluto = {t.source: t for t in trace.candidates[0].retrievers}
    assert (pluto["bm25"].rank, pluto["dense"].rank) == (0, 1)
    assert pluto["bm25"].raw_score == 5.859364
    assert trace.candidates[0].fused_score == pytest.approx(1 / 61 + 1 / 62)

    mercury = next(c for c in trace.candidates if c.key == "mercury")
    absent = {t.source: t for t in mercury.retrievers}["dense"]
    assert (absent.rank, absent.raw_score, absent.contribution) == (None, None, 0.0)
    for candidate in trace.candidates:
        total = math.fsum(t.contribution for t in candidate.retrievers)
        assert candidate.fused_score == pytest.approx(total)


def test_report_says_when_nothing_came_back() -> None:
    assert ex.report(ex.fuse_query([], [])) == ["  (neither retriever returned anything)"]


def test_report_marks_what_a_retriever_did_not_return() -> None:
    bm25 = ex.read_trec_run(TREC)
    dense, _ = ex.read_minirag_hits(HITS)
    rows = ex.report(ex.fuse_query(bm25["q1"], dense["q1"]))
    saturn = next(row for row in rows if " saturn " in row)
    assert saturn.split()[3] == "-"
    assert rows[-1] == "  pluto: bm25 0.016393 + dense 0.016129 = 0.032522"
