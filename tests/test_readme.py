"""The README is the product page; these tests keep it runnable and honest.

A reader installs the tag the README names, then pastes its examples. The
examples run here against the installed (in CI, editable) package, and outputs
shown under them are compared line for line. The install lines are checked as
text only: running them needs the network and a fresh venv. Names that are on
``main`` but not in the tagged release must be marked as such, so the install
line and the API table cannot disagree about what a reader gets. Bumping
``__version__`` therefore means updating the README's install tag too.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ragfuse

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
REPO = "git+https://github.com/Bruhadev45/ragfuse"
MAIN_ONLY = "*(main only)*"
HEADING = re.compile(r"(#{1,6}) +(.+?)\s*$")


def outside_fences(lines: list[str]) -> list[tuple[str, bool]]:
    """Each line with whether it sits inside a ``` code fence (fence lines count as inside)."""
    marked: list[tuple[str, bool]] = []
    fenced = False
    for line in lines:
        if line.startswith("```"):
            fenced = not fenced
            marked.append((line, True))
        else:
            marked.append((line, fenced))
    return marked


def section(heading: str, text: str = README) -> list[str]:
    """Lines under ``heading``, up to the next heading of the same or a higher level."""
    lines = outside_fences(text.splitlines())
    starts = [i for i, (line, fenced) in enumerate(lines) if line == heading and not fenced]
    if not starts:
        pytest.fail(f"no heading {heading!r}")
    level = len(heading) - len(heading.lstrip("#"))
    body: list[str] = []
    for line, fenced in lines[starts[0] + 1 :]:
        match = None if fenced else HEADING.match(line)
        if match and len(match[1]) <= level:
            break
        body.append(line)
    return body


def blocks(lines: list[str]) -> list[tuple[str, str]]:
    """``(language, body)`` for each fenced block in ``lines``."""
    found: list[tuple[str, str]] = []
    lang: str | None = None
    body: list[str] = []
    for line in lines:
        if lang is None and line.startswith("```"):
            lang, body = line[3:].strip(), []
        elif lang is not None and line.rstrip() == "```":
            found.append((lang, "\n".join(body)))
            lang = None
        elif lang is not None:
            body.append(line)
    return found


def example_and_output(heading: str) -> tuple[str, list[str]]:
    """The first python block in a section and the output block right after it."""
    found = blocks(section(heading))
    langs = [lang for lang, _ in found]
    assert "python" in langs, f"{heading!r} has no python block"
    i = langs.index("python")
    assert langs[i + 1 : i + 2] == [""], f"{heading!r}: no output block after the example"
    code, output = found[i][1], found[i + 1][1]
    assert code.strip() and output.strip(), f"{heading!r}: empty example or output"
    return code, output.splitlines()


def run(code: str) -> list[str]:
    """Stdout lines of ``code`` run in a fresh interpreter, as a reader would paste it."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        cwd=ROOT,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout.rstrip("\n").splitlines()


def names_in(text: str) -> set[str]:
    """Public names inside backticks, e.g. ``combsum`` in `ragfuse.combsum(...)`."""
    spans = re.findall(r"`([^`\n]+)`", text)
    return {word for span in spans for word in re.findall(r"\w+", span)} & set(ragfuse.__all__)


def changelog_releases() -> dict[str, list[str]]:
    """Lines under each ``## [name]`` heading of the CHANGELOG, by name."""
    releases: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in CHANGELOG.splitlines():
        if match := re.match(r"## \[([^\]]+)\]", line):
            current = releases.setdefault(match[1], [])
        elif current is not None:
            current.append(line)
    return releases


def test_install_line_pins_the_current_version_tag() -> None:
    commands = [body for lang, body in blocks(section("## Install")) if lang == "bash"]
    assert commands, "the Install section has no bash block"
    expected = f'pip install "ragfuse @ {REPO}@v{ragfuse.__version__}"'
    assert commands[0].splitlines() == [expected]


def test_every_pip_install_of_ragfuse_points_at_this_repository() -> None:
    """The PyPI name belongs to an unrelated package; never tell a reader to use it."""
    checked = [
        line
        for lang, body in blocks(README.splitlines())
        if lang in {"bash", "sh", "console", "shell"}
        for line in body.splitlines()
        if "pip install" in line and "ragfuse" in line
    ]
    assert checked, "expected at least one install line"
    for line in checked:
        assert REPO in line, line


def test_quickstart_prints_the_output_shown_under_it() -> None:
    code, output = example_and_output("## Quickstart")
    assert run(code) == output


def test_provenance_example_continues_from_the_quickstart() -> None:
    quickstart, quickstart_output = example_and_output("## Quickstart")
    code, output = example_and_output("## Every hit explains itself")
    assert run(f"{quickstart}\n{code}") == quickstart_output + output


@pytest.mark.parametrize(
    "heading",
    ["### Migrating off the duplicate-list trick", "## Which fusion?", "### Two precisions at k"],
)
def test_standalone_examples_run(heading: str) -> None:
    """These examples define everything they use, so each must run on its own."""
    code = next((body for lang, body in blocks(section(heading)) if lang == "python"), None)
    assert code, f"{heading!r} has no python block"
    run(code)


def test_api_table_mentions_every_public_name() -> None:
    documented = names_in("\n".join(section("## API")))
    assert set(ragfuse.__all__) - {"__version__"} <= documented


def test_main_only_marks_match_the_unreleased_additions() -> None:
    """A name added since the last tag is marked in the API table, and only those are.

    New means named under Unreleased > Added and in no released section, so a
    bullet that mentions an existing name such as ``FusedHit`` does not count.
    Once a release empties Unreleased, no row may stay marked.
    """
    releases = changelog_releases()
    assert "Unreleased" in releases, "CHANGELOG.md has no [Unreleased] section"
    unreleased = releases.pop("Unreleased")
    added_lines = section("### Added", "\n".join(unreleased)) if "### Added" in unreleased else []
    released = names_in("\n".join(line for body in releases.values() for line in body))
    added = names_in("\n".join(added_lines)) - released
    marked = {
        name
        for line in section("## API")
        if line.startswith("|") and MAIN_ONLY in line
        for name in names_in(line.split("|")[1])
    }
    assert marked == added


def test_in_page_links_reach_a_heading() -> None:
    """GitHub's heading slugs: lower case, punctuation dropped, spaces to hyphens, -N repeats."""
    slugs: set[str] = set()
    seen: dict[str, int] = {}
    for line, fenced in outside_fences(README.splitlines()):
        if not fenced and (match := HEADING.match(line)):
            slug = re.sub(r"[^\w\- ]", "", match[2].lower()).replace(" ", "-")
            slugs.add(f"{slug}-{seen[slug]}" if slug in seen else slug)
            seen[slug] = seen.get(slug, 0) + 1
    targets = set(re.findall(r"\]\(#([^)]+)\)", README))
    assert targets, "expected in-page links"
    assert targets <= slugs, targets - slugs
