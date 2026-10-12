"""Release hygiene: the version has one source, and every release has notes.

The release workflow publishes the CHANGELOG section for the tagged version as
the GitHub Release body, so a missing or empty section would ship a release
with no notes. These tests catch that before the tag is pushed.
"""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import ragfuse

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def _changelog_sections() -> dict[str, tuple[str, str]]:
    """Map each ``## [version] - date`` heading to its (date, body)."""
    heading = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>\S+))?$", re.MULTILINE)
    text = CHANGELOG.read_text(encoding="utf-8")
    matches = list(heading.finditer(text))
    sections: dict[str, tuple[str, str]] = {}
    for match, following in zip(matches, [*matches[1:], None], strict=True):
        end = following.start() if following else len(text)
        sections[match["version"]] = (match["date"] or "", text[match.end() : end].strip())
    return sections


def test_version_is_a_plain_semver_triple() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", ragfuse.__version__)


def test_installed_metadata_reads_the_module_version() -> None:
    """``pyproject.toml`` takes its version from ``__init__.py``; they cannot disagree."""
    assert version("ragfuse") == ragfuse.__version__


def test_changelog_has_a_dated_entry_for_the_current_version() -> None:
    date, body = _changelog_sections()[ragfuse.__version__]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
    assert body.startswith("### "), "the release notes should open with a Keep a Changelog group"


def test_newest_released_entry_is_the_current_version() -> None:
    released = [name for name in _changelog_sections() if name != "Unreleased"]
    assert released[0] == ragfuse.__version__
