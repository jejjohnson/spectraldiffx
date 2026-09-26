"""Every public name has an API reference entry (gh-118).

Walks ``spectraldiffx.__all__`` and checks that each name appears in a
mkdocstrings ``::: spectraldiffx.<name>`` directive under ``docs/api/``, so
new public API cannot ship without a docs page.
"""

from pathlib import Path
import re

import spectraldiffx

API_DOCS = Path(__file__).resolve().parents[1] / "docs" / "api"
DIRECTIVE = re.compile(r"^:::\s*spectraldiffx\.(\w+)\s*$", re.MULTILINE)


def _documented_names() -> set[str]:
    names: set[str] = set()
    for page in API_DOCS.rglob("*.md"):
        names.update(DIRECTIVE.findall(page.read_text()))
    return names


def test_every_public_name_has_an_api_directive():
    missing = sorted(set(spectraldiffx.__all__) - _documented_names())
    assert not missing, (
        f"Add a '::: spectraldiffx.<name>' entry under docs/api/: {missing}"
    )


def test_api_directives_point_at_public_names():
    stale = sorted(_documented_names() - set(spectraldiffx.__all__))
    assert not stale, f"docs/api/ documents names not in spectraldiffx.__all__: {stale}"
