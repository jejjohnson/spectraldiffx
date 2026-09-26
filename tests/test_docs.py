"""Execute the ``python`` code fences in the docs and README (gh-95).

Each page runs top to bottom in one namespace (``memory=True``), so later
fences may use names defined earlier on the same page. A page is excluded
only when its fences are API sketches rather than examples; the reason is
recorded below.
"""

from pathlib import Path

from mktestdocs import check_md_file
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Signature sketches (`psi = solve_helmholtz_dst(rhs, dx, dy, lambda_)`),
# not runnable examples; the runnable versions live in the guides.
_SKETCHES = {"docs/theory/elliptic_solvers.md"}

PAGES = sorted(
    str(p.relative_to(ROOT))
    for p in [
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("docs/theory/*.md"),
        ROOT / "README.md",
    ]
    if str(p.relative_to(ROOT)) not in _SKETCHES
)


@pytest.mark.parametrize("page", PAGES)
def test_markdown_snippets_run(page):
    check_md_file(fpath=ROOT / page, memory=True)
