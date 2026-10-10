"""Generate, or check, the capability index: every public name in spectraldiffx.

``docs/api/capabilities.md`` lists each name in ``spectraldiffx.__all__`` once,
grouped exactly as the API reference groups it (one section per
``docs/api/<family>/<page>.md``, one sub-section per heading on that page),
with the first sentence of its docstring. It then lists the public API of
gaussx, the structured linear algebra spectraldiffx builds on. Agents and
people search it before writing a helper ("Reuse before you write" in
``AGENTS.md``).

Usage::

    make capabilities                                   # rewrite the index
    uv run python scripts/capabilities.py --check       # fail if stale

``tests/test_capabilities.py`` runs the check in the fast tier. The gaussx
section depends on the installed version, which the index records; when it
differs from the installed one, only the spectraldiffx part is compared.

``--check`` also fails when spectraldiffx exports a name that gaussx exports
for a different object, unless ``ALLOWED_SHARED_NAMES`` gives the reason.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import re
import sys
import types
import typing

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "api" / "capabilities.md"
API_DIR = ROOT / "docs" / "api"
MKDOCS = ROOT / "mkdocs.yml"

# Public homes, in order: (distribution, [public modules]).
PACKAGES: dict[str, list[str]] = {
    "spectraldiffx": ["spectraldiffx"],
}

UPSTREAM: dict[str, list[str]] = {
    "gaussx": ["gaussx"],
}
UPSTREAM_BLURB = {
    "gaussx": (
        "Structured linear algebra. spectraldiffx uses it for the masked-domain"
        " capacitance solver (`DiagonalisedOperator`, `circulant_from_symbol`,"
        " `MaskedOperator`, `grid_coupling_indices`, `solve`) and the Chebyshev"
        " Helmholtz solvers (`EigenFactorization`, `kronecker_sum_solve`)."
        " A new solve, factorisation or Woodbury / capacitance correction comes"
        " from here, not from a hand-written `jnp.linalg` call."
    ),
}

# Names spectraldiffx and an upstream library both export, and why.
ALLOWED_SHARED_NAMES: dict[str, str] = {
    "CapacitanceSolver": (
        "spectraldiffx's is the masked-domain Poisson / Helmholtz solver built by"
        " build_capacitance_solver (on gaussx.MaskedOperator); gaussx's is the"
        " generic point-constraint capacitance correction around any base solver"
    ),
}

_DIRECTIVE = re.compile(r"^:::\s*spectraldiffx\.(\w+)\s*$")
_AUTOREF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# A full stop that ends a sentence (not "e.g." / "i.e." / "etc.").
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)(?<!etc)\.\s+(?=[A-Z])")
UPSTREAM_MARKER = "<!-- upstream -->"


def _first_line(text: str) -> str:
    """The docstring's summary: its first sentence, on one line."""
    paragraph = text.strip().split("\n\n")[0]
    line = " ".join(part.strip() for part in paragraph.splitlines())
    # mkdocs-autorefs links resolve only in the docs that wrote them.
    line = _AUTOREF.sub(r"\1", line)
    sentence = _SENTENCE_END.split(line, maxsplit=1)[0]
    if sentence != line:
        line = sentence + "."
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return line.replace("|", "\\|")


def _is_alias(obj: object) -> bool:
    """A typing alias such as ``BoundaryCondition`` (a Union of Literals)."""
    return typing.get_origin(obj) is not None


def _alias_summary(obj: object) -> str:
    """Describe a Union-of-Literals alias by the values it accepts."""
    strings: list[str] = []
    pairs = 0

    def walk(alias: object) -> None:
        nonlocal pairs
        origin = typing.get_origin(alias)
        if origin is tuple:
            pairs += 1
        elif origin is typing.Literal:
            strings.extend(repr(a) for a in typing.get_args(alias))
        else:
            for arg in typing.get_args(alias):
                walk(arg)

    walk(obj)
    values = ", ".join(f"`{s}`" for s in strings)
    if pairs:
        values += f", or one of {pairs} supported `(left, right)` pairs"
    return f"Type alias: {values}."


def _kind(obj: object) -> str:
    if _is_alias(obj):
        return "type alias"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def _origin(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str) or not (callable(obj) or inspect.isclass(obj)):
        module = type(obj).__module__
    return module


def _summary(obj: object, *, home: str) -> str:
    if _is_alias(obj):
        return _alias_summary(obj)
    origin = _origin(obj).split(".")[0]
    if origin != home:
        return f"Re-exported from `{origin}`."
    if not callable(obj):
        value = repr(obj)
        return f"`{value if len(value) <= 60 else value[:57] + '...'}`".replace(
            "|", "\\|"
        )
    return _first_line(inspect.getdoc(obj) or "")


def _public(module: types.ModuleType) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return [
        n
        for n in names
        if not n.startswith("__")
        and not isinstance(getattr(module, n, None), types.ModuleType)
    ]


def _api_pages() -> list[Path]:
    """The API reference pages, in the order the MkDocs nav lists them."""
    nav = re.findall(r"\b(api/\S+?\.md)\b", MKDOCS.read_text(encoding="utf-8"))
    pages = [API_DIR.parent / p for p in dict.fromkeys(nav)]
    listed = set(pages)
    rest = sorted(p for p in API_DIR.rglob("*.md") if p not in listed)
    return [p for p in pages if p.is_file() and p != INDEX] + [
        p for p in rest if p != INDEX
    ]


def _page_layout(page: Path) -> tuple[str, list[tuple[str, list[str]]]]:
    """``(title, [(section heading, [names in directive order])])``."""
    title = page.stem
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ") and title == page.stem:
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif match := _DIRECTIVE.match(line):
            sections[-1][1].append(match.group(1))
    return title, [(head, names) for head, names in sections if names]


def collect() -> tuple[dict[str, dict[str, object]], list[str]]:
    """``{name: obj}`` for every public home, plus the shadowing clashes."""
    owners: dict[str, dict[str, object]] = defaultdict(dict)
    for modules in PACKAGES.values():
        for module_name in modules:
            module = importlib.import_module(module_name)
            for name in _public(module):
                owners[name][f"{module_name}.{name}"] = getattr(module, name)
    clashes = []
    for name, homes in sorted(owners.items()):
        if len({id(o) for o in homes.values()}) > 1:
            clashes.append(f"{name}: {', '.join(sorted(homes))}")
    for modules in UPSTREAM.values():
        for module_name in modules:
            try:
                upstream = importlib.import_module(module_name)
            except ImportError:
                continue
            for name in _public(upstream):
                if name in ALLOWED_SHARED_NAMES:
                    continue
                for path, obj in owners.get(name, {}).items():
                    if obj is not getattr(upstream, name):
                        clashes.append(f"{name}: {path} shadows {module_name}.{name}")
    return owners, clashes


def _table(rows: list[tuple[str, str, str]]) -> list[str]:
    out = ["| Name | Kind | What it does |", "|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} |" for r in rows]
    return [*out, ""]


def _row(module: types.ModuleType, name: str, home: str) -> tuple[str, str, str]:
    obj = getattr(module, name)
    return (name, _kind(obj), _summary(obj, home=home))


def _package_part() -> tuple[list[str], int]:
    out: list[str] = []
    total = 0
    for dist, modules in PACKAGES.items():
        for module_name in modules:
            module = importlib.import_module(module_name)
            public = _public(module)
            placed: set[str] = set()
            out += [f"## `{module_name}`", ""]
            for page in _api_pages():
                title, sections = _page_layout(page)
                sections = [
                    (head, [n for n in names if n in public])
                    for head, names in sections
                ]
                sections = [(head, names) for head, names in sections if names]
                if not sections:
                    continue
                family = page.parent.name.capitalize()
                rel = page.relative_to(ROOT).as_posix()
                out += [f"### {family}: {title} (`{rel}`)", ""]
                for head, names in sections:
                    if head:
                        out += [f"#### {head}", ""]
                    out += _table([_row(module, n, dist) for n in names])
                    placed.update(names)
            missing = sorted(set(public) - placed)
            if missing:
                out += ["### Not on an API page", ""]
                out += _table([_row(module, n, dist) for n in missing])
            total += len(public)
    return out, total


def _upstream_part() -> list[str]:
    out: list[str] = []
    for package, modules in UPSTREAM.items():
        out += [f"## Upstream: `{package}`", "", UPSTREAM_BLURB[package], ""]
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            rows = []
            for name in sorted(_public(module)):
                obj = getattr(module, name)
                rows.append((name, _kind(obj), _summary(obj, home=package)))
            out += [f"### `{module_name}`", "", *_table(rows)]
    return out


def _versions() -> str:
    return ", ".join(f"{p} {importlib.metadata.version(p)}" for p in UPSTREAM)


def render() -> str:
    package_part, total = _package_part()
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        f"Every public name in spectraldiffx ({total} of them), each listed once",
        "and grouped as the API reference groups it, with the first sentence of",
        "its docstring; then the public API of gaussx, which spectraldiffx builds",
        "on. Search this page before writing a helper: if what you need is here,",
        "compose it; if it is almost here, extend it where it lives. Regenerate",
        "with `make capabilities` after changing a public API",
        "(`tests/test_capabilities.py` checks it is current).",
        "",
        *package_part,
        UPSTREAM_MARKER,
        "",
        f"Listed at {_versions()}.",
        "",
        *_upstream_part(),
    ]
    return "\n".join(out).rstrip() + "\n"


def stale(current: str, text: str) -> bool:
    """Whether ``current`` differs from ``text`` where it is comparable.

    The upstream section is compared only when the recorded versions match
    the installed ones; the spectraldiffx part always is.
    """
    if current == text:
        return False
    if f"Listed at {_versions()}." in current:
        return True
    return current.partition(UPSTREAM_MARKER)[0] != text.partition(UPSTREAM_MARKER)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text = render()
    status = 0
    _, clashes = collect()
    if clashes:
        print("Names bound to different objects in two homes:")
        print("\n".join(f"  {c}" for c in clashes))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if stale(current, text):
            print(f"{INDEX.relative_to(ROOT)} is stale; run `make capabilities`")
            status = 1
    else:
        INDEX.write_text(text, encoding="utf-8")
        print(f"wrote {INDEX.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
