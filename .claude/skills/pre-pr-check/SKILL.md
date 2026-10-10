---
name: pre-pr-check
description: Run spectraldiffx's full pre-PR verification — ruff lint and format on the whole repo, ty on the package, the fast test tier as CI runs it, the slow tests of what changed, the API-docs and capability-index checks, the docs snippets, the lockfile and the strict MkDocs build. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings, docs or notebooks.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Always

```bash
uv run ruff check .                 # entire repo: spectraldiffx/, tests/, scripts/ (lint.yml)
uv run ruff format --check .
make typecheck                      # ty check spectraldiffx (typecheck.yml)
make test-fast                      # pytest tests -n auto -m "not slow and not integration" (ci.yml)
```

ruff excludes `docs/` and `notebooks/` (`[tool.ruff] exclude`), so the
notebook `.py` files are not linted; read them yourself.

## The slow tests of what you touched

PR CI skips `slow` and `integration`; `tests-extended.yml` runs them
nightly. Run them for the files you changed so the nightly run is not the
first to see them (a plain path selects every tier, since `pyproject.toml`
sets no default `-m`):

```bash
uv run pytest tests/<file>.py -n auto
make test-slow                      # every slow + integration test
```

The slow tests live in `test_solvers_dense.py` (the 72 mixed-BC pairs),
`test_fourier_solvers.py`, `test_fourier_mixed_bc_3d_solvers.py`,
`test_inhomogeneous_bcs.py`, `test_chebyshev_extensions.py` and
`test_spherical_ylm.py`.

## When the public API changed

```bash
make capabilities
uv run pytest tests/test_capabilities.py tests/test_api_docs.py tests/test_tracing.py
```

…plus the `::: spectraldiffx.<Name>` entry on its `docs/api/<family>/*.md`
page, `__all__` sorted (ruff `RUF022`), and a `CASES` entry in
`tests/test_tracing.py` for a new operator or solver. A rename or removal
needs a `DeprecationWarning` alias: finitevolX imports these names.

## When dependencies changed

`uv lock` and commit `uv.lock`; for gaussx, the `bump-upstream-pins` skill.

## When docs, docstrings or notebooks changed

```bash
uv run pytest tests/test_docs.py    # every python fence in README.md, docs/*.md, docs/theory/*.md
make docs                           # mkdocs build --strict (docs.yml)
make nb-check                       # no .ipynb under notebooks/
```

CI does not collect doctests: run a new `Examples` section yourself
(`uv run python -c "..."`). If `make docs` sits in a `pandoc` subprocess
for minutes (mkdocs-jupyter asks jupytext to read every `.md` page, and
jupytext shells out to pandoc when it is installed), rerun it with pandoc
off the `PATH`.

## Before pushing

- `git status` shows no stray files (`.plans/`, `.ipynb`, `site/`,
  `reports/`), and `uv.lock` changed only if you meant it to.
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
