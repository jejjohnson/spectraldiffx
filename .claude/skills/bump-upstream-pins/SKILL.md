---
name: bump-upstream-pins
description: Bump spectraldiffx's git-pinned upstream, gaussx (a release tag in [tool.uv.sources] plus a >= floor in dependencies), across pyproject.toml, uv.lock and the capability index, then fix what the new version breaks in the capacitance and Chebyshev solvers. Use when asked to upgrade gaussx, after a gaussx release, or when a new gaussx feature is needed.
---

# Bump gaussx

gaussx is not on PyPI, so it is pinned to a release tag and a bump touches
several places that must agree.

## 1. Where the pin lives

| What | Where |
|---|---|
| The floor | `pyproject.toml`, `dependencies`: `gaussx>=X.Y.Z` |
| The tag | `pyproject.toml`, `[tool.uv.sources]`: `gaussx = { git = "https://github.com/jejjohnson/gaussx", tag = "vX.Y.Z" }` (gaussx tags carry a `v`; spectraldiffx's own tags do not) |
| The resolution | `uv.lock` |
| The upstream section of the capability index | `docs/api/capabilities.md` (records `Listed at gaussx X.Y.Z.`) |

Downstream, finitevolX pins gaussx and spectraldiffx separately; after a
bump here, say in the PR that its gaussx pin should move to a compatible
tag.

## 2. Bump

1. Read the gaussx changelog between the old and new tags; note removals,
   renames and deprecations (gaussx warns with `GaussxDeprecationWarning`
   and names the replacement). The names spectraldiffx uses:
   `DiagonalisedOperator`, `circulant_from_symbol`, `MaskedOperator`,
   `grid_coupling_indices`, `solve` (`_src/fourier/capacitance.py`) and
   `EigenFactorization`, `kronecker_sum_solve`
   (`_src/chebyshev/solvers.py`).
2. Update the floor and the tag.
3. `uv lock`, then `uv sync --all-extras`.
4. `make capabilities`: the gaussx section and its recorded version change.
   If gaussx now exports a name spectraldiffx also exports for something
   else, `tests/test_capabilities.py` fails: rename ours (with a
   deprecation, finitevolX imports it) or add it to `ALLOWED_SHARED_NAMES`
   in `scripts/capabilities.py` with the reason, as `CapacitanceSolver` is.

## 3. Fix and verify

```bash
uv run pytest tests/test_fourier_capacitance.py tests/test_chebyshev_solvers.py \
  tests/test_chebyshev_extensions.py tests/test_chebyshev_new_features.py \
  tests/test_tracing.py tests/test_float32.py \
  tests/test_capabilities.py -n auto
make test-fast
```

- `filterwarnings = error` turns a new gaussx warning into a failure:
  replace the deprecated call rather than silencing the warning.
- If gaussx now provides something spectraldiffx hand-rolls, note it in the
  PR; converting it is a separate change unless it is a one-liner.
- `make typecheck`, lint and format, then the `pre-pr-check` skill.

The PR title is `build(deps): bump gaussx to vX.Y.Z`.
