---
applyTo: "spectraldiffx/**/*.py, tests/**/*.py, scripts/**/*.py"
---

# Python Coding Standards

`AGENTS.md` at the repo root is the source of truth (the contracts, reuse
table, tests and checklist); this file is the short form for these paths.

## Modern Python (3.12+)

- `from __future__ import annotations` in new modules
- Type hints on **all** public functions and methods; jaxtyping shapes on
  array arguments and returns (`Float[Array, "Ny Nx"]`, quoted)
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `equinox.Module` for grids, operators, filters, transforms and solvers
  (immutable pytrees; a dataclass is not one); code-path settings as
  `eqx.field(static=True)`, validation in `__check_init__`
- `Literal[...]` for fixed sets of string options (`"2/3"`, BC names)
- Specific exception types (never bare `except:`); `eqx.error_if` for
  failures that must surface under `jit`
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## JAX numerics

- No Python control flow on traced values; validate only concrete values
- Don't promote the input dtype: build constants with `dtype=x.dtype`
  (the transforms and the Fourier `solve_*` functions keep float32 even with
  x64 on)
- Physical-space input is real; complex input raises
- Prefer JAX-native, vectorised operations over Python loops

## Package Preferences

No new runtime dependency without discussion; build on what spectraldiffx
already depends on (see "Boundaries" in `AGENTS.md`).

| Purpose | Preferred Package |
|---------|-------------------|
| Arrays, FFTs, transforms of functions | `jax` |
| Modules / pytrees | `equinox` |
| Structured linear algebra (factorisations, masked / diagonalised operators) | `gaussx` |
| Array type hints | `jaxtyping` |
| Construction-time precomputation on concrete values | `numpy`, `scipy` |
| Path handling | `pathlib` (stdlib) |
| Testing | `pytest` (x64 on via `tests/conftest.py`) |

## Documentation

- Module-level docstrings explaining purpose
- Numpy-style docstrings (`Parameters`, `Returns`, `Raises`, `Examples`)
  for every module, class and function, private ones included
- Equations in plain text, Unicode or ASCII (e.g. `λ_k = −4/dx² · sin²(πk/(2N))`),
  never LaTeX
- Track array shapes in docstrings and comments (e.g. `rhs : Float[Array, "Ny Nx"]`)
- Inline comments explaining *why*, not *what*
- Public classes and functions include an `Examples` section; CI does not
  run doctests, so run new examples yourself
