---
applyTo: "spectraldiffx/**/*.py, tests/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `dataclasses`, `equinox.Module`, or `attrs` for data containers
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

| Purpose | Preferred Package |
|---------|-------------------|
| Logging | `loguru` |
| Data containers | `equinox` (JAX-compatible PyTrees) |
| Configuration | `hydra-core` / `omegaconf` |
| Path handling | `pathlib` (stdlib) |
| Array type hints | `jaxtyping` |
| Runtime type checks | `beartype` |
| Testing | `pytest` |

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (numpy style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- Public classes and functions should include 2–3 example use cases in docstrings
- Track array shapes in docstrings (e.g. `x : Float[Array, "N"]`)
