# Package Guidelines

Standing instructions specific to the `spectraldiffx` source package.

---

## Documentation

* Use numpy-style docstrings for all functions and classes.
* Include examples in docstrings where applicable.
* Ensure all private methods, public methods and classes have docstrings.
* Maintain consistency in terminology and style throughout the documentation.
* Use mathematics notation where applicable, ensuring clarity and correctness.
* Use ascii math, not latex.
* Keep track of array sizes throughout the docstrings and comments.
* Be pedantic and pedagogical in the explanations.
* All scientific algorithms should include Unicode equations in docstrings
  (e.g. `# σ² = Σ(xᵢ − μ)² / N`).

## Packages

* `equinox` for PyTree-compatible dataclasses
* `jaxtyping` for array type annotations
* `jax` for all numerical computations
* `beartype` for runtime type checking

## Testing

* Don't group so many things into a single test.
* Use fixtures for different equations.
* Enable JAX 64-bit precision in tests via `conftest.py`.

