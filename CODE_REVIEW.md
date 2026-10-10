# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. spectraldiffx is a JAX library of pseudospectral transforms,
derivatives, filters and elliptic solvers: most defects worth finding are
about **spectral numerics** (a transform paired with the wrong boundary
condition, a wavenumber missing its 2π/L, a nonlinear product left aliased
or a linear operator dealiased, a null mode or resonance handled silently,
a float32 input promoted to float64, a Python branch on a traced value) or
**boundaries** (a re-implemented DCT, eigenvalue formula or gaussx solve; a
renamed name finitevolX imports), not style. Read "Boundaries", "Reuse
before you write" and "The contracts" in [`AGENTS.md`](AGENTS.md) first;
this file is the checklist and the report format.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order, `__all__` order, the numpydoc section format, a missing
`docs/api` entry); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against
  [`docs/api/capabilities.md`](docs/api/capabilities.md) (spectraldiffx and
  gaussx) and the private helpers in "Reuse before you write". A
  re-implemented wavenumber array, 2/3 mask, DCT / DST, Laplacian eigenvalue
  formula, BC dispatch, spectral solve or gaussx factorisation is a **High**
  finding, with the existing name to use.
- gaussx through its public API only; no new dense `jnp.linalg` solve where
  gaussx has the structure (`EigenFactorization`, `kronecker_sum_solve`,
  `MaskedOperator`).
- No import across families (fourier / chebyshev / spherical); inside a
  family, nothing imports upward from the grid.
- A renamed or removed public name, or a changed signature, keeps the old
  one working with a `DeprecationWarning` naming the replacement
  (finitevolX imports the solvers, eigenvalues and transforms by name).

### 2. Grids and conventions

- Domain conventions kept: Fourier `[0, L)` with `x_j = j·L/N`; Chebyshev
  `[−L, L]` with `L` the half-length and decreasing nodes; spherical
  Gauss–Legendre colatitude with `R = Ly / π`.
- Wavenumbers from the grid (`k`, `KX`, `K2`), in FFT order, with their
  2π/L; axis order `(…, z, y, x)` (a swapped `kx` / `ky` passes on square
  grids: ask for an anisotropic test).
- Validation of grid fields in `__check_init__`, and only for concrete
  values (a grid must still build inside `jit`).

### 3. Operators, filters and dealiasing

- Linear operators keep every resolved mode; nonlinear products are
  truncated (factors and product), through `apply_dealias`, `jacobian`,
  `advection_scalar` or `cheb_dealias_product`.
- The Nyquist mode of an even grid: an odd derivative of it must vanish
  (the Fourier operators get this by taking `.real` of the inverse
  transform; a new path, e.g. one through `rfft`, must keep it).
  Hyperviscosity always dissipative; `inverse_laplacian` zero-mean.
- Complex physical input rejected (`_real`); results real and the input's
  shape; `spectral=True` paths consistent with the physical ones.
- A new public operator joins `CASES` in `tests/test_tracing.py`.

### 4. Transforms

- scipy's definition and normalisation for both `norm=None` and
  `"ortho"`, with the round trip through the inverse; registered in
  `_DCT_IMPLS` / `_DST_IMPLS` and the `_idct_along_axis` /
  `_idst_along_axis` inverses.
- `type` / `norm` / `axes` treated as static; real input only; output in
  the input dtype; checked against `scipy.fft` at odd and prime lengths
  (`tests/test_transforms_scipy.py`).

### 5. Elliptic solvers

- The equation is `(∇² − λ)ψ = f`; `lambda_` in functions, `alpha` in
  classes; no sign flips.
- The BC ↔ transform ↔ eigenvalue triple comes from `_BC_DISPATCH` (a
  regular-grid Dirichlet axis is DST-I on the interior points, staggered is
  DST-II; Neumann DCT-I / DCT-II; mixed pairs DST / DCT III / IV), and the
  grid placement the docstring states matches it.
- FD2 vs pseudo-spectral eigenvalues stated, and the default unchanged
  (`approximation="fd2"` for the functions, continuous `k²` for the
  `SpectralHelmholtzSolver*` classes).
- The null mode set to zero only where the denominator is zero (gh-92);
  `zero_mean` semantics kept; a resonant `λ` raising through
  `_check_finite` rather than being zeroed (gh-94).
- Inhomogeneous BC corrections only with FD2 eigenvalues, never on a
  periodic axis; BC arguments static under `jit`.
- A new BC or solver checked against a dense matrix
  (`tests/test_solvers_dense.py`, `dense_laplacian_1d`).

### 6. JAX numerics

- No Python `if` / `float()` / `.item()` / `np.asarray` on a value derived
  from an array argument; construction-time NumPy only on concrete values.
- Dtypes follow the input: constants built with `dtype=x.dtype`; a bare
  Python scalar combined with an array is weakly typed and fine.
- `jnp.where` branches that divide are safe (`jnp.where(d == 0, 1, d)`
  before the division), so gradients stay finite.
- `eqx.Module` (never a dataclass) for anything holding arrays; code-path
  settings static.

### 7. Public API and documentation

- New names: exported from the family `__init__.py` and
  `spectraldiffx/__init__.py`, a `::: spectraldiffx.<Name>` entry on the
  right `docs/api/<family>/<page>.md` page, `docs/api/capabilities.md`
  regenerated.
- Numpy-style docstrings on every function and class: the equation in
  plain text (Unicode / ASCII, no LaTeX), array shapes, BC and grid
  placement for solvers, a reference for a published method, an
  `Examples` section that actually runs (CI does not collect doctests).
- `python` fences in `README.md`, `docs/*.md`, `docs/theory/*.md` run
  (`tests/test_docs.py`); notebooks stay jupytext `.py` with figures saved
  under `docs/images/<notebook>/`.

### 8. Tests

- Against a closed form, `scipy.fft`, a dense matrix or a float64
  reference, with the tolerance's provenance in a comment.
- One behaviour per test; fixtures for different equations and grids;
  non-square, anisotropic grids where axes could be swapped.
- Tier markers right: unmarked for the fast tier, `slow` above ~3 s
  (only the expensive cases of a sweep).

### 9. Modern Python and dependencies

- Type hints on every public function; `X | None`; specific exceptions
  with `raise ... from ...`; guard clauses over deep nesting.
- No new runtime dependency without discussion; `uv.lock` updated.

---

## spectraldiffx-Specific Checks

### Wavenumbers come from the grid

```python
# ❌ Mode numbers, not wavenumbers: wrong by 2π/L unless L = 2π
k = jnp.fft.fftfreq(N) * N
du_dx = jnp.fft.ifft(1j * k * jnp.fft.fft(u)).real

# ✅ The grid owns k = 2π·fftfreq(N, dx); the operator rejects complex input
grid = sdx.FourierGrid1D.from_N_L(N=N, L=L)
du_dx = sdx.SpectralDerivative1D(grid).gradient(u)
```

On `N = 64`, `L = 1`, `u = sin(2πx)` the first is off by 5.3 (a factor
2π); the second matches `2π cos(2πx)` to 5e-14.

### Truncate nonlinear products

```python
# ❌ sin(8x) · ∂ₓcos(10x) on N = 32 creates cos(18x), which aliases onto cos(14x)
dq_dx, dq_dy = deriv.gradient(q)
adv = vx * dq_dx + vy * dq_dy

# ✅ Factors and product truncated with the grid's strict 2/3 mask
adv = deriv.advection_scalar(vx, vy, q)
```

The first leaves an amplitude-5 `cos(14x)` in the truncated band; the
second returns exactly `−5 cos(2x)` (error 1e-14). A *linear* operator,
by contrast, is never dealiased.

### The boundary condition picks the transform

```python
# ❌ An FFT solve imposes periodicity on the channel walls
psi = sdx.solve_poisson_fft(rhs, dx, dy)

# ✅ FFT along x, DST-I along y (homogeneous Dirichlet walls, interior points)
psi = sdx.solve_poisson_2d(rhs, dx, dy, bc_x="periodic", bc_y="dirichlet")
```

For `ψ = sin(2πx)·sin(πy)` on a 32 × 15 channel the first is off by 0.41;
the second recovers ψ to 2e-15 (it is the exact inverse of the FD2
Laplacian).

### Constants in the input's dtype

```python
# ❌ (-1.0) ** n is float64 under x64, so a float32 input comes back float64
sign = (-1.0) ** jnp.arange(N)

# ✅ (as _alternating_sign does)
sign = (1 - 2 * (jnp.arange(N) % 2)).astype(x.dtype)
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
