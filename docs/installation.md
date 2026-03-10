# Installation

## Requirements

- Python ≥ 3.12, < 3.14
- JAX (CPU or GPU/TPU backend)

---

## Install from GitHub

The simplest way to install SpectralDiffX:

```bash
pip install git+https://github.com/jejjohnson/spectraldiffx
```

---

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management.

### Clone and install

```bash
git clone https://github.com/jejjohnson/spectraldiffx.git
cd spectraldiffx
uv sync --group dev
```

### Install with all optional dependencies

To install the full suite including tests, docs, and example scripts:

```bash
uv sync --all-extras
```

---

## Optional Dependencies

| Extra | Contents |
|-------|----------|
| `test` | `pytest`, `pytest-cov` |
| `docs` | `mkdocs-material`, `mkdocstrings`, `mkdocs-jupyter` |
| `examples` | `diffrax`, `cyclopts`, `xarray`, `matplotlib`, `tqdm` |

Install a specific extra:

```bash
pip install "spectraldiffx[examples] @ git+https://github.com/jejjohnson/spectraldiffx"
```

---

## Verify Installation

```python
import jax
import jax.numpy as jnp
from spectraldiffx import FourierGrid1D, SpectralDerivative1D

jax.config.update("jax_enable_x64", True)

grid = FourierGrid1D.from_N_L(N=64, L=2 * jnp.pi)
deriv = SpectralDerivative1D(grid=grid)

u = jnp.sin(grid.x)
du_dx = deriv(u, order=1)

error = jnp.max(jnp.abs(du_dx - jnp.cos(grid.x)))
print(f"Max error: {error:.2e}")  # should be < 1e-12
```
