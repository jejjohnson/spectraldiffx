"""Float32 smoke tests (gh-117).

``tests/conftest.py`` turns on ``jax_enable_x64`` for the whole session, so
without this file the package default (x64 off, float32) is never exercised.
Each test runs inside ``jax.enable_x64(False)``. It checks that the result
stays float32, and that it matches a float64 reference. Measured relative
differences are 1.4e-7 to 4.1e-7 (eps32 ≈ 1.2e-7) for N = 64 to 1024, so
each bound below is about 10x the worst measured value.
Any warning, such as a float64 request being truncated, fails the test
through the ``filterwarnings = ["error", ...]`` setting in pyproject.toml.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraldiffx as sdx


@pytest.fixture
def x32():
    with jax.enable_x64(False):
        yield


def _field(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def _rel_err(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


@pytest.mark.parametrize("kind", ["dct", "dst"])
@pytest.mark.parametrize("type_", [1, 2, 3, 4])
def test_transform_roundtrip(x32, kind, type_):
    forward = sdx.dctn if kind == "dct" else sdx.dstn
    inverse = sdx.idctn if kind == "dct" else sdx.idstn
    x = jnp.asarray(_field((16, 12)), dtype=jnp.float32)
    y = forward(x, type=type_, axes=[0, 1])
    x_rec = inverse(y, type=type_, axes=[0, 1])
    assert y.dtype == jnp.float32
    assert x_rec.dtype == jnp.float32
    # Measured 1.7e-7 to 2.3e-7 over types 1-4.
    assert _rel_err(x_rec, np.asarray(x, dtype=np.float64)) < 2e-6


@pytest.mark.parametrize("n", [64, 256])
def test_poisson_dst_matches_float64(x32, n):
    f64 = _field((n, n))
    psi32 = sdx.solve_poisson_dst(jnp.asarray(f64, dtype=jnp.float32), 1.0, 1.0)
    assert psi32.dtype == jnp.float32
    with jax.enable_x64(True):
        psi64 = np.asarray(sdx.solve_poisson_dst(jnp.asarray(f64), 1.0, 1.0))
    # Measured 2e-7 to 3e-7 for N = 64 to 1024 (docs/installation.md).
    assert _rel_err(psi32, psi64) < 3e-6


def _dense_masked_helmholtz(mask, f, lam):
    """Float64 dense solve of the five-point (∇² − λ)ψ = f on the interior
    cells (wet, no dry 4-neighbour), with ψ = 0 elsewhere; dx = dy = 1."""
    pad = np.pad(mask, 1)
    interior = mask & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    idx = -np.ones(mask.shape, dtype=int)
    cells = np.argwhere(interior)
    idx[interior] = np.arange(len(cells))
    A = np.zeros((len(cells), len(cells)))
    for row, (j, i) in enumerate(cells):
        A[row, row] = -4.0 - lam
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if idx[j + dj, i + di] >= 0:
                A[row, idx[j + dj, i + di]] = 1.0
    psi = np.zeros(mask.shape)
    psi[interior] = np.linalg.solve(A, f[interior])
    return psi


def test_capacitance_matches_float64(x32):
    ny, nx = 16, 14
    j, i = np.mgrid[:ny, :nx]
    mask = np.hypot(j - 7.5, i - 6.5) < 5.5  # does not touch the edges
    f64 = _field((ny, nx), seed=1)

    solver = sdx.build_capacitance_solver(mask, 1.0, 1.0, lambda_=1.0)
    psi32 = jax.jit(lambda s, f: s(f))(solver, jnp.asarray(f64, dtype=jnp.float32))
    assert psi32.dtype == jnp.float32
    psi64 = _dense_masked_helmholtz(mask, f64, 1.0)
    assert np.all(np.asarray(psi32)[psi64 == 0.0] == 0.0)
    # Measured 1.4e-7 to 4.1e-7 over every base and λ ∈ {0, 1}.
    assert _rel_err(psi32, psi64) < 4e-6


def test_spherical_transform_roundtrip(x32):
    grid = sdx.SphericalGrid2D.from_N_L(32, 16)
    phi, theta = grid.X
    u = jnp.sin(theta) * jnp.cos(phi)
    u_hat = grid.transform(u)
    u_rec = grid.transform(u_hat, inverse=True)
    assert grid.l.dtype == jnp.float32
    assert jnp.real(u_rec).dtype == jnp.float32
    assert _rel_err(jnp.real(u_rec), np.asarray(u, dtype=np.float64)) < 2e-6
