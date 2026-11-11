"""The grids, quadrature and transforms keep their d68f8f6 values (gh-166).

The Chebyshev nodes and transforms, the Clenshaw–Curtis weights, the
Gauss–Legendre nodes and weights and the associated Legendre tables now come
from functax. ``tests/data/grids_d68f8f6.npz`` holds what spectraldiffx
computed at d68f8f6, before the switch, in float64: it was written by loading
the package from ``git archive d68f8f6`` and evaluating the public grids on
fixed (seeded, stored) inputs. Each test below compares today's output with
that record, at the tolerance the issue sets or the one measured, whichever is
looser, and says which.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import spectraldiffx as sdx
from spectraldiffx._src.spherical.grid import _gauss_legendre_nodes_weights

REF = np.load(Path(__file__).parent / "data" / "grids_d68f8f6.npz")

KINDS = {"gauss-lobatto": "lobatto", "gauss": "gauss"}


@pytest.mark.parametrize("node_type", sorted(KINDS))
@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 16, 33, 64])
@pytest.mark.parametrize("half", [1.0, 2.5])
def test_chebyshev_nodes(node_type, n, half):
    kind = KINDS[node_type]
    expected = REF[f"cheb_x_{kind}_{n}_{half}"]
    g1 = sdx.ChebyshevGrid1D.from_N_L(N=n, L=half, node_type=node_type)
    g2 = sdx.ChebyshevGrid2D.from_N_L(Nx=n, Ny=n, Lx=half, Ly=half, node_type=node_type)
    g3 = sdx.ChebyshevGrid3D.from_N_L(
        Nx=n, Ny=n, Nz=n, Lx=half, Ly=half, Lz=half, node_type=node_type
    )
    # rtol=1e-15 as the issue sets. functax evaluates the cosines in NumPy
    # float64 (d68f8f6 in jnp): measured 5.6e-16 absolute, and 2.3e-15 relative
    # only on nodes near x = 0, which the atol covers.
    for x in (g1.x, g2.x, g2.y, g3.x, g3.y, g3.z):
        np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("node_type", sorted(KINDS))
@pytest.mark.parametrize("n", [1, 2, 3, 8, 16, 33])
def test_chebyshev_transform_1d(node_type, n):
    kind = KINDS[node_type]
    grid = sdx.ChebyshevGrid1D.from_N_L(N=n, L=1.0, node_type=node_type)
    cheb = sdx.ChebyshevTransform1D(grid=grid)
    u = jnp.asarray(REF[f"cheb1d_u_{kind}_{n}"])
    a = jnp.asarray(REF[f"cheb1d_a_{kind}_{n}"])
    fwd, inv = REF[f"cheb1d_fwd_{kind}_{n}"], REF[f"cheb1d_inv_{kind}_{n}"]
    # rtol=1e-12 as the issue sets; measured bitwise equal in 1-D, 2-D and 3-D
    # (the FFT kernels are spectraldiffx's, moved).
    for out, expected in (
        (grid.transform(u), fwd),
        (cheb.to_spectral(u), fwd),
        (cheb(u), fwd),
        (grid.transform(a, inverse=True), inv),
        (cheb.from_spectral(a), inv),
        (cheb(a, inverse=True), inv),
    ):
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("node_type", sorted(KINDS))
@pytest.mark.parametrize(("nx", "ny"), [(8, 6), (16, 16), (5, 9)])
def test_chebyshev_transform_2d(node_type, nx, ny):
    key = f"{KINDS[node_type]}_{nx}_{ny}"
    grid = sdx.ChebyshevGrid2D.from_N_L(
        Nx=nx, Ny=ny, Lx=1.0, Ly=2.0, node_type=node_type
    )
    cheb = sdx.ChebyshevTransform2D(grid=grid)
    u = jnp.asarray(REF[f"cheb2d_u_{key}"])
    a = jnp.asarray(REF[f"cheb2d_a_{key}"])
    fwd, inv = REF[f"cheb2d_fwd_{key}"], REF[f"cheb2d_inv_{key}"]
    for out, expected in (
        (grid.transform(u), fwd),
        (cheb.to_spectral(u), fwd),
        (cheb(u), fwd),
        (grid.transform(a, inverse=True), inv),
        (cheb.from_spectral(a), inv),
        (cheb(a, inverse=True), inv),
    ):
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("node_type", sorted(KINDS))
def test_chebyshev_transform_3d(node_type):
    kind = KINDS[node_type]
    grid = sdx.ChebyshevGrid3D.from_N_L(
        Nx=4, Ny=5, Nz=6, Lx=1.0, Ly=2.0, Lz=0.5, node_type=node_type
    )
    u = jnp.asarray(REF[f"cheb3d_u_{kind}"])
    a = jnp.asarray(REF[f"cheb3d_a_{kind}"])
    np.testing.assert_allclose(
        np.asarray(grid.transform(u)), REF[f"cheb3d_fwd_{kind}"], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(grid.transform(a, inverse=True)),
        REF[f"cheb3d_inv_{kind}"],
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 32, 33, 64])
@pytest.mark.parametrize("half", [1.0, 2.5])
def test_clenshaw_curtis_weights(n, half):
    w = sdx.clenshaw_curtis_weights(n, half)
    # The issue asks for rtol=1e-15; measured 7.1e-15 relative (1.1e-16
    # absolute) at N = 64: functax sums the cosine series in another order.
    np.testing.assert_allclose(np.asarray(w), REF[f"cc_{n}_{half}"], rtol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 16, 32, 33, 64, 128, 256])
def test_gauss_legendre_nodes_and_weights(n):
    nodes, weights = _gauss_legendre_nodes_weights(n)
    grid = sdx.SphericalGrid2D.from_N_L(Nx=4, Ny=n)  # few orders: fast
    # The issue asks for rtol=1e-15. functax's rule is NumPy's leggauss,
    # d68f8f6's was SciPy's roots_legendre. The nodes agree to 2.2e-16
    # absolute (one ulp). The weights agree to 1.8e-14 absolute (n = 128) but
    # only 1.4e-10 relative (the smallest end weight, n = 256): against mpmath
    # at 40 digits leggauss is off by 2.1e-11 relative there and roots_legendre
    # by 1.3e-10, so the weights (O(1/n), summing to 2) are compared absolutely.
    for mu in (
        nodes,
        grid.cos_theta,
        sdx.SphericalGrid1D.from_N_L(n, np.pi).cos_theta,
    ):
        np.testing.assert_allclose(
            np.asarray(mu), REF[f"gl_nodes_{n}"], rtol=0, atol=4e-16
        )
    for w in (weights, grid.weights_y):
        np.testing.assert_allclose(
            np.asarray(w), REF[f"gl_weights_{n}"], rtol=0, atol=3e-14
        )


# The Legendre tables and transforms move by the Gauss–Legendre nodes above and
# by functax's mantissa-exponent recurrence. Measured worst gaps: 2.7e-15 (P̃ₗ),
# 1.3e-15 (P̃ₗ¹), 6.1e-15 (the 16-latitude tables), 1.2e-12 (T127, the bound
# the issue sets for geonnax's tables is 1e-11), 9.6e-15 and 5.3e-14 (the 1-D
# and 2-D forward transforms); each atol leaves a margin of 2 to 20.


def test_spherical_grid_1d():
    grid = sdx.SphericalGrid1D.from_N_L(N=16, L=np.pi)
    np.testing.assert_allclose(
        np.asarray(grid._P_matrix), REF["sph1d_P"], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(sdx.SphericalDerivative1D(grid)._P1_matrix),
        REF["sph1d_P1"],
        rtol=0,
        atol=1e-14,
    )
    u, c = jnp.asarray(REF["sph1d_u"]), jnp.asarray(REF["sph1d_c"])
    np.testing.assert_allclose(
        np.asarray(grid.transform(u)), REF["sph1d_fwd"], rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(grid.transform(c, inverse=True)),
        REF["sph1d_inv"],
        rtol=0,
        atol=1e-13,
    )


def test_spherical_grid_2d():
    grid = sdx.SphericalGrid2D.from_N_L(Nx=32, Ny=16)
    np.testing.assert_allclose(
        np.asarray(grid.legendre_table), REF["sph2d_table"], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(grid._P_lm), REF["sph2d_P_lm"], rtol=0, atol=1e-14
    )
    u, c = jnp.asarray(REF["sph2d_u"]), jnp.asarray(REF["sph2d_c"])
    np.testing.assert_allclose(
        np.asarray(grid.transform(u)), REF["sph2d_fwd"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(grid.transform(c, inverse=True)),
        REF["sph2d_inv"],
        rtol=0,
        atol=1e-13,
    )


def test_legendre_table_t127():
    grid = sdx.SphericalGrid2D.from_N_L(Nx=256, Ny=128)
    table = np.asarray(grid.legendre_table)[REF["t127_orders"]]
    np.testing.assert_allclose(table, REF["t127_table"], rtol=0, atol=1e-11)
