"""
Tests for ChebyshevGrid1D and ChebyshevGrid2D.
"""

import jax.numpy as jnp

from spectraldiffx._src.chebyshev.grid import ChebyshevGrid1D, ChebyshevGrid2D

# ============================================================================
# ChebyshevGrid1D — Gauss-Lobatto nodes
# ============================================================================


def test_chebyshev_grid_1d_gauss_lobatto_nodes():
    """GL grid: N+1 nodes, includes endpoints ±L."""
    N, L = 8, 1.0
    grid = ChebyshevGrid1D.from_N_L(N=N, L=L)
    x = grid.x
    assert x.shape == (N + 1,)
    # Endpoints must be ±L
    assert jnp.isclose(x[0], L, atol=1e-6), f"x[0]={x[0]} should be {L}"
    assert jnp.isclose(x[N], -L, atol=1e-6), f"x[N]={x[N]} should be {-L}"


def test_chebyshev_grid_1d_gauss_nodes():
    """Gauss grid: N nodes, excludes endpoints."""
    N, L = 8, 1.0
    grid = ChebyshevGrid1D.from_N_L(N=N, L=L, node_type="gauss")
    x = grid.x
    assert x.shape == (N,)
    # No node should be exactly ±L
    assert not jnp.any(jnp.isclose(jnp.abs(x), L, atol=1e-8))


def test_chebyshev_grid_1d_node_ordering():
    """Nodes should be monotonically decreasing (from +L to -L)."""
    grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0)
    x = grid.x
    diffs = jnp.diff(x)
    assert jnp.all(diffs < 0), "Nodes should be monotonically decreasing"


def test_chebyshev_grid_1d_node_symmetry():
    """GL nodes are symmetric: x[j] = -x[N-j]."""
    N, L = 10, 2.5
    grid = ChebyshevGrid1D.from_N_L(N=N, L=L)
    x = grid.x
    for j in range(N + 1):
        assert jnp.isclose(x[j], -x[N - j], atol=1e-6), (
            f"Symmetry broken at j={j}: x[j]={x[j]}, -x[N-j]={-x[N - j]}"
        )


def test_chebyshev_grid_1d_domain_scaling():
    """Physical nodes span [-L, L]."""
    for L in [0.5, 1.0, 2.0, 3.14]:
        grid = ChebyshevGrid1D.from_N_L(N=12, L=L)
        x = grid.x
        assert jnp.isclose(x.max(), L, atol=1e-6)
        assert jnp.isclose(x.min(), -L, atol=1e-6)


def test_chebyshev_grid_1d_diff_matrix_shape():
    """D has shape (N+1, N+1) for GL."""
    N = 8
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    assert grid.D.shape == (N + 1, N + 1)


def test_chebyshev_grid_1d_diff_matrix_row_sum():
    """Rows of D sum to zero (derivative of constant = 0)."""
    grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0)
    row_sums = grid.D.sum(axis=1)
    assert jnp.allclose(row_sums, 0.0, atol=1e-10), (
        f"Row sums not zero: max abs = {jnp.abs(row_sums).max()}"
    )


def test_chebyshev_grid_1d_diff_matrix_row_sum_gauss():
    """Rows of D sum to zero for Gauss nodes too."""
    grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0, node_type="gauss")
    row_sums = grid.D.sum(axis=1)
    assert jnp.allclose(row_sums, 0.0, atol=1e-10)


def test_chebyshev_grid_1d_transform_roundtrip_gl():
    """Forward + inverse GL transform recovers the original field."""
    import jax

    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    key = jax.random.PRNGKey(42)
    u = jax.random.normal(key, (N + 1,))
    u_hat = grid.transform(u)
    u_rec = grid.transform(u_hat, inverse=True)
    assert jnp.allclose(u, u_rec, atol=1e-5), (
        f"Roundtrip max error = {jnp.abs(u - u_rec).max()}"
    )


def test_chebyshev_grid_1d_transform_roundtrip_gauss():
    """Forward + inverse Gauss transform recovers the original field."""
    import jax

    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0, node_type="gauss")
    key = jax.random.PRNGKey(7)
    u = jax.random.normal(key, (N,))
    u_hat = grid.transform(u)
    u_rec = grid.transform(u_hat, inverse=True)
    assert jnp.allclose(u, u_rec, atol=1e-5), (
        f"Roundtrip max error = {jnp.abs(u - u_rec).max()}"
    )


def test_chebyshev_grid_1d_factory_from_N_dx():
    """from_N_dx: L = N*dx/2."""
    N, dx = 16, 0.125  # L = 16 * 0.125 / 2 = 1.0
    grid = ChebyshevGrid1D.from_N_dx(N=N, dx=dx)
    assert jnp.isclose(grid.L, 1.0, atol=1e-10)
    assert grid.N == N


def test_chebyshev_grid_1d_check_consistency():
    """check_consistency returns True for valid grids."""
    grid = ChebyshevGrid1D.from_N_L(N=8, L=1.0)
    assert grid.check_consistency()


def test_chebyshev_grid_1d_dealias_filter_shape():
    """Dealias filter has correct shape."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0, dealias="2/3")
    mask = grid.dealias_filter()
    assert mask.shape == (N + 1,)


def test_chebyshev_grid_1d_dealias_filter_zeros_high_modes():
    """Dealias filter (2/3 rule) zeros out the top 1/3 of modes."""
    N = 12
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0, dealias="2/3")
    mask = grid.dealias_filter()
    cutoff = int(2 * N / 3)
    # Modes beyond cutoff should be zeroed
    assert jnp.all(mask[cutoff + 1 :] == 0.0)
    # Low modes should be kept
    assert jnp.all(mask[: cutoff + 1] == 1.0)


# ============================================================================
# ChebyshevGrid2D
# ============================================================================


def test_chebyshev_grid_2d_meshgrid_shapes():
    """X and Y meshgrids have shape (Ny+1, Nx+1) for GL."""
    Nx, Ny = 8, 6
    grid = ChebyshevGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=1.0, Ly=2.0)
    X, Y = grid.X
    assert X.shape == (Ny + 1, Nx + 1)
    assert Y.shape == (Ny + 1, Nx + 1)


def test_chebyshev_grid_2d_diff_matrix_shapes():
    """Dx and Dy have correct shapes."""
    Nx, Ny = 8, 6
    grid = ChebyshevGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=1.0, Ly=2.0)
    assert grid.Dx.shape == (Nx + 1, Nx + 1)
    assert grid.Dy.shape == (Ny + 1, Ny + 1)


def test_chebyshev_grid_2d_factory_methods():
    """from_N_L and from_N_dx produce consistent grids."""
    Nx, Ny, Lx, Ly = 8, 6, 1.0, 2.0
    grid1 = ChebyshevGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    dx = 2 * Lx / Nx
    dy = 2 * Ly / Ny
    grid2 = ChebyshevGrid2D.from_N_dx(Nx=Nx, Ny=Ny, dx=dx, dy=dy)
    assert jnp.isclose(grid1.Lx, grid2.Lx)
    assert jnp.isclose(grid1.Ly, grid2.Ly)


def test_chebyshev_grid_2d_transform_roundtrip():
    """2D forward + inverse transform recovers original field."""
    import jax

    Nx, Ny = 8, 8
    grid = ChebyshevGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0)
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (Ny + 1, Nx + 1))
    u_hat = grid.transform(u)
    u_rec = grid.transform(u_hat, inverse=True)
    assert jnp.allclose(u, u_rec, atol=1e-4), (
        f"2D roundtrip max error = {jnp.abs(u - u_rec).max()}"
    )


def test_chebyshev_grid_2d_check_consistency():
    """2D grid check_consistency returns True."""
    grid = ChebyshevGrid2D.from_N_L(Nx=8, Ny=8, Lx=1.0, Ly=1.0)
    assert grid.check_consistency()
