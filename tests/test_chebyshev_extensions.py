"""Tests for the Chebyshev extensions and performance work (epics #32 / #61).

Covers:
    * transform conventions (true Chebyshev coefficients on GL and Gauss nodes,
      float64 preservation on Gauss nodes)
    * coefficient-space calculus: derivative, antiderivative, definite integral
    * ``method="fft"`` derivative operators (1D, 2D, 3D)
    * 2D physics operators and integration
    * :class:`ChebyshevGrid3D` / :class:`ChebyshevDerivative3D`
    * diagonalisation-based Helmholtz solvers (agreement with the dense
      boundary-row solve, traced ``alpha``, traced-grid fallback)

All inputs are deterministic analytic fields, so tolerances are set by
spectral truncation + round-off, not by sampling noise.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spectraldiffx import (
    ChebyshevDerivative1D,
    ChebyshevDerivative2D,
    ChebyshevDerivative3D,
    ChebyshevGrid1D,
    ChebyshevGrid2D,
    ChebyshevGrid3D,
    ChebyshevHelmholtzSolver1D,
    ChebyshevHelmholtzSolver2D,
    ChebyshevPoissonSolver2D,
    chebyshev_antiderivative_coeffs,
    chebyshev_derivative_coeffs,
    chebyshev_integral_coeffs,
    clenshaw_curtis_weights,
)

NODE_TYPES = ["gauss-lobatto", "gauss"]


@pytest.fixture(params=NODE_TYPES)
def node_type(request) -> str:
    return request.param


# ============================================================================
# Transform conventions
# ============================================================================


def test_gl_transform_constant_is_unit_mode():
    """u ≡ 3 has coefficients [3, 0, …, 0] (a₀ is no longer doubled)."""
    grid = ChebyshevGrid1D.from_N_L(N=12, L=1.0)
    a = grid.transform(3.0 * jnp.ones(13))
    assert jnp.allclose(a, jnp.zeros(13).at[0].set(3.0), atol=1e-13)


def test_gl_and_gauss_transforms_agree_on_convention():
    """Both node types expand u = Σ aₖ Tₖ with the same coefficients."""
    u_fn = lambda x: 1.0 + 2.0 * x - 0.5 * (2 * x**2 - 1)  # T₀ + 2T₁ − ½T₂
    expected = jnp.array([1.0, 2.0, -0.5])
    for nt in NODE_TYPES:
        grid = ChebyshevGrid1D.from_N_L(N=10, L=1.0, node_type=nt)
        a = grid.transform(u_fn(grid.x))
        assert jnp.allclose(a[:3], expected, atol=1e-13), nt
        assert jnp.allclose(a[3:], 0.0, atol=1e-13), nt


def test_gauss_transform_preserves_float64():
    """Regression: the Gauss transform used to downcast float64 → float32."""
    grid = ChebyshevGrid1D.from_N_L(N=8, L=1.0, node_type="gauss")
    u = jnp.cos(3 * jnp.arccos(grid.x))
    a = grid.transform(u)
    assert a.dtype == jnp.float64
    assert jnp.abs(a[3] - 1.0) < 1e-13
    assert grid.transform(a, inverse=True).dtype == jnp.float64


def test_2d_transform_tensor_mode(node_type: str):
    """T₂(x)·T₃(y) maps to a single unit coefficient at [ky=3, kx=2]."""
    grid = ChebyshevGrid2D.from_N_L(Nx=8, Ny=9, Lx=1.0, Ly=1.0, node_type=node_type)
    X, Y = grid.X
    u = (2 * X**2 - 1) * (4 * Y**3 - 3 * Y)
    a = grid.transform(u)
    expected = jnp.zeros_like(a).at[3, 2].set(1.0)
    assert jnp.allclose(a, expected, atol=1e-12)


# ============================================================================
# Coefficient-space calculus
# ============================================================================


def test_derivative_coeffs_of_t3():
    """d/dx T₃ = 3T₀ + 6T₂."""
    a = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0])
    assert jnp.allclose(chebyshev_derivative_coeffs(a), jnp.array([3.0, 0, 6, 0, 0]))


def test_derivative_coeffs_order_zero_is_identity():
    a = jnp.arange(6.0)
    assert jnp.array_equal(chebyshev_derivative_coeffs(a, order=0), a)


def test_derivative_coeffs_negative_order_raises():
    with pytest.raises(ValueError, match="order"):
        chebyshev_derivative_coeffs(jnp.ones(4), order=-1)


def test_derivative_coeffs_batched_last_axis():
    """Leading axes are batch axes."""
    a = jnp.stack([jnp.array([0.0, 0, 0, 1, 0]), jnp.array([0.0, 1, 0, 0, 0])])
    d = chebyshev_derivative_coeffs(a, L=2.0)
    assert jnp.allclose(d[0], jnp.array([1.5, 0, 3, 0, 0]))
    assert jnp.allclose(d[1], jnp.array([0.5, 0, 0, 0, 0]))


def test_antiderivative_coeffs_of_constant():
    """∫_{−1}^{x} 1 ds = 1 + x = T₀ + T₁."""
    B = chebyshev_antiderivative_coeffs(jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(B, jnp.array([1.0, 1.0, 0.0]))


def test_integral_coeffs_equals_clenshaw_curtis():
    """On GL nodes the coefficient integral is Clenshaw–Curtis quadrature."""
    grid = ChebyshevGrid1D.from_N_L(N=20, L=1.5)
    f = jnp.exp(grid.x) * jnp.cos(3 * grid.x)
    cc = jnp.sum(clenshaw_curtis_weights(20, 1.5) * f)
    assert jnp.abs(chebyshev_integral_coeffs(grid.transform(f), L=1.5) - cc) < 1e-13


# ============================================================================
# 1D derivative operator: fft method, integrate, antiderivative
# ============================================================================


@pytest.mark.parametrize("order", [1, 2, 3])
def test_1d_fft_matches_matrix(node_type: str, order: int):
    grid = ChebyshevGrid1D.from_N_L(N=24, L=2.0, node_type=node_type)
    u = jnp.exp(jnp.sin(grid.x))
    d_mat = ChebyshevDerivative1D(grid=grid)(u, order=order)
    d_fft = ChebyshevDerivative1D(grid=grid, method="fft")(u, order=order)
    scale = jnp.max(jnp.abs(d_mat))
    # Round-off in either path grows like N^(2·order)·ε; ~3e-12 / 1e-10 / 3e-9.
    tol = 10.0 ** (-13 + 1.5 * order)
    assert jnp.max(jnp.abs(d_mat - d_fft)) / scale < tol


def test_1d_fft_derivative_accuracy(node_type: str):
    grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0, node_type=node_type)
    deriv = ChebyshevDerivative1D(grid=grid, method="fft")
    x = grid.x
    assert jnp.allclose(deriv(jnp.sin(3 * x)), 3 * jnp.cos(3 * x), atol=1e-10)


def test_1d_invalid_method_raises():
    grid = ChebyshevGrid1D.from_N_L(N=8, L=1.0)
    with pytest.raises(ValueError, match="method"):
        ChebyshevDerivative1D(grid=grid, method="spline")


def test_1d_fft_derivative_is_jittable():
    grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0)
    deriv = ChebyshevDerivative1D(grid=grid, method="fft")
    u = jnp.sin(grid.x)
    assert jnp.allclose(eqx.filter_jit(deriv)(u), deriv(u))


def test_1d_integrate(node_type: str):
    grid = ChebyshevGrid1D.from_N_L(N=24, L=1.0, node_type=node_type)
    deriv = ChebyshevDerivative1D(grid=grid)
    exact = jnp.e - 1.0 / jnp.e
    assert jnp.abs(deriv.integrate(jnp.exp(grid.x)) - exact) < 1e-13


def test_1d_antiderivative(node_type: str):
    """∫_{−L}^{x} cos(s) ds = sin(x) + sin(L)."""
    L = 2.0
    grid = ChebyshevGrid1D.from_N_L(N=32, L=L, node_type=node_type)
    U = ChebyshevDerivative1D(grid=grid).antiderivative(jnp.cos(grid.x))
    assert jnp.allclose(U, jnp.sin(grid.x) + jnp.sin(L), atol=1e-12)


# ============================================================================
# 2D derivative operator: fft method + physics operators
# ============================================================================


@pytest.fixture
def grid2d() -> ChebyshevGrid2D:
    return ChebyshevGrid2D.from_N_L(Nx=24, Ny=22, Lx=1.0, Ly=1.5)


@pytest.mark.slow
def test_2d_fft_matches_matrix(grid2d: ChebyshevGrid2D):
    X, Y = grid2d.X
    u = jnp.sin(2 * X) * jnp.exp(Y)
    mat = ChebyshevDerivative2D(grid=grid2d)
    fft = ChebyshevDerivative2D(grid=grid2d, method="fft")
    for a, b in zip(mat.gradient(u), fft.gradient(u), strict=True):
        assert jnp.allclose(a, b, atol=1e-10)
    assert jnp.allclose(mat.laplacian(u), fft.laplacian(u), atol=1e-8)


def test_2d_biharmonic(grid2d: ChebyshevGrid2D):
    """∇⁴[sin(x) sin(y)] = 4 sin(x) sin(y)."""
    X, Y = grid2d.X
    u = jnp.sin(X) * jnp.sin(Y)
    bih = ChebyshevDerivative2D(grid=grid2d).biharmonic(u)
    assert jnp.allclose(bih, 4 * u, atol=1e-6)


def test_2d_hyperviscosity_is_dissipative(grid2d: ChebyshevGrid2D):
    """(−1)ⁿ⁺¹ ν ∇²ⁿ on sin(x) sin(y) (eigenvalue −2 of ∇²) is −ν 2ⁿ u."""
    X, Y = grid2d.X
    u = jnp.sin(X) * jnp.sin(Y)
    deriv = ChebyshevDerivative2D(grid=grid2d)
    for n in (1, 2):
        hv = deriv.hyperviscosity(u, nu=0.1, order=n)
        assert jnp.allclose(hv, -0.1 * 2.0**n * u, atol=1e-6)


def test_2d_hyperviscosity_validates():
    grid = ChebyshevGrid2D.from_N_L(Nx=8, Ny=8, Lx=1.0, Ly=1.0)
    deriv = ChebyshevDerivative2D(grid=grid)
    u = jnp.zeros((9, 9))
    with pytest.raises(ValueError, match="order"):
        deriv.hyperviscosity(u, nu=1.0, order=0)
    with pytest.raises(ValueError, match="nu"):
        deriv.hyperviscosity(u, nu=-1.0)


def test_2d_velocity_from_streamfunction(grid2d: ChebyshevGrid2D):
    """u = −ψ_y, v = ψ_x is divergence-free with vorticity ∇²ψ."""
    X, Y = grid2d.X
    psi = jnp.sin(X) * jnp.cos(2 * Y)
    deriv = ChebyshevDerivative2D(grid=grid2d)
    u, v = deriv.velocity_from_streamfunction(psi)
    assert jnp.allclose(u, 2 * jnp.sin(X) * jnp.sin(2 * Y), atol=1e-10)
    assert jnp.allclose(v, jnp.cos(X) * jnp.cos(2 * Y), atol=1e-10)
    assert jnp.allclose(deriv.divergence(u, v), 0.0, atol=1e-9)
    assert jnp.allclose(deriv.curl(u, v), deriv.laplacian(psi), atol=1e-7)


def test_2d_jacobian(grid2d: ChebyshevGrid2D):
    """J(x², y) = 2x;  J(f, f) = 0."""
    X, Y = grid2d.X
    deriv = ChebyshevDerivative2D(grid=grid2d)
    assert jnp.allclose(deriv.jacobian(X**2, Y), 2 * X, atol=1e-11)
    f = jnp.sin(X * Y)
    assert jnp.allclose(deriv.jacobian(f, f), 0.0, atol=1e-12)


def test_2d_vector_laplacian(grid2d: ChebyshevGrid2D):
    X, Y = grid2d.X
    deriv = ChebyshevDerivative2D(grid=grid2d)
    lx, ly = deriv.vector_laplacian(X**3, X * Y**2)
    assert jnp.allclose(lx, 6 * X, atol=1e-9)
    assert jnp.allclose(ly, 2 * X, atol=1e-9)


def test_2d_integrate(grid2d: ChebyshevGrid2D):
    X, Y = grid2d.X
    exact = (jnp.e - 1 / jnp.e) * (jnp.exp(1.5) - jnp.exp(-1.5))
    got = ChebyshevDerivative2D(grid=grid2d).integrate(jnp.exp(X + Y))
    assert jnp.abs(got - exact) < 1e-12


# ============================================================================
# 3D grid + derivative operator
# ============================================================================


@pytest.fixture
def grid3d() -> ChebyshevGrid3D:
    return ChebyshevGrid3D.from_N_L(Nx=14, Ny=12, Nz=10, Lx=1.0, Ly=1.5, Lz=0.5)


def test_3d_grid_shapes(grid3d: ChebyshevGrid3D):
    Z, Y, X = grid3d.X
    assert Z.shape == Y.shape == X.shape == (11, 13, 15)
    assert jnp.allclose(X[0, 0, :], grid3d.x)
    assert jnp.allclose(Y[0, :, 0], grid3d.y)
    assert jnp.allclose(Z[:, 0, 0], grid3d.z)
    assert grid3d.dealias_filter().shape == (11, 13, 15)
    assert grid3d.check_consistency()


def test_3d_grid_check_consistency_raises():
    grid = ChebyshevGrid3D(Nx=4, Ny=4, Nz=4, Lz=-1.0)
    with pytest.raises(ValueError, match="Lz"):
        grid.check_consistency()


@pytest.mark.slow
def test_3d_transform_roundtrip(grid3d: ChebyshevGrid3D):
    Z, Y, X = grid3d.X
    u = jnp.exp(X) * jnp.cos(Y) * (1 + Z**2)
    a = grid3d.transform(u)
    assert jnp.allclose(grid3d.transform(a, inverse=True), u, atol=1e-13)


def test_3d_transform_tensor_mode():
    grid = ChebyshevGrid3D.from_N_L(Nx=6, Ny=6, Nz=6, Lx=1.0, Ly=1.0, Lz=1.0)
    Z, Y, X = grid.X
    a = grid.transform(X * (2 * Z**2 - 1))  # T₁(x) T₀(y) T₂(z)
    assert jnp.allclose(a, jnp.zeros_like(a).at[2, 0, 1].set(1.0), atol=1e-13)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["matrix", "fft"])
def test_3d_gradient_and_laplacian(grid3d: ChebyshevGrid3D, method: str):
    Z, Y, X = grid3d.X
    u = jnp.sin(X) * jnp.cos(Y) * jnp.exp(Z)
    deriv = ChebyshevDerivative3D(grid=grid3d, method=method)
    du_dz, du_dy, du_dx = deriv.gradient(u)
    assert jnp.allclose(du_dz, u, atol=1e-10)
    assert jnp.allclose(du_dy, -jnp.sin(X) * jnp.sin(Y) * jnp.exp(Z), atol=1e-9)
    assert jnp.allclose(du_dx, jnp.cos(X) * jnp.cos(Y) * jnp.exp(Z), atol=1e-10)
    assert jnp.allclose(deriv.laplacian(u), -u, atol=1e-7)


def test_3d_vector_identities(grid3d: ChebyshevGrid3D):
    """curl(grad φ) = 0 and div(curl V) = 0 hold to round-off."""
    Z, Y, X = grid3d.X
    deriv = ChebyshevDerivative3D(grid=grid3d)
    phi = jnp.sin(X * Y) * jnp.exp(Z)
    for w in deriv.curl(*deriv.gradient(phi)):
        assert jnp.allclose(w, 0.0, atol=1e-9)
    V = (X * Y, jnp.sin(Z), jnp.cos(X + Y))
    assert jnp.allclose(deriv.divergence(*deriv.curl(*V)), 0.0, atol=1e-9)


def test_3d_physics_operators(grid3d: ChebyshevGrid3D):
    Z, Y, X = grid3d.X
    deriv = ChebyshevDerivative3D(grid=grid3d)
    psi = X**2 * Y * Z
    u, v = deriv.velocity_from_streamfunction(psi)
    assert jnp.allclose(u, -(X**2) * Z, atol=1e-11)
    assert jnp.allclose(v, 2 * X * Y * Z, atol=1e-11)
    assert jnp.allclose(deriv.jacobian(X**2, Y), 2 * X, atol=1e-11)
    assert jnp.allclose(deriv.biharmonic(X**4 + Z**4), 48.0, atol=1e-6)
    lz, ly, lx = deriv.vector_laplacian(Z**2, Y**2, X**2)
    for comp in (lz, ly, lx):
        assert jnp.allclose(comp, 2.0, atol=1e-9)
    assert jnp.allclose(deriv.hyperviscosity(X**2, nu=0.5, order=1), 1.0, atol=1e-9)


def test_3d_integrate(grid3d: ChebyshevGrid3D):
    Z, Y, X = grid3d.X
    got = ChebyshevDerivative3D(grid=grid3d).integrate(X**2 + Y * Z + 1.0)
    vol = 2.0 * 3.0 * 1.0
    exact = (2.0 / 3.0) * 3.0 * 1.0 + vol  # ∫x² over box + ∫1; ∫yz = 0
    assert jnp.abs(got - exact) < 1e-12


# ============================================================================
# Diagonalisation-based Helmholtz solvers
# ============================================================================


@pytest.mark.parametrize("bc_type", ["dirichlet", "neumann"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 10.0])
def test_1d_solver_matches_dense(bc_type: str, alpha: float):
    """The O(N²) diagonalised solve reproduces the dense boundary-row solve."""
    grid = ChebyshevGrid1D.from_N_L(N=48, L=1.3)
    solver = ChebyshevHelmholtzSolver1D(grid)
    x = grid.x
    # For pure Neumann Poisson use a compatible RHS (zero mean flux balance).
    f = jnp.cos(jnp.pi * x / 1.3) if alpha == 0.0 else jnp.exp(x) * jnp.sin(3 * x)
    bcs = (0.0, 0.0) if (bc_type == "neumann" and alpha == 0.0) else (0.3, -0.7)
    u = solver.solve(f, alpha, *bcs, bc_type=bc_type)
    u_dense = solver._solve_dense(f, alpha, *bcs, bc_type)
    assert jnp.allclose(u, u_dense, atol=1e-11)


def test_1d_solver_jit_and_grad_in_alpha():
    """alpha may be traced: jit compiles once, grad is finite and correct."""
    grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    solver = ChebyshevHelmholtzSolver1D(grid)
    f = jnp.sin(jnp.pi * grid.x)

    solve = jax.jit(lambda a: solver.solve(f, alpha=a))
    for a in (0.0, 1.0, 4.0):
        assert jnp.allclose(solve(a), solver.solve(f, alpha=a), atol=1e-13)

    # u = −sin(πx) / (π² + α)  ⇒  ∂u/∂α at x_mid = sin(πx)/(π² + α)²
    loss = lambda a: solver.solve(f, alpha=a)[5]
    expected = jnp.sin(jnp.pi * grid.x[5]) / (jnp.pi**2 + 2.0) ** 2
    assert jnp.abs(jax.grad(loss)(2.0) - expected) < 1e-9


def test_1d_solver_negative_alpha_raises():
    solver = ChebyshevHelmholtzSolver1D(ChebyshevGrid1D.from_N_L(N=8, L=1.0))
    with pytest.raises(ValueError, match="alpha"):
        solver.solve(jnp.zeros(9), alpha=-1.0)


def test_1d_solver_traced_grid_falls_back_to_dense():
    """Building the solver from a traced grid still gives the right answer."""
    grid = ChebyshevGrid1D.from_N_L(N=24, L=1.0)
    f = -(jnp.pi**2) * jnp.sin(jnp.pi * grid.x)

    @eqx.filter_jit
    def build_and_solve(g, rhs):
        return ChebyshevHelmholtzSolver1D(g).solve(rhs)

    assert jnp.allclose(build_and_solve(grid, f), jnp.sin(jnp.pi * grid.x), atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("alpha", [0.0, 3.0])
def test_2d_solver_matches_dense(alpha: float):
    grid = ChebyshevGrid2D.from_N_L(Nx=12, Ny=10, Lx=1.0, Ly=0.8)
    solver = ChebyshevHelmholtzSolver2D(grid)
    X, Y = grid.X
    f = jnp.exp(X) * jnp.cos(2 * Y)
    kw = dict(bc_top=jnp.cos(grid.x), bc_bottom=0.5, bc_left=grid.y**2, bc_right=-1.0)
    u = solver.solve(f, alpha, **kw)
    bc = u.at[1:-1, 1:-1].set(0.0)  # boundary values as assembled by solve
    u_dense = solver._solve_dense(f, alpha, bc)
    assert jnp.allclose(u, u_dense, atol=1e-11)


def test_2d_solver_inhomogeneous_manufactured():
    """u = eˣ sin(y) (∇²u = 0) with Dirichlet data taken from u itself."""
    grid = ChebyshevGrid2D.from_N_L(Nx=20, Ny=20, Lx=1.0, Ly=1.0)
    X, Y = grid.X
    ue = jnp.exp(X) * jnp.sin(Y)
    u = ChebyshevPoissonSolver2D(grid).solve(
        jnp.zeros_like(ue),
        bc_top=ue[0, :],
        bc_bottom=ue[-1, :],
        bc_left=ue[:, -1],
        bc_right=ue[:, 0],
    )
    assert jnp.allclose(u, ue, atol=1e-12)


def test_2d_solver_jit_in_alpha():
    grid = ChebyshevGrid2D.from_N_L(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
    solver = ChebyshevHelmholtzSolver2D(grid)
    X, Y = grid.X
    f = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    solve = jax.jit(lambda a: solver.solve(f, alpha=a))
    assert jnp.allclose(solve(2.0), solver.solve(f, alpha=2.0), atol=1e-13)
