"""
Simulation-level sanity tests for spectraldiffx scripts.

These tests verify that:
1. Script RHS functions produce finite (non-NaN) outputs.
2. Key operator identities hold (div(curl)=0, curl(grad)=0, etc.).
3. Solvers correctly invert the Laplacian/Helmholtz operators.
4. Edge cases (zero fields, constant fields) behave correctly.
5. The fixed QG beta term (-beta*v) is periodic, unlike beta*y.
"""

import jax.numpy as jnp
import pytest

from spectraldiffx._src.grid import FourierGrid1D, FourierGrid2D
from spectraldiffx._src.operators import SpectralDerivative2D
from spectraldiffx._src.solvers import SpectralHelmholtzSolver2D

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid2d():
    return FourierGrid2D.from_N_L(32, 32, 2 * jnp.pi, 2 * jnp.pi)


@pytest.fixture
def grid2d_nodealias():
    return FourierGrid2D.from_N_L(32, 32, 2 * jnp.pi, 2 * jnp.pi, dealias=None)


@pytest.fixture
def grid1d():
    return FourierGrid1D.from_N_L(64, 100.0, dealias="2/3")


# ---------------------------------------------------------------------------
# KdV RHS sanity
# ---------------------------------------------------------------------------


def test_kdv_rhs_no_nan(grid1d):
    """One KdV RHS evaluation produces finite output."""
    x = grid1d.x - grid1d.L / 2.0
    c1, x1 = 2.0, grid1d.L / 4.0
    u = 2 * c1 / jnp.cosh(jnp.sqrt(c1) * (x - x1)) ** 2

    u_hat = grid1d.transform(u)
    k = grid1d.k
    k_dealias = grid1d.k_dealias

    du_dx_hat = 1j * k_dealias * u_hat
    du_dx = grid1d.transform(du_dx_hat, inverse=True).real
    advection_phys = -6.0 * u * du_dx
    advection_hat = grid1d.transform(advection_phys)

    d3u_dx3_hat = (1j * k) ** 3 * u_hat
    dispersion_hat = -d3u_dx3_hat

    total_hat = (advection_hat + dispersion_hat) * grid1d.dealias_filter()
    du_dt = grid1d.transform(total_hat, inverse=True).real

    assert jnp.all(jnp.isfinite(du_dt)), "KdV RHS contains NaN or Inf"


# ---------------------------------------------------------------------------
# Navier-Stokes 2D RHS sanity
# ---------------------------------------------------------------------------


def test_ns2d_explicit_rhs_no_nan(grid2d):
    """NS2D explicit term (advection) produces finite output."""
    deriv = SpectralDerivative2D(grid2d)
    solver = SpectralHelmholtzSolver2D(grid2d)

    X, Y = grid2d.X
    omega = jnp.sin(X) * jnp.cos(Y)

    psi = solver.solve(omega, alpha=0.0)
    dpsi_dx, dpsi_dy = deriv.gradient(psi)
    u, v = -dpsi_dy, dpsi_dx

    advection = -deriv.advection_scalar(u, v, omega)
    assert jnp.all(jnp.isfinite(advection)), "NS2D explicit RHS contains NaN or Inf"


def test_ns2d_implicit_rhs_no_nan(grid2d):
    """NS2D implicit term (spectral hyperviscosity) produces finite output."""
    nu = 1e-6
    nv = 2
    K2 = grid2d.K2
    X, Y = grid2d.X
    omega = jnp.sin(X) * jnp.cos(Y)

    omega_hat = grid2d.transform(omega)
    lap_hat = ((-K2) ** nv) * omega_hat
    diffusion = grid2d.transform(lap_hat, inverse=True).real
    result = nu * diffusion

    assert jnp.all(jnp.isfinite(result)), "NS2D implicit RHS contains NaN or Inf"


def test_ns2d_pipeline_finite(grid2d):
    """Run 10 explicit Euler steps of NS2D and verify no NaN."""
    import jax.random as jrandom

    deriv = SpectralDerivative2D(grid2d)
    solver = SpectralHelmholtzSolver2D(grid2d)
    nu = 1e-4
    nv = 2
    dt = 1e-3
    K2 = grid2d.K2

    key = jrandom.PRNGKey(42)
    omega = jrandom.normal(key, shape=(grid2d.Ny, grid2d.Nx))
    # Band-pass filter
    omega_hat = grid2d.transform(omega)
    k_mag = jnp.sqrt(K2)
    mask = (k_mag > 6) & (k_mag < 10)
    omega_hat = jnp.where(mask, omega_hat, 0.0)
    omega = grid2d.transform(omega_hat, inverse=True).real
    omega = omega / jnp.std(omega)

    for _ in range(10):
        # Explicit: advection
        psi = solver.solve(omega, alpha=0.0)
        dpsi_dx, dpsi_dy = deriv.gradient(psi)
        u, v = -dpsi_dy, dpsi_dx
        explicit = -deriv.advection_scalar(u, v, omega)

        # Implicit: spectral hyperviscosity
        omega_hat = grid2d.transform(omega)
        lap_hat = ((-K2) ** nv) * omega_hat
        implicit = grid2d.transform(lap_hat, inverse=True).real * nu

        omega = omega + dt * (explicit + implicit)

    assert jnp.all(jnp.isfinite(omega)), "NS2D pipeline produced NaN after 10 steps"


# ---------------------------------------------------------------------------
# QG Model sanity
# ---------------------------------------------------------------------------


def test_qg_helmholtz_inversion_no_nan(grid2d):
    """QG PV inversion (without beta*y subtraction) produces finite output."""
    import jax.random as jrandom

    solver = SpectralHelmholtzSolver2D(grid2d)
    deriv = SpectralDerivative2D(grid2d)

    key = jrandom.PRNGKey(0)
    q = jrandom.normal(key, shape=(grid2d.Ny, grid2d.Nx)) * 0.1
    # Band-pass
    q_hat = grid2d.transform(q)
    k_mag = jnp.sqrt(grid2d.K2)
    k_def = 2 * jnp.pi / (grid2d.Lx * 0.1)
    mask = (k_mag > k_def * 0.8) & (k_mag < k_def * 1.2)
    q_hat = jnp.where(mask, q_hat, 0.0)
    q = grid2d.transform(q_hat, inverse=True).real

    r_def_inv_sq = 1.0 / (0.1**2)
    psi = solver.solve(q, alpha=r_def_inv_sq)
    dpsi_dx, dpsi_dy = deriv.gradient(psi)
    u, v = -dpsi_dy, dpsi_dx

    assert jnp.all(jnp.isfinite(psi)), "QG psi contains NaN"
    assert jnp.all(jnp.isfinite(u)), "QG u contains NaN"
    assert jnp.all(jnp.isfinite(v)), "QG v contains NaN"


def test_qg_beta_term_periodic(grid2d):
    """Show beta*v is spectrally periodic; beta*y has a non-zero periodicity gap.

    The bug: subtracting beta*y (a linear ramp) from the Helmholtz RHS introduces
    Gibbs ringing because the periodic extension of beta*y has a jump of beta*Ly
    at the domain boundary. The fix uses -beta*v instead, which is periodic.
    """
    beta = 10.0
    _, Y = grid2d.X

    # beta*y is NOT periodic on the domain: its periodic extension has a jump
    # from beta*(Ly-dy) back to 0. The jump magnitude is beta*Ly - 0 = beta*Ly.
    period_gap = float(beta * grid2d.Ly)
    assert period_gap > 0.0, "beta*Ly should be positive (non-zero periodicity gap)"

    # beta*v IS periodic: v is derived from Fourier modes and satisfies v(y+Ly)=v(y)
    solver = SpectralHelmholtzSolver2D(grid2d)
    deriv = SpectralDerivative2D(grid2d)
    X, Y2 = grid2d.X
    # Use a field with only a few Fourier modes
    q = jnp.sin(2 * X) * jnp.cos(2 * Y2)
    psi = solver.solve(q, alpha=0.0)
    dpsi_dx, _ = deriv.gradient(psi)
    v = dpsi_dx

    # beta*v should be finite (no blow-up like a ramp)
    assert jnp.all(jnp.isfinite(beta * v)), "beta*v should be finite everywhere"
    # beta*v is bounded (no linear ramp growth); its max is < beta*Ly
    assert float(jnp.max(jnp.abs(beta * v))) < float(beta * grid2d.Ly), (
        "beta*v should be bounded (unlike the linear ramp beta*y)"
    )
    # The y-direction FFT of beta*y has non-zero Nyquist content (spectral leakage from jump)
    beta_y = beta * Y
    beta_y_hat_ky = jnp.fft.fft(beta_y[:, 0])  # 1D FFT in y-direction
    nyquist_idx = grid2d.Ny // 2
    nyquist_energy_y = float(jnp.abs(beta_y_hat_ky[nyquist_idx]))
    assert nyquist_energy_y > 0.01, (
        "beta*y should have significant Nyquist-mode energy (Gibbs ringing)"
    )
    # beta*v (from wavenumber-2 input) should have zero Nyquist energy
    beta_v_hat_ky = jnp.fft.fft((beta * v)[:, 0])
    nyquist_energy_v = float(jnp.abs(beta_v_hat_ky[nyquist_idx]))
    assert nyquist_energy_v < 1e-10, (
        "beta*v from low-wavenumber input should have no Nyquist energy"
    )


# ---------------------------------------------------------------------------
# Operator identity tests
# ---------------------------------------------------------------------------


def test_div_curl_zero(grid2d_nodealias):
    """Divergence of a curl field should be zero."""
    deriv = SpectralDerivative2D(grid2d_nodealias)
    X, Y = grid2d_nodealias.X
    # scalar potential -> divergence-free field via curl
    phi = jnp.sin(X) * jnp.cos(2 * Y)
    # 2D "curl" of scalar: (-d(phi)/dy, d(phi)/dx) is divergence-free
    dphi_dx, dphi_dy = deriv.gradient(phi)
    vx, vy = -dphi_dy, dphi_dx  # This is the rotated gradient
    div = deriv.divergence(vx, vy)
    assert jnp.allclose(div, 0.0, atol=1e-5), "div(curl(phi)) != 0"


def test_curl_gradient_zero(grid2d_nodealias):
    """2D scalar curl of a gradient field should be zero."""
    deriv = SpectralDerivative2D(grid2d_nodealias)
    X, Y = grid2d_nodealias.X
    phi = jnp.sin(X) * jnp.cos(2 * Y)
    dphi_dx, dphi_dy = deriv.gradient(phi)
    # curl of gradient = d(dphi_dy)/dx - d(dphi_dx)/dy = 0 (by symmetry of mixed partials)
    curl = deriv.curl(dphi_dx, dphi_dy)
    assert jnp.allclose(curl, 0.0, atol=1e-4), "curl(grad(phi)) != 0"


def test_laplacian_equals_div_grad(grid2d_nodealias):
    """Laplacian should equal divergence of gradient."""
    deriv = SpectralDerivative2D(grid2d_nodealias)
    X, Y = grid2d_nodealias.X
    u = jnp.sin(2 * X) * jnp.sin(3 * Y)
    lap = deriv.laplacian(u)
    gu_dx, gu_dy = deriv.gradient(u)
    div_grad = deriv.divergence(gu_dx, gu_dy)
    assert jnp.allclose(lap, div_grad, atol=1e-4), "laplacian(u) != div(grad(u))"


def test_poisson_inverts_laplacian(grid2d_nodealias):
    """Solving (laplacian)psi = f should recover psi from f=laplacian(psi_exact)."""
    deriv = SpectralDerivative2D(grid2d_nodealias)
    solver = SpectralHelmholtzSolver2D(grid2d_nodealias)
    X, Y = grid2d_nodealias.X
    psi_exact = jnp.sin(X) * jnp.sin(2 * Y)
    f = deriv.laplacian(psi_exact)
    psi_solved = solver.solve(f, alpha=0.0)
    assert jnp.allclose(psi_solved, psi_exact, atol=1e-4), (
        "Poisson solver inversion failed"
    )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_zero_field_operators(grid2d):
    """All operators on the zero field return zero."""
    deriv = SpectralDerivative2D(grid2d)
    u = jnp.zeros((grid2d.Ny, grid2d.Nx))
    gx, gy = deriv.gradient(u)
    lap = deriv.laplacian(u)
    assert jnp.allclose(gx, 0.0), "gradient of zero != 0"
    assert jnp.allclose(gy, 0.0), "gradient of zero != 0"
    assert jnp.allclose(lap, 0.0), "laplacian of zero != 0"


def test_constant_field_operators(grid2d):
    """Gradient and Laplacian of constant field are zero."""
    deriv = SpectralDerivative2D(grid2d)
    u = jnp.ones((grid2d.Ny, grid2d.Nx)) * 3.14
    gx, gy = deriv.gradient(u)
    lap = deriv.laplacian(u)
    assert jnp.allclose(gx, 0.0, atol=1e-10), "gradient of constant != 0"
    assert jnp.allclose(gy, 0.0, atol=1e-10), "gradient of constant != 0"
    assert jnp.allclose(lap, 0.0, atol=1e-10), "laplacian of constant != 0"


def test_single_mode_roundtrip(grid2d):
    """A single Fourier mode is preserved through FFT roundtrip."""
    X, Y = grid2d.X
    u = jnp.sin(2 * X) * jnp.cos(3 * Y)
    u_hat = grid2d.transform(u)
    u_reconstructed = grid2d.transform(u_hat, inverse=True).real
    assert jnp.allclose(u, u_reconstructed, atol=1e-10), "FFT roundtrip failed"


def test_dealias_filter_shape(grid2d):
    """Dealiasing filter has the correct shape."""
    mask = grid2d.dealias_filter()
    assert mask.shape == (grid2d.Ny, grid2d.Nx)


def test_dealias_filter_no_dealias():
    """No-dealias mode returns all-ones filter."""
    grid = FourierGrid2D.from_N_L(16, 16, 2 * jnp.pi, 2 * jnp.pi, dealias=None)
    mask = grid.dealias_filter()
    assert jnp.all(mask == 1.0), "No-dealias filter should be all ones"


def test_hyperviscosity_implicit_no_dealias_no_nan(grid2d):
    """Spectral hyperviscosity (nv=4) without dealiasing produces finite output."""
    import jax.random as jrandom

    nv = 4
    K2 = grid2d.K2
    key = jrandom.PRNGKey(7)
    omega = jrandom.normal(key, shape=(grid2d.Ny, grid2d.Nx))

    omega_hat = grid2d.transform(omega)
    lap_hat = ((-K2) ** nv) * omega_hat
    result = grid2d.transform(lap_hat, inverse=True).real

    assert jnp.all(jnp.isfinite(result)), "Spectral hyperviscosity produced NaN"


def test_non_square_grid():
    """Non-square 2D grid (Nx != Ny) works correctly."""
    # Domain [0, 2*pi] x [0, pi] with Nx=32, Ny=16
    grid = FourierGrid2D.from_N_L(32, 16, 2 * jnp.pi, jnp.pi)
    grid.check_consistency()
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    assert X.shape == (16, 32)

    # Use valid Fourier modes for this domain.
    # For Lx=2*pi: kx=m (integers), so sin(x) has kx=1 (valid).
    # For Ly=pi: ky=2*pi*n/pi = 2n, so cos(2*y) has ky=2 (n=1, valid).
    u = jnp.sin(X) * jnp.cos(2 * Y)
    gx, gy = deriv.gradient(u)
    assert jnp.allclose(gx, jnp.cos(X) * jnp.cos(2 * Y), atol=1e-5)
    assert jnp.allclose(gy, -2 * jnp.sin(X) * jnp.sin(2 * Y), atol=1e-5)
