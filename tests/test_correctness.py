"""
Correctness and stability tests for spectraldiffx.

These tests fill the gaps identified from the following reference implementations:
- FourierFlows.jl / GeophysicalFlows.jl: Parseval's theorem, energy/enstrophy conservation
- neuralgcm/dinosaur: spectral convergence rates
- karlotness/pyqg-jax: QG dynamics, PV conservation

Tests cover:
1. Parseval's theorem: energy is preserved through the FFT (1D, 2D)
2. Spectral exponential convergence: derivative error decreases exponentially with N
3. Multi-wavenumber derivative accuracy: d/dx(sin(kx)) = k*cos(kx) for k=1..10
4. Dealiasing effectiveness: 2/3 rule removes aliased modes from nonlinear products
5. Spectral-input consistency: spectral=True and spectral=False give identical results
6. Enstrophy dissipation in 2D NS: enstrophy decreases monotonically with viscosity
7. KdV L2 conservation: total L2 norm is preserved under KdV dynamics
8. Chebyshev spectral convergence: derivative error decreases with polynomial degree N
9. Spherical Legendre orthogonality / Parseval: transform preserves energy under quadrature
10. QG mean-PV conservation: Jacobian advection preserves total PV integral
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx._src.chebyshev.grid import ChebyshevGrid1D
from spectraldiffx._src.chebyshev.operators import ChebyshevDerivative1D
from spectraldiffx._src.grid import FourierGrid1D, FourierGrid2D
from spectraldiffx._src.operators import SpectralDerivative1D, SpectralDerivative2D
from spectraldiffx._src.solvers import SpectralHelmholtzSolver2D
from spectraldiffx._src.spherical.grid import SphericalGrid1D
from spectraldiffx._src.spherical.operators import SphericalDerivative1D

# ---------------------------------------------------------------------------
# 1. Parseval's Theorem
# ---------------------------------------------------------------------------


def test_parseval_1d():
    """
    Parseval's theorem for the 1D FFT.

    For a real signal u of length N:
        sum_n |u[n]|^2 = (1/N) * sum_k |u_hat[k]|^2

    This ensures that the FFT is an energy-preserving (unitary up to scaling)
    transform and is the foundation for spectral energy diagnostics.
    """
    N = 64
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias=None)
    u = jnp.sin(3 * grid.x) + 0.5 * jnp.cos(5 * grid.x) + 0.1 * jnp.sin(10 * grid.x)

    u_hat = grid.transform(u)
    energy_physical = jnp.sum(jnp.abs(u) ** 2)
    energy_spectral = jnp.sum(jnp.abs(u_hat) ** 2) / N

    assert jnp.isclose(energy_physical, energy_spectral, rtol=1e-10), (
        f"Parseval 1D failed: physical={float(energy_physical):.6e}, "
        f"spectral={float(energy_spectral):.6e}"
    )


def test_parseval_2d():
    """
    Parseval's theorem for the 2D FFT.

    For a 2D field u of shape (Ny, Nx):
        sum_{m,n} |u[m,n]|^2 = (1/(Nx*Ny)) * sum_{kx,ky} |u_hat[kx,ky]|^2
    """
    Nx, Ny = 32, 32
    grid = FourierGrid2D.from_N_L(Nx, Ny, 2 * jnp.pi, 2 * jnp.pi, dealias=None)
    X, Y = grid.X
    u = jnp.sin(2 * X) * jnp.cos(3 * Y) + 0.5 * jnp.sin(5 * X)

    u_hat = grid.transform(u)
    energy_physical = jnp.sum(jnp.abs(u) ** 2)
    energy_spectral = jnp.sum(jnp.abs(u_hat) ** 2) / (Nx * Ny)

    assert jnp.isclose(energy_physical, energy_spectral, rtol=1e-10), (
        f"Parseval 2D failed: physical={float(energy_physical):.6e}, "
        f"spectral={float(energy_spectral):.6e}"
    )


def test_parseval_1d_random():
    """Parseval's theorem holds for a band-limited random signal."""
    N = 128
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias=None)
    # Band-limited: keep only modes 1..10
    u_hat = jnp.zeros(N, dtype=jnp.complex128)
    for k in range(1, 11):
        u_hat = u_hat.at[k].set(1.0 / k + 0.5j / k)
        u_hat = u_hat.at[N - k].set(1.0 / k - 0.5j / k)  # conjugate symmetry
    u = grid.transform(u_hat, inverse=True).real

    u_hat2 = grid.transform(u)
    energy_physical = jnp.sum(jnp.abs(u) ** 2)
    energy_spectral = jnp.sum(jnp.abs(u_hat2) ** 2) / N

    assert jnp.isclose(energy_physical, energy_spectral, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. Spectral Exponential Convergence
# ---------------------------------------------------------------------------


def test_spectral_convergence_1d():
    """
    Spectral methods converge exponentially for smooth periodic functions.

    For u = exp(sin(x)), the error in du/dx decreases exponentially
    as the grid resolution N increases — far faster than any algebraic rate.

    Using N=8 (only 4 positive modes) vs N=32 (16 modes), the improvement is
    many orders of magnitude (>1e6x), demonstrating spectral accuracy.
    """
    L = 2 * jnp.pi

    def exact_derivative(x):
        return jnp.cos(x) * jnp.exp(jnp.sin(x))

    errors = []
    for N in [8, 32]:
        grid = FourierGrid1D.from_N_L(N, L, dealias=None)
        deriv = SpectralDerivative1D(grid)
        u = jnp.exp(jnp.sin(grid.x))
        du_dx = deriv.gradient(u)
        exact = exact_derivative(grid.x)
        err = float(jnp.max(jnp.abs(du_dx - exact)))
        errors.append(err)

    # Error should drop by many orders of magnitude (exponential convergence)
    assert errors[0] > 1e-3, (
        f"N=8 should have non-trivial error for exp(sin(x)), got {errors[0]:.2e}"
    )
    assert errors[1] < 1e-10, (
        f"N=32 should have near machine-precision error, got {errors[1]:.2e}"
    )
    assert errors[1] < errors[0] * 1e-6, (
        f"Going from N=8 to N=32 should reduce error by >1e6x, got only "
        f"{errors[0]/errors[1]:.1f}x reduction"
    )


def test_spectral_convergence_2d():
    """
    Spectral convergence in 2D: error in ∂u/∂x for u = exp(sin(x)+sin(y)).

    Verify that N=32 gives essentially machine-precision accuracy (< 1e-8).
    """
    L = 2 * jnp.pi
    N = 32
    grid = FourierGrid2D.from_N_L(N, N, L, L, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.exp(jnp.sin(X) + jnp.sin(Y))
    du_dx, _ = deriv.gradient(u)
    exact = jnp.cos(X) * jnp.exp(jnp.sin(X) + jnp.sin(Y))
    err = float(jnp.max(jnp.abs(du_dx - exact)))
    assert err < 1e-8, f"2D spectral convergence: expected < 1e-8, got {err:.2e}"


# ---------------------------------------------------------------------------
# 3. Multi-Wavenumber Derivative Accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 10])
def test_multiwavenumber_gradient_1d(k):
    """
    d/dx[sin(kx)] = k*cos(kx) must hold for multiple wavenumbers k.

    Spectral methods should handle all resolved wavenumbers with equal accuracy.
    """
    N = 64
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(k * grid.x)
    du_dx = deriv.gradient(u)
    expected = k * jnp.cos(k * grid.x)
    assert jnp.allclose(du_dx, expected, atol=1e-10), (
        f"k={k}: max error = {float(jnp.max(jnp.abs(du_dx - expected))):.2e}"
    )


@pytest.mark.parametrize("kx,ky", [(1, 1), (2, 3), (4, 2), (3, 5)])
def test_multiwavenumber_gradient_2d(kx, ky):
    """
    ∂/∂x[sin(kx*x)*cos(ky*y)] = kx*cos(kx*x)*cos(ky*y) for multiple wavenumber pairs.
    """
    N = 64
    grid = FourierGrid2D.from_N_L(N, N, 2 * jnp.pi, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(kx * X) * jnp.cos(ky * Y)
    du_dx, du_dy = deriv.gradient(u)
    assert jnp.allclose(du_dx, kx * jnp.cos(kx * X) * jnp.cos(ky * Y), atol=1e-10), (
        f"kx={kx},ky={ky}: ∂u/∂x error"
    )
    assert jnp.allclose(du_dy, -ky * jnp.sin(kx * X) * jnp.sin(ky * Y), atol=1e-10), (
        f"kx={kx},ky={ky}: ∂u/∂y error"
    )


def test_higher_order_derivative_1d():
    """
    d^4/dx^4[sin(3x)] = 81*sin(3x) (since (ik)^4 = k^4 for real k).
    Verify fourth-order spectral differentiation.
    """
    N = 64
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(3 * grid.x)
    d4u = deriv(u, order=4)
    expected = 81.0 * jnp.sin(3 * grid.x)  # (3)^4 = 81
    assert jnp.allclose(d4u, expected, atol=1e-8), (
        f"4th-order derivative: max error = {float(jnp.max(jnp.abs(d4u - expected))):.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Dealiasing Effectiveness
# ---------------------------------------------------------------------------


def test_dealias_zeros_above_cutoff_modes():
    """
    The 2/3 dealiasing filter should zero out modes above k_cutoff = k_max * 2/3.

    For N=32, k_max=16, k_cutoff ≈ 10.67. A mode at k=11 should be removed.
    A mode at k=5 (below cutoff) should be preserved.
    """
    N = 32
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias="2/3")
    deriv = SpectralDerivative1D(grid)

    # Mode at k=5 (well below cutoff ~10.67): should be preserved
    u_safe = jnp.sin(5 * grid.x)
    u_safe_dealiased = deriv.apply_dealias(u_safe)
    assert jnp.allclose(u_safe_dealiased, u_safe, atol=1e-10), (
        "Mode k=5 should be preserved by 2/3 dealiasing"
    )

    # Mode at k=11 (above cutoff ~10.67): should be zeroed
    u_above = jnp.sin(11 * grid.x)
    u_above_dealiased = deriv.apply_dealias(u_above)
    assert jnp.allclose(u_above_dealiased, 0.0, atol=1e-10), (
        "Mode k=11 should be zeroed by 2/3 dealiasing (above cutoff)"
    )


def test_dealias_removes_aliased_energy_in_gradient():
    """
    The 2/3 rule removes aliased energy that appears in nonlinear products.

    For N=32, u=sin(9x): u^2 = (1-cos(18x))/2.
    The k=18 mode aliases to k=-14 (index 18 in FFT → physical frequency -14).
    |k|=14 > k_cutoff ≈ 10.67, so it IS above the dealiasing threshold.

    Without dealiasing: d(u^2)/dx has spurious energy at |k|=14.
    With dealiasing: the mode at |k|=14 is zeroed (aliased mode removed).
    """
    N = 32
    # Without dealiasing
    grid_nd = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias=None)
    deriv_nd = SpectralDerivative1D(grid_nd)

    # With dealiasing
    grid_d = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias="2/3")
    deriv_d = SpectralDerivative1D(grid_d)

    # k=9 is just within the cutoff (~10.67); u^2 produces mode k=18 that aliases to k=-14
    u = jnp.sin(9 * grid_nd.x)
    u_sq = u * u

    # Compute d(u^2)/dx via spectral derivative
    dusq_nd = deriv_nd.gradient(u_sq)
    dusq_d = deriv_d.gradient(u_sq)

    # In spectral space: check energy at k=14 (the aliased mode index)
    dusq_nd_hat = grid_nd.transform(dusq_nd)
    dusq_d_hat = grid_d.transform(dusq_d)

    # Without dealiasing: significant spurious energy at k=14 (alias of k=18)
    energy_k14_no_dealias = float(jnp.abs(dusq_nd_hat[14]))
    # With dealiasing: k=14 is above cutoff, so it is zeroed
    energy_k14_dealias = float(jnp.abs(dusq_d_hat[14]))

    assert energy_k14_no_dealias > 1.0, (
        f"Without dealiasing, aliased mode k=14 should have significant energy, "
        f"got {energy_k14_no_dealias:.4f}"
    )
    assert energy_k14_dealias < 1e-10, (
        f"With dealiasing, aliased mode k=14 should be ~0, "
        f"got {energy_k14_dealias:.2e}"
    )


def test_dealias_low_modes_accurate():
    """
    With dealiasing, derivatives of low-k fields remain accurate.

    For modes well below the 2/3 cutoff, the dealiasing filter should not
    affect accuracy — derivatives should remain spectrally exact.
    """
    N = 32
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias="2/3")
    deriv = SpectralDerivative1D(grid)
    # k=3 is well below cutoff ~10.67
    u = jnp.sin(3 * grid.x) + jnp.cos(5 * grid.x)
    du_dx = deriv.gradient(u)
    exact = 3 * jnp.cos(3 * grid.x) - 5 * jnp.sin(5 * grid.x)
    assert jnp.allclose(du_dx, exact, atol=1e-10), (
        f"Low-k derivative inaccurate with dealiasing: "
        f"max error = {float(jnp.max(jnp.abs(du_dx - exact))):.2e}"
    )


# ---------------------------------------------------------------------------
# 5. Spectral-Input Consistency
# ---------------------------------------------------------------------------


def test_spectral_input_gradient_1d():
    """
    gradient(u, spectral=True) and gradient(u, spectral=False) must agree.

    The spectral=True path skips the forward FFT and uses the pre-computed
    Fourier coefficients directly. Both paths should give the same result.
    """
    N = 64
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(4 * grid.x) + jnp.cos(7 * grid.x)

    du_phys = deriv.gradient(u, spectral=False)
    u_hat = grid.transform(u)
    du_spec = deriv.gradient(u_hat, spectral=True)

    assert jnp.allclose(du_phys, du_spec, atol=1e-12), (
        "spectral=True and spectral=False must give identical gradient results"
    )


def test_spectral_input_laplacian_1d():
    """
    laplacian(u, spectral=True) and laplacian(u, spectral=False) must agree.
    """
    N = 64
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(3 * grid.x) * jnp.cos(5 * grid.x)

    lap_phys = deriv.laplacian(u, spectral=False)
    u_hat = grid.transform(u)
    lap_spec = deriv.laplacian(u_hat, spectral=True)

    assert jnp.allclose(lap_phys, lap_spec, atol=1e-12)


def test_spectral_input_gradient_2d():
    """
    gradient(u, spectral=True) and gradient(u, spectral=False) must agree in 2D.
    """
    N = 32
    grid = FourierGrid2D.from_N_L(N, N, 2 * jnp.pi, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(2 * X) * jnp.cos(3 * Y)

    gx_phys, gy_phys = deriv.gradient(u, spectral=False)
    u_hat = grid.transform(u)
    gx_spec, gy_spec = deriv.gradient(u_hat, spectral=True)

    assert jnp.allclose(gx_phys, gx_spec, atol=1e-12)
    assert jnp.allclose(gy_phys, gy_spec, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Enstrophy Dissipation in 2D Navier-Stokes
# ---------------------------------------------------------------------------


def test_ns2d_enstrophy_decreases_with_viscosity():
    """
    In 2D NS with viscous dissipation (nu > 0), enstrophy Z = (1/2)*integral(omega^2)
    must decrease monotonically over time.

    Enstrophy equation:  dZ/dt = -2*nu * integral(|nabla omega|^2) <= 0.

    Physical correctness: if enstrophy ever increases when nu > 0, the time-stepper
    or operator is incorrect.
    """
    N = 32
    nu = 0.1  # large viscosity to ensure clear dissipation
    dt = 1e-3
    nsteps = 20

    grid = FourierGrid2D.from_N_L(N, N, 2 * jnp.pi, 2 * jnp.pi, dealias="2/3")
    deriv = SpectralDerivative2D(grid)
    solver = SpectralHelmholtzSolver2D(grid)
    K2 = grid.K2
    dx = grid.dx
    dy = grid.dy

    X, Y = grid.X
    omega = jnp.sin(2 * X) * jnp.cos(2 * Y)

    enstrophies = [float(jnp.sum(omega**2) * dx * dy)]

    for _ in range(nsteps):
        # Compute velocity from streamfunction
        psi = solver.solve(omega, alpha=0.0)
        dpsi_dx, dpsi_dy = deriv.gradient(psi)
        u, v = -dpsi_dy, dpsi_dx

        # Advection (explicit)
        adv = -deriv.advection_scalar(u, v, omega)

        # Viscous dissipation (explicit for simplicity)
        omega_hat = grid.transform(omega)
        lap_hat = -K2 * omega_hat
        diffusion = grid.transform(lap_hat, inverse=True).real

        omega = omega + dt * (adv + nu * diffusion)
        enstrophies.append(float(jnp.sum(omega**2) * dx * dy))

    # Enstrophy must decrease at every step with strong viscosity
    assert all(enstrophies[i + 1] <= enstrophies[i] for i in range(len(enstrophies) - 1)), (
        f"Enstrophy did not decrease monotonically; values={[f'{e:.4f}' for e in enstrophies]}"
    )
    # No step should give NaN
    assert all(np.isfinite(e) for e in enstrophies), "Enstrophy contains NaN/Inf"


def test_ns2d_energy_finite_no_viscosity():
    """
    In 2D NS without viscosity (inviscid), energy should remain approximately constant
    for small dt (no blowup over short time).

    Energy = (1/2)*integral(u^2 + v^2) = (1/2)*integral(|nabla psi|^2)
           = (1/2)*integral(omega * psi)  (using Green's identity for stream func)

    We test that energy stays finite and bounded over 20 small time steps.
    """
    N = 32
    dt = 1e-4
    nsteps = 20

    grid = FourierGrid2D.from_N_L(N, N, 2 * jnp.pi, 2 * jnp.pi, dealias="2/3")
    deriv = SpectralDerivative2D(grid)
    solver = SpectralHelmholtzSolver2D(grid)
    dx = grid.dx
    dy = grid.dy

    X, Y = grid.X
    omega = jnp.sin(2 * X) * jnp.cos(3 * Y)

    psi_init = solver.solve(omega, alpha=0.0)
    initial_energy = float(0.5 * jnp.sum(omega * psi_init) * dx * dy)

    for _ in range(nsteps):
        psi = solver.solve(omega, alpha=0.0)
        dpsi_dx, dpsi_dy = deriv.gradient(psi)
        u, v = -dpsi_dy, dpsi_dx
        adv = -deriv.advection_scalar(u, v, omega)
        omega = omega + dt * adv

    psi_final = solver.solve(omega, alpha=0.0)
    final_energy = float(0.5 * jnp.sum(omega * psi_final) * dx * dy)

    assert jnp.all(jnp.isfinite(omega)), "Inviscid NS2D: omega contains NaN/Inf"
    # Kinetic energy E = (1/2)*integral(psi*omega) should remain within 10%
    assert abs(final_energy - initial_energy) / (initial_energy + 1e-10) < 0.1, (
        f"Inviscid NS2D: energy changed by more than 10% in 20 small steps. "
        f"Initial={initial_energy:.4f}, final={final_energy:.4f}"
    )


# ---------------------------------------------------------------------------
# 7. KdV L2 Conservation
# ---------------------------------------------------------------------------


def test_kdv_l2_conservation():
    """
    The KdV equation du/dt = -6u*du/dx - d^3u/dx^3 conserves the L2 norm.

    Integral invariant: d/dt * integral(u^2 dx) = 0.

    We use a single-soliton initial condition and run a few Euler steps.
    The L2 norm should remain approximately constant (within ~1% for small dt).
    """
    N = 128
    L = 100.0
    dt = 5e-4
    nsteps = 10

    grid = FourierGrid1D.from_N_L(N, L, dealias="2/3")
    x = grid.x - L / 2.0  # center the soliton

    c = 2.0
    u = 2.0 * c / jnp.cosh(jnp.sqrt(c) * x) ** 2  # single soliton

    l2_initial = float(jnp.sum(u**2) * grid.dx)

    for _ in range(nsteps):
        u_hat = grid.transform(u)
        k = grid.k
        k_dealias = grid.k_dealias

        # Nonlinear term: -6u * du/dx
        du_dx = grid.transform(1j * k_dealias * u_hat, inverse=True).real
        advection = -6.0 * u * du_dx
        advection_hat = grid.transform(advection)

        # Dispersion term: -d^3u/dx^3
        d3u_hat = (1j * k) ** 3 * u_hat
        dispersion_hat = -d3u_hat

        total_hat = (advection_hat + dispersion_hat) * grid.dealias_filter()
        du_dt = grid.transform(total_hat, inverse=True).real
        u = u + dt * du_dt

    l2_final = float(jnp.sum(u**2) * grid.dx)

    assert jnp.all(jnp.isfinite(u)), "KdV: u contains NaN/Inf"
    relative_change = abs(l2_final - l2_initial) / (l2_initial + 1e-10)
    assert relative_change < 0.01, (
        f"KdV L2 norm changed by {100*relative_change:.2f}% (should be <1%): "
        f"initial={l2_initial:.6f}, final={l2_final:.6f}"
    )


# ---------------------------------------------------------------------------
# 8. Chebyshev Spectral Convergence
# ---------------------------------------------------------------------------


def test_chebyshev_convergence():
    """
    Chebyshev derivatives converge spectrally (faster than any algebraic rate)
    as N increases for smooth functions.

    For u = exp(sin(pi*x)) on [-1, 1], we verify that:
    - N=8 has a large O(0.5) error (only 4 interior modes, function not resolved)
    - N=32 achieves accuracy much better than any algebraic rate (< 1e-6)
    - The error ratio (N=8 / N=32) reflects super-algebraic convergence (>1e4x)
    """

    def u_func(x):
        return jnp.exp(jnp.sin(jnp.pi * x))

    def du_func(x):
        return jnp.pi * jnp.cos(jnp.pi * x) * jnp.exp(jnp.sin(jnp.pi * x))

    errors = []
    for N in [8, 32]:
        grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
        deriv = ChebyshevDerivative1D(grid)
        x = grid.x
        u = u_func(x)
        du_dx = deriv.gradient(u)
        exact = du_func(x)
        err = float(jnp.max(jnp.abs(du_dx - exact)))
        errors.append(err)

    assert errors[0] > 0.1, (
        f"N=8 should have significant error; got {errors[0]:.2e}"
    )
    assert errors[1] < 1e-6, (
        f"N=32 should be well below 1e-6, got {errors[1]:.2e}"
    )
    assert errors[1] < errors[0] * 1e-4, (
        f"Chebyshev convergence: expected >1e4x improvement from N=8 to N=32, "
        f"got {errors[0]/errors[1]:.1f}x"
    )


def test_chebyshev_derivative_chain_rule():
    """
    Chebyshev derivatives satisfy the chain rule: d/dx[u*v] = u*dv/dx + v*du/dx.

    Tests a fundamental property of the differentiation matrix.
    """
    N = 20
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    deriv = ChebyshevDerivative1D(grid)
    x = grid.x
    u = jnp.sin(jnp.pi * x)
    v = jnp.cos(jnp.pi * x / 2.0)

    product = u * v
    d_product = deriv.gradient(product)
    u_dv = u * deriv.gradient(v)
    v_du = v * deriv.gradient(u)

    assert jnp.allclose(d_product, u_dv + v_du, atol=1e-8), (
        f"Chain rule failed: max error = {float(jnp.max(jnp.abs(d_product - u_dv - v_du))):.2e}"
    )


# ---------------------------------------------------------------------------
# 9. Spherical Legendre Orthogonality / Parseval
# ---------------------------------------------------------------------------


def test_spherical_legendre_parseval():
    """
    The Legendre spectral transform preserves energy under Gauss-Legendre quadrature.

    For a field u on Gauss-Legendre nodes with weights w_j:
        sum_j w_j * u(theta_j)^2 ≈ sum_l |a_l|^2

    where a_l = sum_j w_j * P_l(cos(theta_j)) * u(theta_j) are Legendre coefficients
    and P_l are normalized Legendre polynomials. This is Parseval's theorem for
    the Discrete Legendre Transform.
    """
    from scipy.special import eval_legendre

    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)  # unit sphere R=1
    d = SphericalDerivative1D(g)

    mu = np.array(g.cos_theta)
    weights = np.array(g.weights)

    # Build a function as a truncated Legendre series with known coefficients
    # u = sum_l c_l * P_l(cos(theta)), l = 0..4
    true_coeffs = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.1, 4: 0.05}
    u = jnp.zeros(N)
    for l, c in true_coeffs.items():
        u = u + c * jnp.asarray(eval_legendre(l, mu))

    # Forward transform: compute spectral coefficients
    a = d.to_spectral(u)

    # Physical-space energy (weighted L2 norm under GL quadrature)
    energy_phys = float(jnp.sum(jnp.asarray(weights) * u**2))

    # Spectral energy: sum of squared coefficients (Parseval)
    energy_spec = float(jnp.sum(jnp.abs(a) ** 2))

    assert abs(energy_phys - energy_spec) / (energy_phys + 1e-10) < 1e-6, (
        f"Spherical Parseval failed: physical={energy_phys:.6e}, "
        f"spectral={energy_spec:.6e}"
    )


def test_spherical_legendre_orthogonality():
    """
    Legendre polynomials are orthogonal under Gauss-Legendre quadrature:
        sum_j w_j * P_l(mu_j) * P_m(mu_j) ≈ 2/(2l+1) * delta_{lm}

    This validates that the quadrature weights correctly integrate Legendre products.
    """
    from scipy.special import eval_legendre

    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    mu = np.array(g.cos_theta)
    weights = np.array(g.weights)

    # Compute inner products <P_l, P_m>_GL for l, m in 0..5
    for l in range(6):
        for m in range(6):
            Pl = eval_legendre(l, mu)
            Pm = eval_legendre(m, mu)
            inner = float(np.sum(weights * Pl * Pm))
            expected = 2.0 / (2 * l + 1) if l == m else 0.0
            assert abs(inner - expected) < 1e-10, (
                f"Orthogonality failed for l={l}, m={m}: "
                f"<P_{l}, P_{m}> = {inner:.2e}, expected {expected:.2e}"
            )


def test_spherical_transform_roundtrip_accuracy():
    """
    Forward then inverse Legendre transform should recover the original field.

    Verifies that the DLT and its inverse are consistent to machine precision.
    """
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    d = SphericalDerivative1D(g)
    mu = np.array(g.cos_theta)
    u = jnp.asarray(np.sin(np.arccos(mu)) ** 2 * np.cos(2 * np.arccos(mu)))
    a = d.to_spectral(u)
    u_rec = d.from_spectral(a)
    assert jnp.allclose(u, u_rec, atol=1e-10), (
        f"Legendre roundtrip max error = {float(jnp.max(jnp.abs(u - u_rec))):.2e}"
    )


# ---------------------------------------------------------------------------
# 10. QG PV Integral Conservation
# ---------------------------------------------------------------------------


def test_qg_mean_pv_conservation():
    """
    Under QG Jacobian advection, the domain-mean PV is conserved.

    The Jacobian J(psi, q) = psi_x * q_y - psi_y * q_x has zero spatial integral
    on a doubly-periodic domain (by integration by parts):
        integral[ J(psi, q) ] dA = 0.

    This means d/dt * mean(q) = 0 (the mean PV is an invariant).

    We run several Euler steps and verify the mean PV stays constant.
    """
    N = 32
    dt = 1e-3
    nsteps = 20

    grid = FourierGrid2D.from_N_L(N, N, 2 * jnp.pi, 2 * jnp.pi, dealias="2/3")
    deriv = SpectralDerivative2D(grid)
    solver = SpectralHelmholtzSolver2D(grid)

    X, Y = grid.X
    # Initial PV with zero mean (pure wave, no mean)
    q = jnp.sin(2 * X) * jnp.cos(3 * Y) + 0.5 * jnp.cos(X) * jnp.sin(2 * Y)

    mean_pv_initial = float(jnp.mean(q))

    for _ in range(nsteps):
        psi = solver.solve(q, alpha=0.0)
        dpsi_dx, dpsi_dy = deriv.gradient(psi)
        u, v = -dpsi_dy, dpsi_dx
        jacobian = -deriv.advection_scalar(u, v, q)
        q = q + dt * jacobian

    mean_pv_final = float(jnp.mean(q))

    assert jnp.all(jnp.isfinite(q)), "QG: q contains NaN/Inf"
    assert abs(mean_pv_final - mean_pv_initial) < 1e-10, (
        f"Mean PV changed: initial={mean_pv_initial:.2e}, "
        f"final={mean_pv_final:.2e}, diff={abs(mean_pv_final - mean_pv_initial):.2e}"
    )


def test_qg_pv_spectral_modes_preserved():
    """
    For a single Fourier mode q = sin(kx*x)*cos(ky*y), the Poisson solve
    gives psi with the same mode shape (just scaled).

    The Poisson relation is: ∇²psi = q, i.e., -(kx²+ky²)*psi_hat = q_hat,
    so psi_hat = -q_hat/(kx²+ky²), and thus psi = -q/(kx²+ky²).
    """
    N = 32
    kx, ky = 3, 2
    grid = FourierGrid2D.from_N_L(N, N, 2 * jnp.pi, 2 * jnp.pi, dealias=None)
    solver = SpectralHelmholtzSolver2D(grid)
    X, Y = grid.X

    q = jnp.sin(kx * X) * jnp.cos(ky * Y)
    psi = solver.solve(q, alpha=0.0)

    # The Helmholtz solver solves (∇² - α)psi = q, so for α=0:
    # -(kx²+ky²) * psi_hat = q_hat → psi = -q / (kx²+ky²)
    expected_psi = -q / (kx**2 + ky**2)
    assert jnp.allclose(psi, expected_psi, atol=1e-10), (
        f"PV inversion mode check failed: "
        f"max error = {float(jnp.max(jnp.abs(psi - expected_psi))):.2e}"
    )


def test_qg_energy_bounded_many_steps():
    """
    QG dynamics without dissipation: total energy should stay bounded over
    100 small time steps with a low-amplitude initial condition.

    Energy = (1/2) * integral(psi * q) dA for the barotropic QG model.
    """
    N = 32
    dt = 1e-4
    nsteps = 100

    grid = FourierGrid2D.from_N_L(N, N, 2 * jnp.pi, 2 * jnp.pi, dealias="2/3")
    deriv = SpectralDerivative2D(grid)
    solver = SpectralHelmholtzSolver2D(grid)
    dx = grid.dx
    dy = grid.dy

    X, Y = grid.X
    q = 0.1 * (jnp.sin(2 * X) * jnp.cos(3 * Y))

    def compute_energy(q_):
        psi_ = solver.solve(q_, alpha=0.0)
        return float(0.5 * jnp.sum(psi_ * q_) * dx * dy)

    energy_initial = compute_energy(q)

    for _ in range(nsteps):
        psi = solver.solve(q, alpha=0.0)
        dpsi_dx, dpsi_dy = deriv.gradient(psi)
        u, v = -dpsi_dy, dpsi_dx
        q = q + dt * (-deriv.advection_scalar(u, v, q))

    energy_final = compute_energy(q)

    assert jnp.all(jnp.isfinite(q)), "QG energy test: q contains NaN"
    assert abs(energy_final - energy_initial) / abs(energy_initial) < 0.1, (
        f"QG energy changed by more than 10% in 100 small steps: "
        f"initial={energy_initial:.4e}, final={energy_final:.4e}"
    )
