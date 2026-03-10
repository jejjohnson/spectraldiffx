"""
Tests for functionality gaps identified across all spectraldiffx modules.

These tests cover the following genuine coverage gaps:
1.  SpectralDerivative3D.advection_scalar() — completely untested
2.  SpectralFilter1D.hyperviscosity() — DC and monotonicity not tested
3.  SpectralFilter3D — only shapes tested; DC and monotonicity untested
4.  ChebyshevDerivative2D — operator identities curl(grad)=0 and div(grad)=laplacian not tested
5.  ChebyshevFilter2D.hyperviscosity() — functional correctness not tested
6.  ChebyshevHelmholtzSolver1D — error paths (wrong node_type, wrong f shape) not tested
7.  ChebyshevGrid1D.transform() — correct Chebyshev coefficient recovery for T_n not tested
8.  SphericalDerivative2D.divergence() — correctness not tested
9.  SphericalDerivative2D.advection_scalar() — correctness not tested
10. SphericalHarmonicTransform — all four class methods completely untested
11. SphericalHelmholtzSolver 1D path — only 2D tested
12. SphericalFilter1D — spectral=True consistency; hyperviscosity DC and monotone not tested
13. SphericalFilter2D — high modes damped by exponential filter not tested
14. SphericalGrid2D.weights — sphere area integration not tested
15. FourierGrid1D.k_dealias — direct correctness (zeros above cutoff) not tested
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx._src.chebyshev.filters import ChebyshevFilter2D
from spectraldiffx._src.chebyshev.grid import ChebyshevGrid1D, ChebyshevGrid2D
from spectraldiffx._src.chebyshev.operators import ChebyshevDerivative2D
from spectraldiffx._src.chebyshev.solvers import ChebyshevHelmholtzSolver1D
from spectraldiffx._src.filters import SpectralFilter1D, SpectralFilter3D
from spectraldiffx._src.grid import FourierGrid1D, FourierGrid3D
from spectraldiffx._src.operators import SpectralDerivative3D
from spectraldiffx._src.spherical.filters import SphericalFilter1D, SphericalFilter2D
from spectraldiffx._src.spherical.grid import SphericalGrid1D, SphericalGrid2D
from spectraldiffx._src.spherical.harmonics import SphericalHarmonicTransform
from spectraldiffx._src.spherical.operators import SphericalDerivative2D
from spectraldiffx._src.spherical.solvers import SphericalHelmholtzSolver


# ---------------------------------------------------------------------------
# 1. SpectralDerivative3D.advection_scalar
# ---------------------------------------------------------------------------


def test_deriv3d_advection_scalar_unit_velocity_x():
    """
    (V·∇)q for V = (0, 0, vx=1), q = sin(x): result is cos(x).

    Ordering is (Nz, Ny, Nx) with advection_scalar(vz, vy, vx, q).
    """
    N = 32
    L = 2 * jnp.pi
    grid = FourierGrid3D.from_N_L(N, N, N, L, L, L, dealias=None)
    d = SpectralDerivative3D(grid)
    Z, Y, X = grid.X

    q = jnp.sin(X)
    vz = jnp.zeros((N, N, N))
    vy = jnp.zeros((N, N, N))
    vx = jnp.ones((N, N, N))

    adv = d.advection_scalar(vz, vy, vx, q)
    expected = jnp.cos(X)
    assert jnp.allclose(adv, expected, atol=1e-10), (
        f"3D advection (vx=1) * sin(X): max error = {float(jnp.max(jnp.abs(adv - expected))):.2e}"
    )


def test_deriv3d_advection_scalar_unit_velocity_y():
    """
    (V·∇)q for V = (0, vy=1, 0), q = sin(y): result is cos(y).
    """
    N = 32
    L = 2 * jnp.pi
    grid = FourierGrid3D.from_N_L(N, N, N, L, L, L, dealias=None)
    d = SpectralDerivative3D(grid)
    Z, Y, X = grid.X

    q = jnp.sin(Y)
    vz = jnp.zeros((N, N, N))
    vy = jnp.ones((N, N, N))
    vx = jnp.zeros((N, N, N))

    adv = d.advection_scalar(vz, vy, vx, q)
    expected = jnp.cos(Y)
    assert jnp.allclose(adv, expected, atol=1e-10), (
        f"3D advection (vy=1) * sin(Y): max error = {float(jnp.max(jnp.abs(adv - expected))):.2e}"
    )


def test_deriv3d_advection_scalar_unit_velocity_z():
    """
    (V·∇)q for V = (vz=1, 0, 0), q = sin(z): result is cos(z).
    """
    N = 32
    L = 2 * jnp.pi
    grid = FourierGrid3D.from_N_L(N, N, N, L, L, L, dealias=None)
    d = SpectralDerivative3D(grid)
    Z, Y, X = grid.X

    q = jnp.sin(Z)
    vz = jnp.ones((N, N, N))
    vy = jnp.zeros((N, N, N))
    vx = jnp.zeros((N, N, N))

    adv = d.advection_scalar(vz, vy, vx, q)
    expected = jnp.cos(Z)
    assert jnp.allclose(adv, expected, atol=1e-10), (
        f"3D advection (vz=1) * sin(Z): max error = {float(jnp.max(jnp.abs(adv - expected))):.2e}"
    )


def test_deriv3d_advection_scalar_zero_velocity():
    """Zero velocity gives zero advection for any tracer."""
    N = 16
    L = 2 * jnp.pi
    grid = FourierGrid3D.from_N_L(N, N, N, L, L, L, dealias=None)
    d = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    q = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)
    zero = jnp.zeros((N, N, N))
    adv = d.advection_scalar(zero, zero, zero, q)
    assert jnp.allclose(adv, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. SpectralFilter1D.hyperviscosity — DC and monotonicity
# ---------------------------------------------------------------------------


def test_filter1d_hyperviscosity_dc_preserved():
    """
    Hyperviscosity filter: F(k=0) = exp(-nu * 0^power * dt) = 1.
    The DC mode (k=0) must be exactly preserved.
    """
    N = 64
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi)
    filt = SpectralFilter1D(grid)
    u_hat = jnp.zeros(N, dtype=jnp.complex128).at[0].set(1.0)  # DC only
    u_hat_f = filt.hyperviscosity(u_hat, nu_hyper=1e-4, dt=0.1, spectral=True)
    assert jnp.isclose(jnp.abs(u_hat_f[0]), 1.0, atol=1e-15), (
        f"DC mode should be preserved by hyperviscosity; got {float(jnp.abs(u_hat_f[0])):.6f}"
    )


def test_filter1d_hyperviscosity_monotone():
    """
    Hyperviscosity filter is monotonically decreasing: higher |k| → more damping.

    F(k) = exp(-nu * |k|^p * dt), so F(k3) < F(k10) for k3 < k10 when nu, dt > 0.
    """
    N = 64
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi)
    filt = SpectralFilter1D(grid)

    amplitudes = []
    for k in [0, 1, 3, 5, 10, 20]:
        u_hat = jnp.zeros(N, dtype=jnp.complex128).at[k].set(1.0)
        u_filt = filt.hyperviscosity(u_hat, nu_hyper=1e-4, dt=0.1, spectral=True)
        amplitudes.append(float(jnp.abs(u_filt[k])))

    # All amplitudes should be <= 1.0
    assert all(a <= 1.0 for a in amplitudes), f"Hyperviscosity must not amplify: {amplitudes}"
    # Amplitudes should decrease with k (higher k → more damping → smaller amplitude)
    for i in range(1, len(amplitudes) - 1):
        assert amplitudes[i + 1] <= amplitudes[i], (
            f"Hyperviscosity not monotone: F(k={[0,1,3,5,10,20][i]})={amplitudes[i]:.6f} "
            f"> F(k={[0,1,3,5,10,20][i+1]})={amplitudes[i+1]:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. SpectralFilter3D — DC preservation and monotonicity
# ---------------------------------------------------------------------------


def test_filter3d_exponential_dc_preserved():
    """
    3D exponential filter: F(k=0) = 1 — the DC mode must be unaffected.
    """
    N = 16
    grid = FourierGrid3D.from_N_L(N, N, N, 2 * jnp.pi, 2 * jnp.pi, 2 * jnp.pi)
    filt = SpectralFilter3D(grid)

    u_hat = jnp.zeros((N, N, N), dtype=jnp.complex128).at[0, 0, 0].set(1.0)
    u_hat_f = filt.exponential_filter(u_hat, spectral=True)
    assert jnp.isclose(jnp.abs(u_hat_f[0, 0, 0]), 1.0, atol=1e-15), (
        "3D exponential filter: DC mode must be preserved"
    )


def test_filter3d_exponential_high_modes_damped():
    """
    3D exponential filter: higher wavenumber magnitude → more damping.

    Low mode (index 2) should be damped less than high mode (index 8).
    """
    N = 16
    grid = FourierGrid3D.from_N_L(N, N, N, 2 * jnp.pi, 2 * jnp.pi, 2 * jnp.pi)
    filt = SpectralFilter3D(grid)

    u_low = jnp.zeros((N, N, N), dtype=jnp.complex128).at[2, 0, 0].set(1.0)
    u_high = jnp.zeros((N, N, N), dtype=jnp.complex128).at[8, 0, 0].set(1.0)

    f_low = float(jnp.abs(filt.exponential_filter(u_low, spectral=True)[2, 0, 0]))
    f_high = float(jnp.abs(filt.exponential_filter(u_high, spectral=True)[8, 0, 0]))

    assert f_high < f_low, (
        f"3D filter: high mode (k=8) should be damped more than low mode (k=2), "
        f"got F(2)={f_low:.4f} and F(8)={f_high:.4f}"
    )


def test_filter3d_hyperviscosity_dc_preserved():
    """3D hyperviscosity filter: DC mode must be exactly preserved."""
    N = 16
    grid = FourierGrid3D.from_N_L(N, N, N, 2 * jnp.pi, 2 * jnp.pi, 2 * jnp.pi)
    filt = SpectralFilter3D(grid)
    u_hat = jnp.zeros((N, N, N), dtype=jnp.complex128).at[0, 0, 0].set(1.0)
    u_hat_f = filt.hyperviscosity(u_hat, nu_hyper=1e-4, dt=0.1, spectral=True)
    assert jnp.isclose(jnp.abs(u_hat_f[0, 0, 0]), 1.0, atol=1e-15), (
        "3D hyperviscosity: DC mode must be preserved"
    )


# ---------------------------------------------------------------------------
# 4. ChebyshevDerivative2D operator identities
# ---------------------------------------------------------------------------


def test_cheb_deriv2d_curl_of_gradient_zero():
    """
    curl(grad(phi)) = 0 for any smooth scalar phi on the Chebyshev grid.

    curl(grad phi) = d(dphi/dy)/dx - d(dphi/dx)/dy
                  = Dx(Dy @ phi) - Dy(phi @ Dx.T) = 0 (matrices commute).

    This is the vector calculus identity curl(grad) = 0 verified on the
    Chebyshev differentiation matrices.
    """
    N = 16
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    d = ChebyshevDerivative2D(grid)
    X, Y = grid.X

    phi = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)
    dphi_dx, dphi_dy = d.gradient(phi)
    curl_val = d.curl(dphi_dx, dphi_dy)

    assert jnp.allclose(curl_val, 0.0, atol=1e-10), (
        f"curl(grad(phi)) ≠ 0: max abs = {float(jnp.max(jnp.abs(curl_val))):.2e}"
    )


def test_cheb_deriv2d_laplacian_equals_div_grad():
    """
    laplacian(u) == div(grad(u)) for the Chebyshev operators.

    Both are constructed from D^2 applied along each axis, so they must agree
    exactly up to floating-point rounding.

    Verifies: (u @ Dx2.T) + (Dy2 @ u) == d/dx(du/dx) + d/dy(du/dy).
    """
    N = 16
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    d = ChebyshevDerivative2D(grid)
    X, Y = grid.X

    u = jnp.sin(jnp.pi * X) * jnp.cos(0.5 * jnp.pi * Y)
    lap = d.laplacian(u)
    gx, gy = d.gradient(u)
    div_g = d.divergence(gx, gy)

    assert jnp.allclose(lap, div_g, atol=1e-10), (
        f"laplacian ≠ div(grad): max diff = {float(jnp.max(jnp.abs(lap - div_g))):.2e}"
    )


def test_cheb_deriv2d_gradient_known_values():
    """
    d/dx[sin(pi*x)*cos(pi*y)] = pi*cos(pi*x)*cos(pi*y) at Chebyshev nodes.
    d/dy[sin(pi*x)*cos(pi*y)] = -pi*sin(pi*x)*sin(pi*y) at Chebyshev nodes.
    """
    N = 24
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    d = ChebyshevDerivative2D(grid)
    X, Y = grid.X

    u = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)
    du_dx, du_dy = d.gradient(u)

    expected_dx = jnp.pi * jnp.cos(jnp.pi * X) * jnp.cos(jnp.pi * Y)
    expected_dy = -jnp.pi * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

    assert jnp.allclose(du_dx, expected_dx, atol=1e-8), (
        f"d/dx error: {float(jnp.max(jnp.abs(du_dx - expected_dx))):.2e}"
    )
    assert jnp.allclose(du_dy, expected_dy, atol=1e-8), (
        f"d/dy error: {float(jnp.max(jnp.abs(du_dy - expected_dy))):.2e}"
    )


# ---------------------------------------------------------------------------
# 5. ChebyshevFilter2D.hyperviscosity — functional correctness
# ---------------------------------------------------------------------------


def test_cheb_filter2d_hyperviscosity_dc_preserved():
    """
    ChebyshevFilter2D.hyperviscosity(): the k=0 coefficient must be unchanged.

    F(kx=0, ky=0) = exp(-nu * 0^power * dt) = 1.
    A constant field has only the k=0 Chebyshev mode and should pass through.
    """
    N = 8
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    n_pts = N + 1  # Gauss-Lobatto has N+1 points
    u = jnp.ones((n_pts, n_pts))
    u_filt = filt.hyperviscosity(u, nu_hyper=1e-3, dt=0.1)
    assert jnp.allclose(u_filt, u, atol=1e-13), (
        f"Constant field should be unchanged by hyperviscosity filter; "
        f"max diff = {float(jnp.max(jnp.abs(u_filt - u))):.2e}"
    )


def test_cheb_filter2d_hyperviscosity_high_modes_damped():
    """
    ChebyshevFilter2D.hyperviscosity(): high-index coefficients are reduced.
    """
    N = 8
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    n_modes = N + 1  # Gauss-Lobatto

    a_high = jnp.zeros((n_modes, n_modes)).at[-1, -1].set(1.0)
    a_filt = filt.hyperviscosity(a_high, nu_hyper=1e-3, dt=0.1, spectral=True)
    assert float(jnp.abs(a_filt[-1, -1])) < 1.0, (
        "Highest Chebyshev coefficient should be damped by hyperviscosity"
    )


def test_cheb_filter2d_exponential_spectral_input_consistent():
    """
    ChebyshevFilter2D.exponential_filter(spectral=True) must match
    transforming to spectral space, filtering, then transforming back.
    """
    N = 8
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    n_pts = N + 1
    X, Y = grid.X
    u = jnp.sin(jnp.pi * X / 2) * jnp.cos(jnp.pi * Y / 2)

    # Filter in physical space
    u_filt_phys = filt.exponential_filter(u, spectral=False)

    # Filter in spectral space
    a = grid.transform(u)
    a_filt = filt.exponential_filter(a, spectral=True)
    u_filt_from_spec = grid.transform(a_filt, inverse=True)

    assert jnp.allclose(u_filt_phys, u_filt_from_spec, atol=1e-12), (
        f"exponential_filter spectral=True vs False: max diff = "
        f"{float(jnp.max(jnp.abs(u_filt_phys - u_filt_from_spec))):.2e}"
    )


# ---------------------------------------------------------------------------
# 6. ChebyshevHelmholtzSolver1D — error paths
# ---------------------------------------------------------------------------


def test_cheb_solver1d_raises_for_gauss_nodes():
    """
    ChebyshevHelmholtzSolver1D must raise ValueError when given Gauss (not
    Gauss-Lobatto) nodes, because the boundary-row replacement method requires
    the endpoints x[0]=+L and x[N]=-L.
    """
    grid_gauss = ChebyshevGrid1D.from_N_L(N=16, L=1.0, node_type="gauss")
    solver = ChebyshevHelmholtzSolver1D(grid_gauss)
    with pytest.raises(ValueError, match="gauss-lobatto"):
        solver.solve(jnp.zeros(16), alpha=0.0)


def test_cheb_solver1d_raises_for_wrong_f_shape():
    """
    ChebyshevHelmholtzSolver1D must raise ValueError when f has wrong length.

    For N GL nodes, f must have length N+1; any other length should fail.
    """
    N = 16
    grid_gl = ChebyshevGrid1D.from_N_L(N=N, L=1.0, node_type="gauss-lobatto")
    solver = ChebyshevHelmholtzSolver1D(grid_gl)
    with pytest.raises(ValueError, match=f"N\\+1={N + 1}"):
        solver.solve(jnp.zeros(N), alpha=0.0)  # N instead of N+1


# ---------------------------------------------------------------------------
# 7. ChebyshevGrid1D.transform — correct Chebyshev coefficient recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_cheb_grid_transform_chebyshev_basis(n: int):
    """
    Forward transform of T_n(x/L) must give coefficient vector with:
        a[n] = 2 if n == 0, else 1
        a[k] = 0 for all k != n.

    This tests that the Chebyshev transform correctly identifies each basis
    function T_n as having a single nonzero spectral coefficient.
    """
    N = 24
    L = 1.0
    grid = ChebyshevGrid1D.from_N_L(N=N, L=L)
    x = grid.x

    # T_n(x/L) = cos(n * arccos(x/L)) — valid because x ∈ [-L, L]
    u = jnp.cos(n * jnp.arccos(x / L))
    a = grid.transform(u)

    expected_amplitude = 2.0 if n == 0 else 1.0
    assert jnp.isclose(a[n], expected_amplitude, atol=1e-12), (
        f"T_{n}: a[{n}] = {float(a[n]):.8f}, expected {expected_amplitude}"
    )
    # All other coefficients should be zero
    a_other = a.at[n].set(0.0)
    assert jnp.allclose(a_other, 0.0, atol=1e-12), (
        f"T_{n}: non-zero off-diagonal coefficients, max = "
        f"{float(jnp.max(jnp.abs(a_other))):.2e}"
    )


# ---------------------------------------------------------------------------
# 8. SphericalDerivative2D.divergence — correctness
# ---------------------------------------------------------------------------


def test_spherical_deriv2d_divergence_of_zonal_gradient():
    """
    div(grad(cos(theta))) = laplacian(cos(theta)) for a zonal field.

    For a purely zonal field (no phi dependence), the theta-gradient is
    accurately computed by the 1D Legendre derivative, so div(grad(u))
    should match laplacian(u) from the eigenvalue computation.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    THETA = g.y[:, None] * jnp.ones((Ny, Nx))

    u = jnp.cos(THETA)  # P_1(cos(theta)), l=1 eigenvalue -2/R^2
    grad_th, grad_ph = d.gradient(u)
    div_grad = d.divergence(grad_th, grad_ph)
    lap = d.laplacian(u)

    assert jnp.allclose(div_grad, lap, atol=1e-8), (
        f"div(grad(cos(theta))) vs laplacian: max error = "
        f"{float(jnp.max(jnp.abs(div_grad - lap))):.2e}"
    )


def test_spherical_deriv2d_divergence_free_streamfunction_velocity():
    """
    The velocity field derived from a streamfunction must be divergence-free.

    For ψ(theta, phi) = (1 - cos²(theta)) * cos(phi) [theta-part is polynomial],
    the geostrophic velocity components are:
        v_theta = -1/(R*sin) * dψ/dphi = sin(phi)*sin(theta)/R
        v_phi   = 1/R * dψ/dtheta     = 2*sin(theta)*cos(theta)*cos(phi)/R

    Analytically: div(v_theta, v_phi) = 0.
    """
    import jax

    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    # ψ = (1 - cos²θ)*cos(φ) = sin²(θ)*cos(φ) — theta part is a polynomial in cos(θ)
    psi = (1.0 - jnp.cos(THETA) ** 2) * jnp.cos(PHI)
    sin_theta = jnp.sin(THETA)

    # Compute dψ/dphi via FFT
    psi_hat = jnp.fft.fft(psi, axis=-1)
    m_phys = 2 * jnp.pi * jnp.fft.fftfreq(Nx, g.dx)
    dpsi_dphi = jnp.fft.ifft(1j * m_phys[None, :] * psi_hat, axis=-1).real

    # Compute dψ/dtheta via Legendre derivative
    dpsi_dtheta = jax.vmap(d.deriv_theta.gradient, in_axes=1, out_axes=1)(psi)

    v_theta = -dpsi_dphi / (R * sin_theta)
    v_phi = dpsi_dtheta / R

    div_v = d.divergence(v_theta, v_phi)
    assert jnp.allclose(div_v, 0.0, atol=1e-10), (
        f"Streamfunction velocity should be divergence-free; "
        f"max |div| = {float(jnp.max(jnp.abs(div_v))):.2e}"
    )


def test_spherical_deriv2d_divergence_known_div_free_field():
    """
    Analytically divergence-free field should have zero divergence.

    For V = (sin(phi)*sin(theta)/R, sin(2*theta)*cos(phi)/R):
    Analytically: div(V) = 0.

    This can be verified by computing:
        d(V_theta*sin)/dtheta = d(sin²θ*sin(phi)/R)/dtheta = sin(2θ)*sin(phi)/R
        dV_phi/dphi = d(sin(2θ)*cos(phi)/R)/dphi = -sin(2θ)*sin(phi)/R
    They cancel, so div = 0.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    v_theta = jnp.sin(PHI) * jnp.sin(THETA) / R
    v_phi = jnp.sin(2 * THETA) * jnp.cos(PHI) / R

    div_v = d.divergence(v_theta, v_phi)
    assert jnp.allclose(div_v, 0.0, atol=1e-10), (
        f"Known div-free field: max |div| = {float(jnp.max(jnp.abs(div_v))):.2e}"
    )


# ---------------------------------------------------------------------------
# 9. SphericalDerivative2D.advection_scalar — correctness
# ---------------------------------------------------------------------------


def test_spherical_deriv2d_advection_scalar_zero_velocity():
    """
    Zero velocity must give zero advection for any tracer field.
    (V·∇)q = 0 when V = (0, 0).
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    THETA = g.y[:, None] * jnp.ones((Ny, Nx))

    q = jnp.cos(THETA)
    zero = jnp.zeros((Ny, Nx))
    adv = d.advection_scalar(zero, zero, q)
    assert jnp.allclose(adv, 0.0, atol=1e-15), (
        "Zero velocity must give zero advection"
    )


def test_spherical_deriv2d_advection_scalar_zonal_flow():
    """
    For v_theta = 1/R, v_phi = 0, q = cos(theta) (zonal field):

        (V·∇)q = v_theta * grad_theta(q)
               = (1/R) * (-sin(theta)/R)
               = -sin(theta) / R²

    This tests that the theta-gradient path of advection_scalar is correct.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    v_theta = jnp.ones((Ny, Nx)) / R
    v_phi = jnp.zeros((Ny, Nx))
    q = jnp.cos(THETA)

    adv = d.advection_scalar(v_theta, v_phi, q)
    expected = -jnp.sin(THETA) / R**2
    assert jnp.allclose(adv, expected, atol=1e-10), (
        f"Zonal advection: max error = {float(jnp.max(jnp.abs(adv - expected))):.2e}"
    )


def test_spherical_deriv2d_advection_scalar_phi_flow():
    """
    For v_theta = 0, v_phi = 1.0, q = cos(phi) (periodic in phi):

        (V·∇)q = v_phi * grad_phi(cos(phi))
               = 1.0 * (-sin(phi) / (R * sin(theta)))
               = -sin(phi) / (R * sin(theta))

    This tests that the phi-gradient path of advection_scalar is correct.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    v_theta = jnp.zeros((Ny, Nx))
    v_phi = jnp.ones((Ny, Nx))
    q = jnp.cos(PHI)

    adv = d.advection_scalar(v_theta, v_phi, q)
    sin_theta = jnp.sin(THETA)
    expected = -jnp.sin(PHI) / (R * sin_theta)
    assert jnp.allclose(adv, expected, atol=1e-10), (
        f"Phi-flow advection: max error = {float(jnp.max(jnp.abs(adv - expected))):.2e}"
    )


# ---------------------------------------------------------------------------
# 10. SphericalHarmonicTransform — class methods
# ---------------------------------------------------------------------------


def test_sht_to_spectral_from_spectral_roundtrip():
    """
    SphericalHarmonicTransform.to_spectral then from_spectral must recover
    the original field.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    sht = SphericalHarmonicTransform(g)
    PHI, THETA = g.X
    u = jnp.sin(THETA) * jnp.cos(PHI) + 0.5 * jnp.cos(2 * THETA)

    u_hat = sht.to_spectral(u)
    u_rec = sht.from_spectral(u_hat)

    assert jnp.allclose(u, u_rec, atol=1e-12), (
        f"SHT class roundtrip: max error = {float(jnp.max(jnp.abs(u_rec - u))):.2e}"
    )


def test_sht_to_spectral_1d_from_spectral_1d_roundtrip():
    """
    SphericalHarmonicTransform.to_spectral_1d then from_spectral_1d must
    recover the original 1D (zonal) field.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    sht = SphericalHarmonicTransform(g)
    theta = g.y
    u_col = jnp.cos(theta) + 0.3 * (1.0 - jnp.cos(theta) ** 2)  # polynomial in cos(theta)

    c = sht.to_spectral_1d(u_col)
    u_rec = sht.from_spectral_1d(c)

    assert jnp.allclose(u_col, u_rec, atol=1e-12), (
        f"SHT 1D roundtrip: max error = {float(jnp.max(jnp.abs(u_rec - u_col))):.2e}"
    )


def test_sht_class_consistency_with_grid_transform():
    """
    SphericalHarmonicTransform.to_spectral/from_spectral must produce the
    same results as SphericalGrid2D.transform (they are thin wrappers).
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    sht = SphericalHarmonicTransform(g)
    PHI, THETA = g.X
    u = jnp.sin(THETA) ** 2 * jnp.cos(PHI)

    u_hat_sht = sht.to_spectral(u)
    u_hat_grid = g.transform(u)
    assert jnp.allclose(u_hat_sht, u_hat_grid, atol=1e-15), (
        "SHT.to_spectral must match SphericalGrid2D.transform"
    )

    u_rec_sht = sht.from_spectral(u_hat_sht)
    u_rec_grid = g.transform(u_hat_grid, inverse=True)
    assert jnp.allclose(u_rec_sht, u_rec_grid, atol=1e-15), (
        "SHT.from_spectral must match SphericalGrid2D.transform(inverse=True)"
    )


# ---------------------------------------------------------------------------
# 11. SphericalHelmholtzSolver 1D path
# ---------------------------------------------------------------------------


def test_spherical_helmholtz_1d_eigenfunction():
    """
    Helmholtz equation on the 1D sphere: (∇²_sphere - alpha) phi = f.

    For phi = P_2(cos(theta)) (l=2, eigenvalue -6/R²):
        f = (∇² - alpha) phi = -(6/R² + alpha) * P_2
    Solving should recover phi = P_2.
    """
    from scipy.special import eval_legendre

    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)  # R = 1
    solver = SphericalHelmholtzSolver(grid=g)
    R = g.L / np.pi  # = 1

    mu = np.array(g.cos_theta)
    P2 = jnp.asarray(eval_legendre(2, mu))  # P_2, l=2 → eigenvalue 6/R²

    alpha = 1.0
    f = -(6.0 / R**2 + alpha) * P2  # (∇² - α) P_2 = -(6/R² + α) * P_2

    phi = solver.solve(f, alpha=alpha, zero_mean=False)
    assert jnp.allclose(phi, P2, atol=1e-10), (
        f"1D Helmholtz: max error = {float(jnp.max(jnp.abs(phi - P2))):.2e}"
    )


def test_spherical_helmholtz_1d_reduces_to_poisson_for_alpha_zero():
    """
    With alpha=0, SphericalHelmholtzSolver reduces to Poisson: ∇² phi = f.
    Solution of ∇² P_l = -l(l+1)/R² * P_l should give back P_l.
    """
    from scipy.special import eval_legendre

    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    solver = SphericalHelmholtzSolver(grid=g)
    R = g.L / np.pi

    mu = np.array(g.cos_theta)
    l = 3
    Pl = jnp.asarray(eval_legendre(l, mu))
    f = -(l * (l + 1)) / R**2 * Pl  # ∇² Pl = -l(l+1)/R² * Pl

    phi = solver.solve(f, alpha=0.0, zero_mean=True)
    assert jnp.allclose(phi, Pl, atol=1e-10), (
        f"1D Helmholtz α=0 (Poisson): max error = {float(jnp.max(jnp.abs(phi - Pl))):.2e}"
    )


# ---------------------------------------------------------------------------
# 12. SphericalFilter1D — spectral=True consistency and hyperviscosity
# ---------------------------------------------------------------------------


def test_spherical_filter1d_spectral_input_consistent():
    """
    SphericalFilter1D.exponential_filter(u, spectral=False) and
    (transform, filter, inverse-transform) must give the same result.
    """
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    filt = SphericalFilter1D(g)
    theta = g.x
    u = jnp.sin(theta) ** 2 + 0.1 * (1 - jnp.cos(theta) ** 2)  # polynomial in cos

    u_filt_phys = filt.exponential_filter(u, spectral=False)

    c = g.transform(u)
    c_filt = filt.exponential_filter(c, spectral=True)
    u_filt_from_spec = g.transform(c_filt, inverse=True)

    assert jnp.allclose(u_filt_phys, u_filt_from_spec, atol=1e-14), (
        f"SphericalFilter1D spectral=True vs False: max diff = "
        f"{float(jnp.max(jnp.abs(u_filt_phys - u_filt_from_spec))):.2e}"
    )


def test_spherical_filter1d_hyperviscosity_dc_preserved():
    """
    SphericalFilter1D.hyperviscosity(): l=0 (mean) mode must be preserved.

    F(l=0) = exp(-nu * [0*(0+1)/R²]^(p/2) * dt) = exp(0) = 1.
    """
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    filt = SphericalFilter1D(g)
    u = jnp.ones(N)  # constant field → only l=0 mode
    u_filt = filt.hyperviscosity(u, nu_hyper=1e-3, dt=0.1)
    assert jnp.allclose(u_filt, u, atol=1e-13), (
        f"SphericalFilter1D hyperviscosity: constant field changed; "
        f"max diff = {float(jnp.max(jnp.abs(u_filt - u))):.2e}"
    )


def test_spherical_filter1d_hyperviscosity_high_modes_damped():
    """
    SphericalFilter1D.hyperviscosity(): the highest-l mode must be damped.
    """
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    filt = SphericalFilter1D(g)
    # Put all amplitude in the highest mode
    c_high = jnp.zeros(N).at[-1].set(1.0)
    c_filt = filt.hyperviscosity(c_high, nu_hyper=1e-4, dt=0.1, spectral=True)
    assert float(jnp.abs(c_filt[-1])) < 1.0, (
        "Highest-l mode should be damped by hyperviscosity"
    )


# ---------------------------------------------------------------------------
# 13. SphericalFilter2D — high modes damped
# ---------------------------------------------------------------------------


def test_spherical_filter2d_exponential_high_modes_damped():
    """
    SphericalFilter2D.exponential_filter(): highest-l mode must be damped.

    For a spectral coefficient at l=Ny-1 (highest degree), the filter value
    F(l_max) = exp(-alpha * 1^power) = exp(-alpha) << 1.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    filt = SphericalFilter2D(g)

    # Spectral coefficient at the highest l
    u_hat_high = jnp.zeros((Ny, Nx), dtype=jnp.complex128).at[-1, 0].set(1.0)
    u_hat_filt = filt.exponential_filter(u_hat_high, alpha=36.0, power=16, spectral=True)

    assert float(jnp.abs(u_hat_filt[-1, 0])) < 1e-10, (
        f"SphericalFilter2D: highest-l mode should be near-zero after filtering; "
        f"got {float(jnp.abs(u_hat_filt[-1, 0])):.2e}"
    )


def test_spherical_filter2d_hyperviscosity_dc_preserved():
    """
    SphericalFilter2D.hyperviscosity(): l=0 (DC) mode must be preserved.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    filt = SphericalFilter2D(g)
    u = jnp.ones((Ny, Nx))
    u_filt = filt.hyperviscosity(u, nu_hyper=1e-3, dt=0.1)
    assert jnp.allclose(u_filt, u, atol=1e-12), (
        f"SphericalFilter2D hyperviscosity: constant field changed; "
        f"max diff = {float(jnp.max(jnp.abs(u_filt - u))):.2e}"
    )


# ---------------------------------------------------------------------------
# 14. SphericalGrid2D.weights — sphere area integration
# ---------------------------------------------------------------------------


def test_spherical_grid_2d_weights_sphere_area():
    """
    The sum of SphericalGrid2D.weights should equal the area of the sphere.

    Integrating u=1 over the sphere:
        ∫∫ dΩ = ∫₀²π dphi * ∫₀^π sin(theta) dtheta = 2π * 2 = 4π

    For radius R = Ly/π: area = 4π * R².
    The weights w[j,k] = w_lat[j] * dx_lon absorb the sin(theta) Jacobian
    (GL weights integrate mu=cos(theta) over [-1,1]), so:
        Σ_jk w[j,k] * 1 = (Σ_j w_lat[j]) * Nx * dx = 2 * Lx = 4π (for R=1).
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    R = g.Ly / np.pi

    area = float(jnp.sum(g.weights))
    expected = 4 * np.pi * R**2

    assert abs(area - expected) < 1e-12, (
        f"Sphere area: sum(weights) = {area:.12f}, expected 4πR² = {expected:.12f}"
    )


def test_spherical_grid_2d_weights_shape():
    """SphericalGrid2D.weights must have shape (Ny, Nx)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    assert g.weights.shape == (Ny, Nx)


# ---------------------------------------------------------------------------
# 15. FourierGrid1D.k_dealias — direct correctness
# ---------------------------------------------------------------------------


def test_fourier_grid_1d_k_dealias_zeros_above_cutoff():
    """
    FourierGrid1D.k_dealias must set |k| > k_max*2/3 to zero.

    For N=32, k_max = 16 * 2π/L, cutoff = k_max * 2/3 ≈ 10.67 * 2π/L.
    Wavenumbers with |k| > cutoff must be zeroed out.
    """
    N = 32
    L = 2 * jnp.pi
    grid = FourierGrid1D.from_N_L(N, L, dealias="2/3")
    k = grid.k
    k_d = grid.k_dealias

    k_max = float(jnp.max(jnp.abs(k)))
    cutoff = k_max * 2.0 / 3.0

    above_cutoff_mask = jnp.abs(k) > cutoff
    below_or_equal_mask = ~above_cutoff_mask

    # Modes above cutoff must be zeroed
    assert jnp.allclose(k_d[above_cutoff_mask], 0.0, atol=1e-15), (
        "k_dealias: modes above cutoff must be zero"
    )
    # Modes below or equal cutoff must be preserved
    assert jnp.allclose(k_d[below_or_equal_mask], k[below_or_equal_mask], atol=1e-15), (
        "k_dealias: modes at or below cutoff must be unchanged"
    )


def test_fourier_grid_1d_k_dealias_none_unchanged():
    """
    With dealias=None, k_dealias must equal k (no modes zeroed out).
    """
    N = 32
    L = 2 * jnp.pi
    grid = FourierGrid1D.from_N_L(N, L, dealias=None)
    k = grid.k
    k_d = grid.k_dealias
    assert jnp.allclose(k_d, k, atol=1e-15), (
        "With dealias=None, k_dealias must equal k"
    )


def test_fourier_grid_1d_k_dealias_dc_always_zero():
    """
    The DC mode (k=0) in k_dealias must always be zero (the 0th wavenumber
    is zero in the FFT convention, so the dealiased value is also zero).
    """
    N = 32
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias="2/3")
    k_d = grid.k_dealias
    assert float(k_d[0]) == 0.0, "k_dealias[0] (DC mode) must be zero"
