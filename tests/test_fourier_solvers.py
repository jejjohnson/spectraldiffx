"""Tests for spectral elliptic solvers (Layer 0 functions + Layer 1 classes)."""

import jax.numpy as jnp

from spectraldiffx._src.fourier.eigenvalues import (
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
)
from spectraldiffx._src.fourier.solvers import (
    DirichletHelmholtzSolver2D,
    NeumannHelmholtzSolver2D,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
    solve_helmholtz_fft_1d,
    solve_poisson_dct,
    solve_poisson_dst,
    solve_poisson_fft,
    solve_poisson_fft_1d,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX = 32
NY = 24
DX = 1.0
DY = 1.0


def _eigenfunction_dst(Ny, Nx, dx, dy, kx=1, ky=1):
    """Build a DST-I eigenfunction sin(π(kx)(i+1)/(Nx+1)) * sin(π(ky)(j+1)/(Ny+1))."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    fx = jnp.sin(jnp.pi * kx * (i + 1) / (Nx + 1))
    fy = jnp.sin(jnp.pi * ky * (j + 1) / (Ny + 1))
    return fy[:, None] * fx[None, :]


def _eigenfunction_dct(Ny, Nx, dx, dy, kx=1, ky=1):
    """Build a DCT-II eigenfunction cos(πkx·i/Nx) * cos(πky·j/Ny)."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    # DCT-II eigenfunction: cos(πk(2n+1)/(2N)) but for the stencil eigenfunction
    # we use cos(πk·n/N) which is the grid-function that the DCT-II diagonalises
    fx = jnp.cos(jnp.pi * kx * (2 * i + 1) / (2 * Nx))
    fy = jnp.cos(jnp.pi * ky * (2 * j + 1) / (2 * Ny))
    return fy[:, None] * fx[None, :]


def _eigenfunction_fft(Ny, Nx, dx, dy, kx=1, ky=1):
    """Build a periodic eigenfunction cos(2πkx·i/Nx) * cos(2πky·j/Ny)."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    fx = jnp.cos(2 * jnp.pi * kx * i / Nx)
    fy = jnp.cos(2 * jnp.pi * ky * j / Ny)
    return fy[:, None] * fx[None, :]


# ---------------------------------------------------------------------------
# DST-I (Dirichlet) solvers
# ---------------------------------------------------------------------------


class TestDSTSolvers:
    """Dirichlet spectral solvers via DST-I."""

    def test_helmholtz_eigenfunction_recovery(self):
        """Solving (∇² - λ)ψ = f for a known eigenfunction."""
        lambda_ = 1.0
        psi_exact = _eigenfunction_dst(NY, NX, DX, DY, kx=2, ky=3)
        # Eigenvalue of the 5-point Laplacian for this mode
        eigx = dst1_eigenvalues(NX, DX)[1]  # kx=2 → index 1
        eigy = dst1_eigenvalues(NY, DY)[2]  # ky=3 → index 2
        eig = eigx + eigy
        rhs = (eig - lambda_) * psi_exact
        psi_got = solve_helmholtz_dst(rhs, DX, DY, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_poisson_eigenfunction_recovery(self):
        psi_exact = _eigenfunction_dst(NY, NX, DX, DY, kx=1, ky=1)
        eigx = dst1_eigenvalues(NX, DX)[0]
        eigy = dst1_eigenvalues(NY, DY)[0]
        rhs = (eigx + eigy) * psi_exact
        psi_got = solve_poisson_dst(rhs, DX, DY)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros((NY, NX))
        psi = solve_poisson_dst(rhs, DX, DY)
        assert jnp.allclose(psi, 0.0, atol=1e-10)

    def test_helmholtz_nonzero_lambda(self):
        """Non-zero λ should not crash and should differ from Poisson."""
        rhs = _eigenfunction_dst(NY, NX, DX, DY)
        psi_poisson = solve_poisson_dst(rhs, DX, DY)
        psi_helm = solve_helmholtz_dst(rhs, DX, DY, lambda_=2.0)
        # Different λ → different solution
        assert not jnp.allclose(psi_poisson, psi_helm, atol=1e-3)


# ---------------------------------------------------------------------------
# DCT-II (Neumann) solvers
# ---------------------------------------------------------------------------


class TestDCTSolvers:
    """Neumann spectral solvers via DCT-II."""

    def test_helmholtz_eigenfunction_recovery(self):
        lambda_ = 1.5
        psi_exact = _eigenfunction_dct(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct2_eigenvalues(NX, DX)[2]
        eigy = dct2_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_dct(rhs, DX, DY, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_poisson_zero_mean_gauge(self):
        """Poisson solution with Neumann BCs should have zero mean."""
        rhs = _eigenfunction_dct(NY, NX, DX, DY, kx=1, ky=1)
        psi = solve_poisson_dct(rhs, DX, DY)
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros((NY, NX))
        psi = solve_poisson_dct(rhs, DX, DY)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# FFT (Periodic) solvers — 2D
# ---------------------------------------------------------------------------


class TestFFTSolvers2D:
    """Periodic spectral solvers via FFT (2D)."""

    def test_helmholtz_eigenfunction_recovery(self):
        lambda_ = 1.0
        psi_exact = _eigenfunction_fft(NY, NX, DX, DY, kx=2, ky=3)
        eigx = fft_eigenvalues(NX, DX)[2]
        eigy = fft_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_fft(rhs, DX, DY, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_poisson_zero_mean_gauge(self):
        """FFT Poisson solution should have zero mean."""
        rhs = _eigenfunction_fft(NY, NX, DX, DY, kx=1, ky=1)
        psi = solve_poisson_fft(rhs, DX, DY)
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros((NY, NX))
        psi = solve_poisson_fft(rhs, DX, DY)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# FFT (Periodic) solvers — 1D
# ---------------------------------------------------------------------------


class TestFFTSolvers1D:
    """Periodic spectral solvers via FFT (1D)."""

    def test_helmholtz_eigenfunction_recovery(self):
        N = 32
        dx = 1.0
        k = 3
        x = jnp.arange(N)
        psi_exact = jnp.cos(2 * jnp.pi * k * x / N)
        eig = fft_eigenvalues(N, dx)[k]
        lambda_ = 0.5
        rhs = (eig - lambda_) * psi_exact
        psi_got = solve_helmholtz_fft_1d(rhs, dx, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_poisson_zero_mean(self):
        N = 32
        dx = 1.0
        x = jnp.arange(N)
        rhs = jnp.cos(2 * jnp.pi * x / N)
        psi = solve_poisson_fft_1d(rhs, dx)
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros(32)
        psi = solve_poisson_fft_1d(rhs, dx=1.0)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Layer 1 classes
# ---------------------------------------------------------------------------


class TestDirichletHelmholtzSolver2D:
    """DirichletHelmholtzSolver2D wraps solve_helmholtz_dst."""

    def test_callable(self):
        solver = DirichletHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        rhs = _eigenfunction_dst(NY, NX, DX, DY)
        eigx = dst1_eigenvalues(NX, DX)[0]
        eigy = dst1_eigenvalues(NY, DY)[0]
        rhs_scaled = (eigx + eigy) * rhs
        psi = solver(rhs_scaled)
        assert jnp.allclose(psi, rhs, atol=1e-5)

    def test_with_alpha(self):
        solver = DirichletHelmholtzSolver2D(dx=DX, dy=DY, alpha=1.0)
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


class TestNeumannHelmholtzSolver2D:
    """NeumannHelmholtzSolver2D wraps solve_helmholtz_dct."""

    def test_callable(self):
        solver = NeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert jnp.allclose(psi, 0.0, atol=1e-10)

    def test_with_alpha(self):
        solver = NeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=1.5)
        psi_exact = _eigenfunction_dct(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct2_eigenvalues(NX, DX)[2]
        eigy = dct2_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - 1.5) * psi_exact
        psi_got = solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


# ---------------------------------------------------------------------------
# vmap compatibility
# ---------------------------------------------------------------------------


class TestVmapCompatibility:
    """Spectral solvers work with jax.vmap."""

    def test_vmap_solve_helmholtz_dst(self):
        import jax

        batch = jnp.stack(
            [_eigenfunction_dst(NY, NX, DX, DY, kx=i + 1) for i in range(3)]
        )
        solve_batch = jax.vmap(lambda rhs: solve_helmholtz_dst(rhs, DX, DY, 0.0))
        results = solve_batch(batch)
        assert results.shape == (3, NY, NX)

    def test_vmap_solve_helmholtz_fft(self):
        import jax

        batch = jnp.stack(
            [_eigenfunction_fft(NY, NX, DX, DY, kx=i + 1) for i in range(3)]
        )
        solve_batch = jax.vmap(lambda rhs: solve_helmholtz_fft(rhs, DX, DY, 0.0))
        results = solve_batch(batch)
        assert results.shape == (3, NY, NX)
