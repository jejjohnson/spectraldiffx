"""Tests for spectral elliptic solvers (Layer 0 functions + Layer 1 classes)."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spectraldiffx._src.fourier.eigenvalues import (
    dct1_eigenvalues,
    dct2_eigenvalues,
    dst1_eigenvalues,
    dst2_eigenvalues,
    fft_eigenvalues,
)
from spectraldiffx._src.fourier.solvers import (
    DirichletHelmholtzSolver2D,
    NeumannHelmholtzSolver2D,
    RegularNeumannHelmholtzSolver2D,
    StaggeredDirichletHelmholtzSolver2D,
    solve_helmholtz_dct,
    solve_helmholtz_dct1,
    solve_helmholtz_dct1_1d,
    solve_helmholtz_dct1_3d,
    solve_helmholtz_dct2,
    solve_helmholtz_dct2_1d,
    solve_helmholtz_dct2_3d,
    solve_helmholtz_dst,
    solve_helmholtz_dst1,
    solve_helmholtz_dst1_1d,
    solve_helmholtz_dst1_3d,
    solve_helmholtz_dst2,
    solve_helmholtz_dst2_1d,
    solve_helmholtz_dst2_3d,
    solve_helmholtz_fft,
    solve_helmholtz_fft_1d,
    solve_helmholtz_fft_3d,
    solve_poisson_dct,
    solve_poisson_dct1,
    solve_poisson_dct1_1d,
    solve_poisson_dst,
    solve_poisson_dst1_1d,
    solve_poisson_dst2,
    solve_poisson_dst2_1d,
    solve_poisson_fft,
    solve_poisson_fft_1d,
    solve_poisson_fft_3d,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX = 32
NY = 24
NZ = 8
DX = 1.0
DY = 1.0
DZ = 1.0


def _eigenfunction_dst1(Ny, Nx, dx, dy, kx=1, ky=1):
    """DST-I eigenfunction: sin(π(kx)(i+1)/(Nx+1)) * sin(π(ky)(j+1)/(Ny+1))."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    fx = jnp.sin(jnp.pi * kx * (i + 1) / (Nx + 1))
    fy = jnp.sin(jnp.pi * ky * (j + 1) / (Ny + 1))
    return fy[:, None] * fx[None, :]


def _eigenfunction_dst2(Ny, Nx, dx, dy, kx=1, ky=1):
    """DST-II eigenfunction: sin(π(kx)(2i+1)/(2Nx)) * sin(π(ky)(2j+1)/(2Ny))."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    fx = jnp.sin(jnp.pi * kx * (2 * i + 1) / (2 * Nx))
    fy = jnp.sin(jnp.pi * ky * (2 * j + 1) / (2 * Ny))
    return fy[:, None] * fx[None, :]


def _eigenfunction_dct1(Ny, Nx, dx, dy, kx=1, ky=1):
    """DCT-I eigenfunction: cos(πkx·i/(Nx-1)) * cos(πky·j/(Ny-1))."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    fx = jnp.cos(jnp.pi * kx * i / (Nx - 1))
    fy = jnp.cos(jnp.pi * ky * j / (Ny - 1))
    return fy[:, None] * fx[None, :]


def _eigenfunction_dct2(Ny, Nx, dx, dy, kx=1, ky=1):
    """DCT-II eigenfunction: cos(πkx(2i+1)/(2Nx)) * cos(πky(2j+1)/(2Ny))."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    fx = jnp.cos(jnp.pi * kx * (2 * i + 1) / (2 * Nx))
    fy = jnp.cos(jnp.pi * ky * (2 * j + 1) / (2 * Ny))
    return fy[:, None] * fx[None, :]


def _eigenfunction_fft(Ny, Nx, dx, dy, kx=1, ky=1):
    """Periodic eigenfunction: cos(2πkx·i/Nx) * cos(2πky·j/Ny)."""
    i = jnp.arange(Nx)
    j = jnp.arange(Ny)
    fx = jnp.cos(2 * jnp.pi * kx * i / Nx)
    fy = jnp.cos(2 * jnp.pi * ky * j / Ny)
    return fy[:, None] * fx[None, :]


# ---------------------------------------------------------------------------
# DST-I (Dirichlet, regular) solvers — 2D
# ---------------------------------------------------------------------------


class TestDSTSolvers:
    """Dirichlet spectral solvers via DST-I."""

    def test_helmholtz_eigenfunction_recovery(self):
        """Solving (∇² - λ)ψ = f for a known eigenfunction."""
        lambda_ = 1.0
        psi_exact = _eigenfunction_dst1(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dst1_eigenvalues(NX, DX)[1]  # kx=2 → index 1
        eigy = dst1_eigenvalues(NY, DY)[2]  # ky=3 → index 2
        eig = eigx + eigy
        rhs = (eig - lambda_) * psi_exact
        psi_got = solve_helmholtz_dst(rhs, DX, DY, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_poisson_eigenfunction_recovery(self):
        psi_exact = _eigenfunction_dst1(NY, NX, DX, DY, kx=1, ky=1)
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
        rhs = _eigenfunction_dst1(NY, NX, DX, DY)
        psi_poisson = solve_poisson_dst(rhs, DX, DY)
        psi_helm = solve_helmholtz_dst(rhs, DX, DY, lambda_=2.0)
        assert not jnp.allclose(psi_poisson, psi_helm, atol=1e-3)

    def test_alias_solve_helmholtz_dst1(self):
        """solve_helmholtz_dst1 is an alias for solve_helmholtz_dst."""
        assert solve_helmholtz_dst1 is solve_helmholtz_dst


# ---------------------------------------------------------------------------
# DST-II (Dirichlet, staggered) solvers — 2D
# ---------------------------------------------------------------------------


class TestDST2Solvers:
    """Dirichlet spectral solvers via DST-II (staggered grid)."""

    def test_helmholtz_eigenfunction_recovery(self):
        lambda_ = 1.0
        psi_exact = _eigenfunction_dst2(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dst2_eigenvalues(NX, DX)[1]  # kx=2 → index 1
        eigy = dst2_eigenvalues(NY, DY)[2]  # ky=3 → index 2
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_dst2(rhs, DX, DY, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_poisson_eigenfunction_recovery(self):
        psi_exact = _eigenfunction_dst2(NY, NX, DX, DY, kx=1, ky=1)
        eigx = dst2_eigenvalues(NX, DX)[0]
        eigy = dst2_eigenvalues(NY, DY)[0]
        rhs = (eigx + eigy) * psi_exact
        psi_got = solve_poisson_dst2(rhs, DX, DY)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros((NY, NX))
        psi = solve_poisson_dst2(rhs, DX, DY)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# DCT-I (Neumann, regular) solvers — 2D
# ---------------------------------------------------------------------------


class TestDCT1Solvers:
    """Neumann spectral solvers via DCT-I (regular grid)."""

    def test_helmholtz_eigenfunction_recovery(self):
        lambda_ = 1.5
        psi_exact = _eigenfunction_dct1(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct1_eigenvalues(NX, DX)[2]
        eigy = dct1_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_dct1(rhs, DX, DY, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_poisson_zero_mean_gauge(self):
        rhs = _eigenfunction_dct1(NY, NX, DX, DY, kx=1, ky=1)
        psi = solve_poisson_dct1(rhs, DX, DY)
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros((NY, NX))
        psi = solve_poisson_dct1(rhs, DX, DY)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# DCT-II (Neumann, staggered) solvers — 2D
# ---------------------------------------------------------------------------


class TestDCTSolvers:
    """Neumann spectral solvers via DCT-II."""

    def test_helmholtz_eigenfunction_recovery(self):
        lambda_ = 1.5
        psi_exact = _eigenfunction_dct2(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct2_eigenvalues(NX, DX)[2]
        eigy = dct2_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_dct(rhs, DX, DY, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_poisson_zero_mean_gauge(self):
        """Poisson solution with Neumann BCs should have zero mean."""
        rhs = _eigenfunction_dct2(NY, NX, DX, DY, kx=1, ky=1)
        psi = solve_poisson_dct(rhs, DX, DY)
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros((NY, NX))
        psi = solve_poisson_dct(rhs, DX, DY)
        assert jnp.allclose(psi, 0.0, atol=1e-10)

    def test_alias_solve_helmholtz_dct2(self):
        """solve_helmholtz_dct2 is an alias for solve_helmholtz_dct."""
        assert solve_helmholtz_dct2 is solve_helmholtz_dct


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
# DST-I (Dirichlet, regular) solvers — 1D
# ---------------------------------------------------------------------------


class TestDST1Solvers1D:
    """DST-I 1D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        N = 32
        dx = 1.0
        k = 3  # mode index (0-based), maps to eigenvalue index 2
        x = jnp.arange(N)
        psi_exact = jnp.sin(jnp.pi * k * (x + 1) / (N + 1))
        eig = dst1_eigenvalues(N, dx)[k - 1]
        lambda_ = 0.5
        rhs = (eig - lambda_) * psi_exact
        psi_got = solve_helmholtz_dst1_1d(rhs, dx, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros(32)
        psi = solve_poisson_dst1_1d(rhs, dx=1.0)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# DST-II (Dirichlet, staggered) solvers — 1D
# ---------------------------------------------------------------------------


class TestDST2Solvers1D:
    """DST-II 1D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        N = 32
        dx = 1.0
        k = 3
        x = jnp.arange(N)
        psi_exact = jnp.sin(jnp.pi * k * (2 * x + 1) / (2 * N))
        eig = dst2_eigenvalues(N, dx)[k - 1]
        lambda_ = 0.5
        rhs = (eig - lambda_) * psi_exact
        psi_got = solve_helmholtz_dst2_1d(rhs, dx, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros(32)
        psi = solve_poisson_dst2_1d(rhs, dx=1.0)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# DCT-I (Neumann, regular) solvers — 1D
# ---------------------------------------------------------------------------


class TestDCT1Solvers1D:
    """DCT-I 1D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        N = 32
        dx = 1.0
        k = 3
        x = jnp.arange(N)
        psi_exact = jnp.cos(jnp.pi * k * x / (N - 1))
        eig = dct1_eigenvalues(N, dx)[k]
        lambda_ = 0.5
        rhs = (eig - lambda_) * psi_exact
        psi_got = solve_helmholtz_dct1_1d(rhs, dx, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_poisson_zero_mean(self):
        N = 32
        dx = 1.0
        x = jnp.arange(N)
        rhs = jnp.cos(jnp.pi * x / (N - 1))
        psi = solve_poisson_dct1_1d(rhs, dx)
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros(32)
        psi = solve_poisson_dct1_1d(rhs, dx=1.0)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# DCT-II (Neumann, staggered) solvers — 1D
# ---------------------------------------------------------------------------


class TestDCT2Solvers1D:
    """DCT-II 1D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        N = 32
        dx = 1.0
        k = 3
        x = jnp.arange(N)
        psi_exact = jnp.cos(jnp.pi * k * (2 * x + 1) / (2 * N))
        eig = dct2_eigenvalues(N, dx)[k]
        lambda_ = 0.5
        rhs = (eig - lambda_) * psi_exact
        psi_got = solve_helmholtz_dct2_1d(rhs, dx, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-5)

    def test_poisson_zero_mean(self):
        N = 32
        dx = 1.0
        x = jnp.arange(N)
        rhs = jnp.cos(jnp.pi * (2 * x + 1) / (2 * N))
        psi = solve_helmholtz_dct2_1d(rhs, dx, lambda_=0.0)
        assert jnp.abs(jnp.mean(psi)) < 1e-6


# ---------------------------------------------------------------------------
# 3D solvers
# ---------------------------------------------------------------------------


class TestFFTSolvers3D:
    """Periodic spectral solvers via FFT (3D)."""

    def test_helmholtz_eigenfunction_recovery(self):
        kx, ky, kz = 2, 1, 1
        x = jnp.arange(NX)
        y = jnp.arange(NY)
        z = jnp.arange(NZ)
        psi_exact = (
            jnp.cos(2 * jnp.pi * kz * z / NZ)[:, None, None]
            * jnp.cos(2 * jnp.pi * ky * y / NY)[None, :, None]
            * jnp.cos(2 * jnp.pi * kx * x / NX)[None, None, :]
        )
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = fft_eigenvalues(NY, DY)[ky]
        eigz = fft_eigenvalues(NZ, DZ)[kz]
        lambda_ = 0.5
        rhs = (eigx + eigy + eigz - lambda_) * psi_exact
        psi_got = solve_helmholtz_fft_3d(rhs, DX, DY, DZ, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_poisson_zero_mean(self):
        rhs = jnp.zeros((NZ, NY, NX))
        psi = solve_poisson_fft_3d(rhs, DX, DY, DZ)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


class TestDST1Solvers3D:
    """DST-I 3D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        kx, ky, kz = 2, 1, 1
        x = jnp.arange(NX)
        y = jnp.arange(NY)
        z = jnp.arange(NZ)
        psi_exact = (
            jnp.sin(jnp.pi * kz * (z + 1) / (NZ + 1))[:, None, None]
            * jnp.sin(jnp.pi * ky * (y + 1) / (NY + 1))[None, :, None]
            * jnp.sin(jnp.pi * kx * (x + 1) / (NX + 1))[None, None, :]
        )
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        eigz = dst1_eigenvalues(NZ, DZ)[kz - 1]
        lambda_ = 0.5
        rhs = (eigx + eigy + eigz - lambda_) * psi_exact
        psi_got = solve_helmholtz_dst1_3d(rhs, DX, DY, DZ, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


class TestDST2Solvers3D:
    """DST-II 3D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        kx, ky, kz = 2, 1, 1
        x = jnp.arange(NX)
        y = jnp.arange(NY)
        z = jnp.arange(NZ)
        psi_exact = (
            jnp.sin(jnp.pi * kz * (2 * z + 1) / (2 * NZ))[:, None, None]
            * jnp.sin(jnp.pi * ky * (2 * y + 1) / (2 * NY))[None, :, None]
            * jnp.sin(jnp.pi * kx * (2 * x + 1) / (2 * NX))[None, None, :]
        )
        eigx = dst2_eigenvalues(NX, DX)[kx - 1]
        eigy = dst2_eigenvalues(NY, DY)[ky - 1]
        eigz = dst2_eigenvalues(NZ, DZ)[kz - 1]
        lambda_ = 0.5
        rhs = (eigx + eigy + eigz - lambda_) * psi_exact
        psi_got = solve_helmholtz_dst2_3d(rhs, DX, DY, DZ, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


class TestDCT1Solvers3D:
    """DCT-I 3D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        kx, ky, kz = 2, 1, 1
        x = jnp.arange(NX)
        y = jnp.arange(NY)
        z = jnp.arange(NZ)
        psi_exact = (
            jnp.cos(jnp.pi * kz * z / (NZ - 1))[:, None, None]
            * jnp.cos(jnp.pi * ky * y / (NY - 1))[None, :, None]
            * jnp.cos(jnp.pi * kx * x / (NX - 1))[None, None, :]
        )
        eigx = dct1_eigenvalues(NX, DX)[kx]
        eigy = dct1_eigenvalues(NY, DY)[ky]
        eigz = dct1_eigenvalues(NZ, DZ)[kz]
        lambda_ = 0.5
        rhs = (eigx + eigy + eigz - lambda_) * psi_exact
        psi_got = solve_helmholtz_dct1_3d(rhs, DX, DY, DZ, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


class TestDCT2Solvers3D:
    """DCT-II 3D solvers."""

    def test_helmholtz_eigenfunction_recovery(self):
        kx, ky, kz = 2, 1, 1
        x = jnp.arange(NX)
        y = jnp.arange(NY)
        z = jnp.arange(NZ)
        psi_exact = (
            jnp.cos(jnp.pi * kz * (2 * z + 1) / (2 * NZ))[:, None, None]
            * jnp.cos(jnp.pi * ky * (2 * y + 1) / (2 * NY))[None, :, None]
            * jnp.cos(jnp.pi * kx * (2 * x + 1) / (2 * NX))[None, None, :]
        )
        eigx = dct2_eigenvalues(NX, DX)[kx]
        eigy = dct2_eigenvalues(NY, DY)[ky]
        eigz = dct2_eigenvalues(NZ, DZ)[kz]
        lambda_ = 0.5
        rhs = (eigx + eigy + eigz - lambda_) * psi_exact
        psi_got = solve_helmholtz_dct2_3d(rhs, DX, DY, DZ, lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


# ---------------------------------------------------------------------------
# Layer 1 classes
# ---------------------------------------------------------------------------


class TestDirichletHelmholtzSolver2D:
    """DirichletHelmholtzSolver2D wraps solve_helmholtz_dst."""

    def test_callable(self):
        solver = DirichletHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        rhs = _eigenfunction_dst1(NY, NX, DX, DY)
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

    def test_filter_jit(self):
        solver = DirichletHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        jit_solver = eqx.filter_jit(solver)
        rhs = _eigenfunction_dst1(NY, NX, DX, DY)
        eigx = dst1_eigenvalues(NX, DX)[0]
        eigy = dst1_eigenvalues(NY, DY)[0]
        rhs_scaled = (eigx + eigy) * rhs
        psi = jit_solver(rhs_scaled)
        assert jnp.allclose(psi, rhs, atol=1e-5)


class TestNeumannHelmholtzSolver2D:
    """NeumannHelmholtzSolver2D wraps solve_helmholtz_dct."""

    def test_callable(self):
        solver = NeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert jnp.allclose(psi, 0.0, atol=1e-10)

    def test_with_alpha(self):
        solver = NeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=1.5)
        psi_exact = _eigenfunction_dct2(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct2_eigenvalues(NX, DX)[2]
        eigy = dct2_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - 1.5) * psi_exact
        psi_got = solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_filter_jit(self):
        solver = NeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=1.5)
        jit_solver = eqx.filter_jit(solver)
        psi_exact = _eigenfunction_dct2(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct2_eigenvalues(NX, DX)[2]
        eigy = dct2_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - 1.5) * psi_exact
        psi_got = jit_solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


class TestStaggeredDirichletHelmholtzSolver2D:
    """StaggeredDirichletHelmholtzSolver2D wraps solve_helmholtz_dst2."""

    def test_callable(self):
        solver = StaggeredDirichletHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        psi_exact = _eigenfunction_dst2(NY, NX, DX, DY, kx=1, ky=1)
        eigx = dst2_eigenvalues(NX, DX)[0]
        eigy = dst2_eigenvalues(NY, DY)[0]
        rhs = (eigx + eigy) * psi_exact
        psi = solver(rhs)
        assert jnp.allclose(psi, psi_exact, atol=1e-5)

    def test_with_alpha(self):
        solver = StaggeredDirichletHelmholtzSolver2D(dx=DX, dy=DY, alpha=1.0)
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert jnp.allclose(psi, 0.0, atol=1e-10)

    def test_filter_jit(self):
        solver = StaggeredDirichletHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        jit_solver = eqx.filter_jit(solver)
        psi_exact = _eigenfunction_dst2(NY, NX, DX, DY, kx=1, ky=1)
        eigx = dst2_eigenvalues(NX, DX)[0]
        eigy = dst2_eigenvalues(NY, DY)[0]
        rhs = (eigx + eigy) * psi_exact
        psi = jit_solver(rhs)
        assert jnp.allclose(psi, psi_exact, atol=1e-5)


class TestRegularNeumannHelmholtzSolver2D:
    """RegularNeumannHelmholtzSolver2D wraps solve_helmholtz_dct1."""

    def test_callable(self):
        solver = RegularNeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=0.0)
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert jnp.allclose(psi, 0.0, atol=1e-10)

    def test_with_alpha(self):
        solver = RegularNeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=1.5)
        psi_exact = _eigenfunction_dct1(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct1_eigenvalues(NX, DX)[2]
        eigy = dct1_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - 1.5) * psi_exact
        psi_got = solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_filter_jit(self):
        solver = RegularNeumannHelmholtzSolver2D(dx=DX, dy=DY, alpha=1.5)
        jit_solver = eqx.filter_jit(solver)
        psi_exact = _eigenfunction_dct1(NY, NX, DX, DY, kx=2, ky=3)
        eigx = dct1_eigenvalues(NX, DX)[2]
        eigy = dct1_eigenvalues(NY, DY)[3]
        rhs = (eigx + eigy - 1.5) * psi_exact
        psi_got = jit_solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


# ---------------------------------------------------------------------------
# vmap compatibility
# ---------------------------------------------------------------------------


class TestVmapCompatibility:
    """Spectral solvers work with jax.vmap."""

    def test_vmap_solve_helmholtz_dst(self):
        batch = jnp.stack(
            [_eigenfunction_dst1(NY, NX, DX, DY, kx=i + 1) for i in range(3)]
        )
        solve_batch = jax.vmap(lambda rhs: solve_helmholtz_dst(rhs, DX, DY, 0.0))
        results = solve_batch(batch)
        assert results.shape == (3, NY, NX)

    def test_vmap_solve_helmholtz_fft(self):
        batch = jnp.stack(
            [_eigenfunction_fft(NY, NX, DX, DY, kx=i + 1) for i in range(3)]
        )
        solve_batch = jax.vmap(lambda rhs: solve_helmholtz_fft(rhs, DX, DY, 0.0))
        results = solve_batch(batch)
        assert results.shape == (3, NY, NX)

    def test_vmap_solve_helmholtz_dst2(self):
        batch = jnp.stack(
            [_eigenfunction_dst2(NY, NX, DX, DY, kx=i + 1) for i in range(3)]
        )
        solve_batch = jax.vmap(lambda rhs: solve_helmholtz_dst2(rhs, DX, DY, 0.0))
        results = solve_batch(batch)
        assert results.shape == (3, NY, NX)

    def test_vmap_solve_helmholtz_dct1(self):
        batch = jnp.stack(
            [_eigenfunction_dct1(NY, NX, DX, DY, kx=i + 1) for i in range(3)]
        )
        solve_batch = jax.vmap(lambda rhs: solve_helmholtz_dct1(rhs, DX, DY, 0.0))
        results = solve_batch(batch)
        assert results.shape == (3, NY, NX)
