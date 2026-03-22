"""Tests for mixed per-axis boundary condition solvers (solve_helmholtz_2d)."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from spectraldiffx._src.fourier.eigenvalues import (
    dct1_eigenvalues,
    dct2_eigenvalues,
    dct4_eigenvalues,
    dst1_eigenvalues,
    dst2_eigenvalues,
    dst3_eigenvalues,
    dst4_eigenvalues,
    fft_eigenvalues,
)
from spectraldiffx._src.fourier.solvers import (
    MixedBCHelmholtzSolver2D,
    solve_helmholtz_2d,
    solve_helmholtz_dct,
    solve_helmholtz_dct1,
    solve_helmholtz_dst,
    solve_helmholtz_dst2,
    solve_helmholtz_fft,
    solve_poisson_2d,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX = 32
NY = 24
DX = 1.0
DY = 1.0


# ---------------------------------------------------------------------------
# Eigenfunction helpers (per-axis)
# ---------------------------------------------------------------------------


def _ef_dst1_1d(N, k):
    """DST-I eigenfunction: sin(pi*k*(i+1)/(N+1))."""
    i = jnp.arange(N)
    return jnp.sin(jnp.pi * k * (i + 1) / (N + 1))


def _ef_dst2_1d(N, k):
    """DST-II eigenfunction: sin(pi*k*(2i+1)/(2N))."""
    i = jnp.arange(N)
    return jnp.sin(jnp.pi * k * (2 * i + 1) / (2 * N))


def _ef_dct1_1d(N, k):
    """DCT-I eigenfunction: cos(pi*k*i/(N-1))."""
    i = jnp.arange(N)
    return jnp.cos(jnp.pi * k * i / (N - 1))


def _ef_dct2_1d(N, k):
    """DCT-II eigenfunction: cos(pi*k*(2i+1)/(2N))."""
    i = jnp.arange(N)
    return jnp.cos(jnp.pi * k * (2 * i + 1) / (2 * N))


def _ef_fft_1d(N, k):
    """FFT eigenfunction: cos(2*pi*k*i/N)."""
    i = jnp.arange(N)
    return jnp.cos(2 * jnp.pi * k * i / N)


def _ef_dst3_1d(N, k):
    """DST-III eigenfunction: sin(pi*(k+1)*(2i+1)/(2N)).

    For DST-III (Dirichlet-left + Neumann-right, regular grid).
    """
    i = jnp.arange(N)
    return jnp.sin(jnp.pi * (k + 1) * (2 * i + 1) / (2 * N))


def _ef_dst4_1d(N, k):
    """DST-IV eigenfunction: sin(pi*(2k+1)*(2i+1)/(4N)).

    For DST-IV (Dirichlet-left + Neumann-right, staggered grid).
    """
    i = jnp.arange(N)
    return jnp.sin(jnp.pi * (2 * k + 1) * (2 * i + 1) / (4 * N))


def _ef_dct4_1d(N, k):
    """DCT-IV eigenfunction: cos(pi*(2k+1)*(2i+1)/(4N)).

    For DCT-IV (Neumann-left + Dirichlet-right, staggered grid).
    """
    i = jnp.arange(N)
    return jnp.cos(jnp.pi * (2 * k + 1) * (2 * i + 1) / (4 * N))


def _ef_2d(fx, fy):
    """Build 2D eigenfunction from 1D factors."""
    return fy[:, None] * fx[None, :]


# ---------------------------------------------------------------------------
# Same-BC validation: solve_helmholtz_2d must match existing solvers exactly
# ---------------------------------------------------------------------------


class TestSameBCValidation:
    """When both axes use the same BC, solve_helmholtz_2d must match
    the existing monolithic same-BC solvers exactly."""

    def test_periodic_matches_fft(self):
        rhs = _ef_2d(_ef_fft_1d(NX, 2), _ef_fft_1d(NY, 3))
        eigx = fft_eigenvalues(NX, DX)[2]
        eigy = fft_eigenvalues(NY, DY)[3]
        rhs_scaled = (eigx + eigy - 1.0) * rhs
        psi_ref = solve_helmholtz_fft(rhs_scaled, DX, DY, 1.0)
        psi_got = solve_helmholtz_2d(rhs_scaled, DX, DY, "periodic", "periodic", 1.0)
        assert jnp.allclose(psi_got, psi_ref, atol=1e-5)

    def test_dirichlet_matches_dst1(self):
        rhs = _ef_2d(_ef_dst1_1d(NX, 2), _ef_dst1_1d(NY, 3))
        eigx = dst1_eigenvalues(NX, DX)[1]
        eigy = dst1_eigenvalues(NY, DY)[2]
        rhs_scaled = (eigx + eigy - 1.0) * rhs
        psi_ref = solve_helmholtz_dst(rhs_scaled, DX, DY, 1.0)
        psi_got = solve_helmholtz_2d(rhs_scaled, DX, DY, "dirichlet", "dirichlet", 1.0)
        assert jnp.allclose(psi_got, psi_ref, atol=1e-5)

    def test_dirichlet_stag_matches_dst2(self):
        rhs = _ef_2d(_ef_dst2_1d(NX, 2), _ef_dst2_1d(NY, 3))
        eigx = dst2_eigenvalues(NX, DX)[1]
        eigy = dst2_eigenvalues(NY, DY)[2]
        rhs_scaled = (eigx + eigy - 1.0) * rhs
        psi_ref = solve_helmholtz_dst2(rhs_scaled, DX, DY, 1.0)
        psi_got = solve_helmholtz_2d(
            rhs_scaled,
            DX,
            DY,
            "dirichlet_stag",
            "dirichlet_stag",
            1.0,
        )
        assert jnp.allclose(psi_got, psi_ref, atol=1e-5)

    def test_neumann_stag_matches_dct2(self):
        rhs = _ef_2d(_ef_dct2_1d(NX, 2), _ef_dct2_1d(NY, 3))
        eigx = dct2_eigenvalues(NX, DX)[2]
        eigy = dct2_eigenvalues(NY, DY)[3]
        rhs_scaled = (eigx + eigy - 1.5) * rhs
        psi_ref = solve_helmholtz_dct(rhs_scaled, DX, DY, 1.5)
        psi_got = solve_helmholtz_2d(
            rhs_scaled,
            DX,
            DY,
            "neumann_stag",
            "neumann_stag",
            1.5,
        )
        assert jnp.allclose(psi_got, psi_ref, atol=1e-5)

    def test_neumann_matches_dct1(self):
        rhs = _ef_2d(_ef_dct1_1d(NX, 2), _ef_dct1_1d(NY, 3))
        eigx = dct1_eigenvalues(NX, DX)[2]
        eigy = dct1_eigenvalues(NY, DY)[3]
        rhs_scaled = (eigx + eigy - 1.5) * rhs
        psi_ref = solve_helmholtz_dct1(rhs_scaled, DX, DY, 1.5)
        psi_got = solve_helmholtz_2d(rhs_scaled, DX, DY, "neumann", "neumann", 1.5)
        assert jnp.allclose(psi_got, psi_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Mixed real-real BC eigenfunction recovery
# ---------------------------------------------------------------------------


class TestMixedRealRealBC:
    """Different DST/DCT types on x and y axes."""

    def test_dirichlet_x_neumann_stag_y(self):
        """Dirichlet (DST-I) in x, Neumann staggered (DCT-II) in y."""
        kx, ky = 2, 3
        fx = _ef_dst1_1d(NX, kx)
        fy = _ef_dct2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = dct2_eigenvalues(NY, DY)[ky]
        lambda_ = 1.0
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(rhs, DX, DY, "dirichlet", "neumann_stag", lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_neumann_stag_x_dirichlet_y(self):
        """Neumann staggered (DCT-II) in x, Dirichlet (DST-I) in y."""
        kx, ky = 3, 2
        fx = _ef_dct2_1d(NX, kx)
        fy = _ef_dst1_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dct2_eigenvalues(NX, DX)[kx]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        lambda_ = 0.5
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(rhs, DX, DY, "neumann_stag", "dirichlet", lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_dirichlet_stag_x_neumann_y(self):
        """Dirichlet staggered (DST-II) in x, Neumann regular (DCT-I) in y."""
        kx, ky = 2, 3
        fx = _ef_dst2_1d(NX, kx)
        fy = _ef_dct1_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dst2_eigenvalues(NX, DX)[kx - 1]
        eigy = dct1_eigenvalues(NY, DY)[ky]
        lambda_ = 1.0
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            "dirichlet_stag",
            "neumann",
            lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_dirichlet_x_dirichlet_stag_y(self):
        """DST-I in x, DST-II in y (different DST types)."""
        kx, ky = 2, 3
        fx = _ef_dst1_1d(NX, kx)
        fy = _ef_dst2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = dst2_eigenvalues(NY, DY)[ky - 1]
        lambda_ = 0.5
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            "dirichlet",
            "dirichlet_stag",
            lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_zero_rhs_returns_zero(self):
        rhs = jnp.zeros((NY, NX))
        psi = solve_helmholtz_2d(rhs, DX, DY, "dirichlet", "neumann_stag", 1.0)
        assert jnp.allclose(psi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Mixed periodic + real BC eigenfunction recovery
# ---------------------------------------------------------------------------


class TestMixedPeriodicRealBC:
    """One periodic axis (FFT) and one non-periodic axis (DST/DCT)."""

    def test_periodic_x_dirichlet_y(self):
        """Periodic (FFT) in x, Dirichlet regular (DST-I) in y."""
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dst1_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        lambda_ = 0.5
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(rhs, DX, DY, "periodic", "dirichlet", lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_dirichlet_x_periodic_y(self):
        """Dirichlet regular (DST-I) in x, periodic (FFT) in y."""
        kx, ky = 2, 3
        fx = _ef_dst1_1d(NX, kx)
        fy = _ef_fft_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = fft_eigenvalues(NY, DY)[ky]
        lambda_ = 1.0
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(rhs, DX, DY, "dirichlet", "periodic", lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_periodic_x_neumann_stag_y(self):
        """Periodic (FFT) in x, Neumann staggered (DCT-II) in y."""
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dct2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dct2_eigenvalues(NY, DY)[ky]
        lambda_ = 1.0
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            "periodic",
            "neumann_stag",
            lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_neumann_stag_x_periodic_y(self):
        """Neumann staggered (DCT-II) in x, periodic (FFT) in y."""
        kx, ky = 3, 2
        fx = _ef_dct2_1d(NX, kx)
        fy = _ef_fft_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dct2_eigenvalues(NX, DX)[kx]
        eigy = fft_eigenvalues(NY, DY)[ky]
        lambda_ = 0.5
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            "neumann_stag",
            "periodic",
            lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_periodic_x_dirichlet_stag_y(self):
        """Periodic (FFT) in x, Dirichlet staggered (DST-II) in y."""
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dst2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dst2_eigenvalues(NY, DY)[ky - 1]
        lambda_ = 0.5
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            "periodic",
            "dirichlet_stag",
            lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_periodic_x_neumann_y(self):
        """Periodic (FFT) in x, Neumann regular (DCT-I) in y."""
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dct1_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dct1_eigenvalues(NY, DY)[ky]
        lambda_ = 0.5
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(rhs, DX, DY, "periodic", "neumann", lambda_)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


# ---------------------------------------------------------------------------
# Mixed left/right BCs on same axis (DST-III/IV, DCT-III/IV)
# ---------------------------------------------------------------------------


class TestMixedLeftRightBC:
    """Different BCs on left vs right of the same axis (tuples)."""

    def test_dirichlet_stag_neumann_stag_x(self):
        """Dirichlet+Neumann staggered in x (DST-IV), Dirichlet stag in y."""
        kx, ky = 2, 3
        fx = _ef_dst4_1d(NX, kx)
        fy = _ef_dst2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dst4_eigenvalues(NX, DX)[kx]
        eigy = dst2_eigenvalues(NY, DY)[ky - 1]
        lambda_ = 1.0
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            bc_x=("dirichlet_stag", "neumann_stag"),
            bc_y="dirichlet_stag",
            lambda_=lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_neumann_stag_dirichlet_stag_y(self):
        """Neumann+Dirichlet staggered in y (DCT-IV), periodic in x."""
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dct4_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dct4_eigenvalues(NY, DY)[ky]
        lambda_ = 0.5
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            bc_x="periodic",
            bc_y=("neumann_stag", "dirichlet_stag"),
            lambda_=lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_dirichlet_neumann_regular_x(self):
        """Dirichlet+Neumann regular in x (DST-III), Neumann stag in y.

        Constructs the eigenfunction numerically via the inverse transform
        (the DST-III eigenfunctions don't have a simple closed-form formula
        on the regular grid with mixed left/right BCs).
        """
        from spectraldiffx._src.fourier.transforms import idstn as _idstn

        kx, ky = 2, 3
        # Build DST-III eigenfunction numerically: IDST-III of a delta
        delta_x = jnp.zeros(NX).at[kx].set(1.0)
        fx = _idstn(delta_x, type=3, axes=[0])
        fy = _ef_dct2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dst3_eigenvalues(NX, DX)[kx]
        eigy = dct2_eigenvalues(NY, DY)[ky]
        lambda_ = 1.0
        rhs = (eigx + eigy - lambda_) * psi_exact
        psi_got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            bc_x=("dirichlet", "neumann"),
            bc_y="neumann_stag",
            lambda_=lambda_,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


# ---------------------------------------------------------------------------
# Null-mode handling
# ---------------------------------------------------------------------------


class TestNullModeHandling:
    """Poisson problems with Neumann/periodic BCs (null-mode singularity)."""

    def test_periodic_periodic_poisson_zero_mean(self):
        """Doubly periodic Poisson solution has zero mean."""
        rhs = _ef_2d(_ef_fft_1d(NX, 1), _ef_fft_1d(NY, 1))
        psi = solve_poisson_2d(rhs, DX, DY, "periodic", "periodic")
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_neumann_stag_neumann_stag_poisson_zero_mean(self):
        """Doubly Neumann Poisson solution has zero mean."""
        rhs = _ef_2d(_ef_dct2_1d(NX, 1), _ef_dct2_1d(NY, 1))
        psi = solve_poisson_2d(rhs, DX, DY, "neumann_stag", "neumann_stag")
        assert jnp.abs(jnp.mean(psi)) < 1e-6

    def test_periodic_dirichlet_no_null_mode(self):
        """Periodic x + Dirichlet y: no null mode, Poisson is well-posed."""
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dst1_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        rhs = (eigx + eigy) * psi_exact
        psi_got = solve_poisson_2d(rhs, DX, DY, "periodic", "dirichlet")
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class TestSolvePoisson2D:
    """solve_poisson_2d is a convenience wrapper for solve_helmholtz_2d."""

    def test_matches_helmholtz_lambda_zero(self):
        rhs = _ef_2d(_ef_dst1_1d(NX, 2), _ef_dct2_1d(NY, 3))
        psi_h = solve_helmholtz_2d(rhs, DX, DY, "dirichlet", "neumann_stag", 0.0)
        psi_p = solve_poisson_2d(rhs, DX, DY, "dirichlet", "neumann_stag")
        assert jnp.allclose(psi_h, psi_p, atol=1e-10)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_bc_raises(self):
        rhs = jnp.zeros((NY, NX))
        with pytest.raises(ValueError, match="Unsupported boundary condition"):
            solve_helmholtz_2d(rhs, DX, DY, "invalid", "periodic")


# ---------------------------------------------------------------------------
# Module class
# ---------------------------------------------------------------------------


class TestMixedBCHelmholtzSolver2D:
    """MixedBCHelmholtzSolver2D wraps solve_helmholtz_2d."""

    def test_callable(self):
        solver = MixedBCHelmholtzSolver2D(
            dx=DX,
            dy=DY,
            bc_x="periodic",
            bc_y="dirichlet",
        )
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dst1_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        rhs = (eigx + eigy) * psi_exact
        psi_got = solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_with_alpha(self):
        solver = MixedBCHelmholtzSolver2D(
            dx=DX,
            dy=DY,
            bc_x="dirichlet",
            bc_y="neumann_stag",
            alpha=1.0,
        )
        kx, ky = 2, 3
        fx = _ef_dst1_1d(NX, kx)
        fy = _ef_dct2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = dct2_eigenvalues(NY, DY)[ky]
        rhs = (eigx + eigy - 1.0) * psi_exact
        psi_got = solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_filter_jit(self):
        solver = MixedBCHelmholtzSolver2D(
            dx=DX,
            dy=DY,
            bc_x="periodic",
            bc_y="dirichlet_stag",
        )
        jit_solver = eqx.filter_jit(solver)
        kx, ky = 2, 3
        fx = _ef_fft_1d(NX, kx)
        fy = _ef_dst2_1d(NY, ky)
        psi_exact = _ef_2d(fx, fy)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dst2_eigenvalues(NY, DY)[ky - 1]
        rhs = (eigx + eigy) * psi_exact
        psi_got = jit_solver(rhs)
        assert jnp.allclose(psi_got, psi_exact, atol=1e-4)

    def test_with_tuple_bc(self):
        solver = MixedBCHelmholtzSolver2D(
            dx=DX,
            dy=DY,
            bc_x=("dirichlet_stag", "neumann_stag"),
            bc_y="dirichlet_stag",
            alpha=1.0,
        )
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert jnp.allclose(psi, 0.0, atol=1e-10)
