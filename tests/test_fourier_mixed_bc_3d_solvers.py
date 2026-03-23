"""Tests for 3D mixed per-axis boundary condition solvers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spectraldiffx._src.fourier.eigenvalues import (
    dct1_eigenvalues,
    dct2_eigenvalues,
    dct4_eigenvalues,
    dst1_eigenvalues,
    dst2_eigenvalues,
    dst4_eigenvalues,
    fft_eigenvalues,
)
from spectraldiffx._src.fourier.solvers import (
    MixedBCHelmholtzSolver3D,
    solve_helmholtz_3d,
    solve_helmholtz_dct1_3d,
    solve_helmholtz_dct2_3d,
    solve_helmholtz_dst1_3d,
    solve_helmholtz_dst2_3d,
    solve_helmholtz_fft_3d,
    solve_poisson_3d,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX = 16
NY = 12
NZ = 10
DX = 1.0
DY = 1.0
DZ = 1.0


# ---------------------------------------------------------------------------
# Eigenfunction helpers (1D)
# ---------------------------------------------------------------------------


def _ef_dst1(N, k):
    """DST-I eigenfunction: sin(pi*k*(i+1)/(N+1))."""
    i = jnp.arange(N)
    return jnp.sin(jnp.pi * k * (i + 1) / (N + 1))


def _ef_dst2(N, k):
    """DST-II eigenfunction: sin(pi*k*(2i+1)/(2N))."""
    i = jnp.arange(N)
    return jnp.sin(jnp.pi * k * (2 * i + 1) / (2 * N))


def _ef_dct1(N, k):
    """DCT-I eigenfunction: cos(pi*k*i/(N-1))."""
    i = jnp.arange(N)
    return jnp.cos(jnp.pi * k * i / (N - 1))


def _ef_dct2(N, k):
    """DCT-II eigenfunction: cos(pi*k*(2i+1)/(2N))."""
    i = jnp.arange(N)
    return jnp.cos(jnp.pi * k * (2 * i + 1) / (2 * N))


def _ef_fft(N, k):
    """FFT eigenfunction: cos(2*pi*k*i/N)."""
    i = jnp.arange(N)
    return jnp.cos(2 * jnp.pi * k * i / N)


def _ef_dst4(N, k):
    """DST-IV eigenfunction: sin(pi*(2k+1)*(2i+1)/(4N))."""
    i = jnp.arange(N)
    return jnp.sin(jnp.pi * (2 * k + 1) * (2 * i + 1) / (4 * N))


def _ef_dct4(N, k):
    """DCT-IV eigenfunction: cos(pi*(2k+1)*(2i+1)/(4N))."""
    i = jnp.arange(N)
    return jnp.cos(jnp.pi * (2 * k + 1) * (2 * i + 1) / (4 * N))


def _ef_3d(fx, fy, fz):
    """Build 3D eigenfunction from 1D factors."""
    return fz[:, None, None] * fy[None, :, None] * fx[None, None, :]


# ---------------------------------------------------------------------------
# BC config helpers for parametrized tests
# ---------------------------------------------------------------------------

# (bc_string, eigenvalue_fn, eigenfunction_fn, mode_index, has_null)
_SAME_BC_CONFIGS = [
    ("periodic", fft_eigenvalues, _ef_fft, 2, True),
    ("dirichlet", dst1_eigenvalues, _ef_dst1, 2, False),
    ("dirichlet_stag", dst2_eigenvalues, _ef_dst2, 2, False),
    ("neumann", dct1_eigenvalues, _ef_dct1, 2, True),
    ("neumann_stag", dct2_eigenvalues, _ef_dct2, 2, True),
]

# (bc_string, eigenvalue_fn, eigenfunction_fn, mode_index)
# No null-mode BCs only, for eigenfunction recovery
_NO_NULL_CONFIGS = [
    ("dirichlet", dst1_eigenvalues, _ef_dst1, 2),
    ("dirichlet_stag", dst2_eigenvalues, _ef_dst2, 2),
]

_ALL_CONFIGS = [
    ("periodic", fft_eigenvalues, _ef_fft, 2),
    ("dirichlet", dst1_eigenvalues, _ef_dst1, 2),
    ("dirichlet_stag", dst2_eigenvalues, _ef_dst2, 2),
    ("neumann", dct1_eigenvalues, _ef_dct1, 2),
    ("neumann_stag", dct2_eigenvalues, _ef_dct2, 2),
]


# ---------------------------------------------------------------------------
# 1. Same-BC consistency: matches existing monolithic 3D solvers
# ---------------------------------------------------------------------------


class TestSameBCConsistency:
    """solve_helmholtz_3d with identical BCs on all axes must match
    the existing monolithic same-BC 3D solvers exactly."""

    def _make_rhs(self, bc, eig_fn, ef_fn, k):
        fx = ef_fn(NX, k)
        fy = ef_fn(NY, k)
        fz = ef_fn(NZ, k)
        psi = _ef_3d(fx, fy, fz)
        eigx = eig_fn(NX, DX)[k]
        eigy = eig_fn(NY, DY)[k]
        eigz = eig_fn(NZ, DZ)[k]
        return (eigx + eigy + eigz - 1.0) * psi

    def test_periodic_matches_fft(self):
        rhs = self._make_rhs("periodic", fft_eigenvalues, _ef_fft, 2)
        ref = solve_helmholtz_fft_3d(rhs, DX, DY, DZ, 1.0)
        got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "periodic", "periodic", "periodic", 1.0
        )
        assert jnp.allclose(got, ref, atol=1e-10)

    def test_dirichlet_matches_dst1(self):
        rhs = self._make_rhs("dirichlet", dst1_eigenvalues, _ef_dst1, 2)
        ref = solve_helmholtz_dst1_3d(rhs, DX, DY, DZ, 1.0)
        got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "dirichlet", "dirichlet", "dirichlet", 1.0
        )
        assert jnp.allclose(got, ref, atol=1e-10)

    def test_dirichlet_stag_matches_dst2(self):
        rhs = self._make_rhs("dirichlet_stag", dst2_eigenvalues, _ef_dst2, 2)
        ref = solve_helmholtz_dst2_3d(rhs, DX, DY, DZ, 1.0)
        got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "dirichlet_stag", "dirichlet_stag", "dirichlet_stag", 1.0
        )
        assert jnp.allclose(got, ref, atol=1e-10)

    def test_neumann_matches_dct1(self):
        rhs = self._make_rhs("neumann", dct1_eigenvalues, _ef_dct1, 2)
        ref = solve_helmholtz_dct1_3d(rhs, DX, DY, DZ, 1.0)
        got = solve_helmholtz_3d(rhs, DX, DY, DZ, "neumann", "neumann", "neumann", 1.0)
        assert jnp.allclose(got, ref, atol=1e-10)

    def test_neumann_stag_matches_dct2(self):
        rhs = self._make_rhs("neumann_stag", dct2_eigenvalues, _ef_dct2, 2)
        ref = solve_helmholtz_dct2_3d(rhs, DX, DY, DZ, 1.0)
        got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "neumann_stag", "neumann_stag", "neumann_stag", 1.0
        )
        assert jnp.allclose(got, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Mixed real-real-real BCs: different DST/DCT on each axis
# ---------------------------------------------------------------------------


class TestMixedRealRealReal:
    """All three axes use DST/DCT (no FFT). Case RRR."""

    def test_dirichlet_x_neumann_stag_y_dirichlet_stag_z(self):
        kx, ky, kz = 2, 2, 2
        fx = _ef_dst1(NX, kx)
        fy = _ef_dct2(NY, ky)
        fz = _ef_dst2(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = dct2_eigenvalues(NY, DY)[ky]
        eigz = dst2_eigenvalues(NZ, DZ)[kz - 1]
        lam = 1.5
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "dirichlet", "neumann_stag", "dirichlet_stag", lam
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)

    def test_neumann_x_dirichlet_y_neumann_stag_z(self):
        kx, ky, kz = 2, 2, 2
        fx = _ef_dct1(NX, kx)
        fy = _ef_dst1(NY, ky)
        fz = _ef_dct2(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = dct1_eigenvalues(NX, DX)[kx]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        eigz = dct2_eigenvalues(NZ, DZ)[kz]
        lam = 2.0
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "neumann", "dirichlet", "neumann_stag", lam
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Mixed periodic+real: physical use cases
# ---------------------------------------------------------------------------


class TestMixedPeriodicReal:
    """Combinations with some periodic and some real axes."""

    def test_atmospheric_bl_periodic_xy_neumann_z(self):
        """Case RPP: periodic x/y, Neumann staggered z."""
        kx, ky, kz = 2, 3, 2
        fx = _ef_fft(NX, kx)
        fy = _ef_fft(NY, ky)
        fz = _ef_dct2(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = fft_eigenvalues(NY, DY)[ky]
        eigz = dct2_eigenvalues(NZ, DZ)[kz]
        lam = 1.0
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "periodic", "periodic", "neumann_stag", lam
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)

    def test_channel_flow_periodic_xz_dirichlet_y(self):
        """Case PRP: periodic x/z, Dirichlet y."""
        kx, ky, kz = 2, 2, 3
        fx = _ef_fft(NX, kx)
        fy = _ef_dst1(NY, ky)
        fz = _ef_fft(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        eigz = fft_eigenvalues(NZ, DZ)[kz]
        lam = 0.5
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "periodic", "dirichlet", "periodic", lam
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)

    def test_periodic_x_real_yz(self):
        """Case PRR: periodic x, Dirichlet y, Neumann staggered z."""
        kx, ky, kz = 2, 2, 2
        fx = _ef_fft(NX, kx)
        fy = _ef_dst1(NY, ky)
        fz = _ef_dct2(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dst1_eigenvalues(NY, DY)[ky - 1]
        eigz = dct2_eigenvalues(NZ, DZ)[kz]
        lam = 1.0
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "periodic", "dirichlet", "neumann_stag", lam
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)

    def test_real_x_periodic_yz(self):
        """Case PPR: Dirichlet x, periodic y/z."""
        kx, ky, kz = 2, 2, 3
        fx = _ef_dst1(NX, kx)
        fy = _ef_fft(NY, ky)
        fz = _ef_fft(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = fft_eigenvalues(NY, DY)[ky]
        eigz = fft_eigenvalues(NZ, DZ)[kz]
        lam = 1.0
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "dirichlet", "periodic", "periodic", lam
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)

    def test_periodic_xz_neumann_stag_y(self):
        """Case RPR: Neumann staggered y, periodic x and z."""
        kx, ky, kz = 2, 2, 3
        fx = _ef_fft(NX, kx)
        fy = _ef_dct2(NY, ky)
        fz = _ef_fft(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = dct2_eigenvalues(NY, DY)[ky]
        eigz = fft_eigenvalues(NZ, DZ)[kz]
        lam = 1.0
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "periodic", "neumann_stag", "periodic", lam
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Mixed left/right BCs on one axis (DST-III/IV, DCT-III/IV)
# ---------------------------------------------------------------------------


class TestMixedLeftRightBC:
    """One axis uses a mixed left/right BC (tuple), others use same-BC."""

    def test_dst4_z_periodic_xy(self):
        """DST-IV on z (Dirichlet-left + Neumann-right staggered), periodic x/y."""
        kx, ky, kz = 2, 3, 2
        fx = _ef_fft(NX, kx)
        fy = _ef_fft(NY, ky)
        fz = _ef_dst4(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = fft_eigenvalues(NY, DY)[ky]
        eigz = dst4_eigenvalues(NZ, DZ)[kz]
        lam = 1.0
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs,
            DX,
            DY,
            DZ,
            "periodic",
            "periodic",
            ("dirichlet_stag", "neumann_stag"),
            lam,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)

    def test_dct4_y_dirichlet_xz(self):
        """DCT-IV on y (Neumann-left + Dirichlet-right staggered), Dirichlet x/z."""
        kx, ky, kz = 2, 2, 2
        fx = _ef_dst1(NX, kx)
        fy = _ef_dct4(NY, ky)
        fz = _ef_dst1(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = dst1_eigenvalues(NX, DX)[kx - 1]
        eigy = dct4_eigenvalues(NY, DY)[ky]
        eigz = dst1_eigenvalues(NZ, DZ)[kz - 1]
        lam = 2.0
        rhs = (eigx + eigy + eigz - lam) * psi_exact
        psi_got = solve_helmholtz_3d(
            rhs,
            DX,
            DY,
            DZ,
            "dirichlet",
            ("neumann_stag", "dirichlet_stag"),
            "dirichlet",
            lam,
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Null-mode handling
# ---------------------------------------------------------------------------


class TestNullMode:
    """Null-mode handling for Poisson with Neumann/periodic BCs."""

    def test_triply_periodic_poisson_zero_mean(self):
        """Triply periodic Poisson: zero-mean gauge."""
        rhs = _ef_3d(_ef_fft(NX, 2), _ef_fft(NY, 3), _ef_fft(NZ, 2))
        psi = solve_poisson_3d(rhs, DX, DY, DZ, "periodic", "periodic", "periodic")
        assert jnp.isfinite(psi).all()
        assert jnp.abs(psi.mean()) < 1e-10

    def test_triply_neumann_stag_poisson_zero_mean(self):
        """Triply Neumann staggered Poisson: zero-mean gauge."""
        rhs = _ef_3d(_ef_dct2(NX, 2), _ef_dct2(NY, 2), _ef_dct2(NZ, 2))
        psi = solve_poisson_3d(
            rhs, DX, DY, DZ, "neumann_stag", "neumann_stag", "neumann_stag"
        )
        assert jnp.isfinite(psi).all()
        assert jnp.abs(psi.mean()) < 1e-10

    def test_mixed_with_dirichlet_no_null(self):
        """When any axis is Dirichlet, there's no null mode — Poisson is well-posed."""
        kx, ky, kz = 2, 2, 2
        fx = _ef_fft(NX, kx)
        fy = _ef_fft(NY, ky)
        fz = _ef_dst1(NZ, kz)
        psi_exact = _ef_3d(fx, fy, fz)
        eigx = fft_eigenvalues(NX, DX)[kx]
        eigy = fft_eigenvalues(NY, DY)[ky]
        eigz = dst1_eigenvalues(NZ, DZ)[kz - 1]
        rhs = (eigx + eigy + eigz) * psi_exact
        psi_got = solve_poisson_3d(rhs, DX, DY, DZ, "periodic", "periodic", "dirichlet")
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. solve_poisson_3d convenience wrapper
# ---------------------------------------------------------------------------


class TestSolvePoisson3D:
    """solve_poisson_3d is equivalent to solve_helmholtz_3d with lambda_=0."""

    def test_equivalence(self):
        rhs = _ef_3d(_ef_dst1(NX, 2), _ef_dst1(NY, 2), _ef_dst1(NZ, 2))
        ref = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "dirichlet", "dirichlet", "dirichlet", 0.0
        )
        got = solve_poisson_3d(rhs, DX, DY, DZ, "dirichlet", "dirichlet", "dirichlet")
        assert jnp.allclose(got, ref, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:
    """Solver must be JIT-compatible with static BC args."""

    def test_jit_helmholtz(self):
        solve_jit = jax.jit(
            solve_helmholtz_3d, static_argnames=("bc_x", "bc_y", "bc_z")
        )
        rhs = _ef_3d(_ef_dst1(NX, 2), _ef_fft(NY, 2), _ef_dct2(NZ, 2))
        psi = solve_jit(
            rhs, DX, DY, DZ, bc_x="dirichlet", bc_y="periodic", bc_z="neumann_stag"
        )
        assert jnp.isfinite(psi).all()


# ---------------------------------------------------------------------------
# 8. Module class (Layer 2)
# ---------------------------------------------------------------------------


class TestMixedBCHelmholtzSolver3D:
    """Layer 2 module wrapper."""

    def test_matches_functional(self):
        rhs = _ef_3d(_ef_dst1(NX, 2), _ef_fft(NY, 3), _ef_dct2(NZ, 2))
        ref = solve_helmholtz_3d(
            rhs,
            DX,
            DY,
            DZ,
            bc_x="dirichlet",
            bc_y="periodic",
            bc_z="neumann_stag",
            lambda_=1.5,
        )
        solver = MixedBCHelmholtzSolver3D(
            dx=DX,
            dy=DY,
            dz=DZ,
            bc_x="dirichlet",
            bc_y="periodic",
            bc_z="neumann_stag",
            alpha=1.5,
        )
        got = solver(rhs)
        assert jnp.allclose(got, ref, atol=1e-12)

    def test_filter_jit(self):
        solver = MixedBCHelmholtzSolver3D(
            dx=DX,
            dy=DY,
            dz=DZ,
            bc_x="periodic",
            bc_y="dirichlet",
            bc_z="periodic",
        )
        solver_jit = eqx.filter_jit(solver)
        rhs = _ef_3d(_ef_fft(NX, 2), _ef_dst1(NY, 2), _ef_fft(NZ, 3))
        psi = solver_jit(rhs)
        assert jnp.isfinite(psi).all()


# ---------------------------------------------------------------------------
# 9. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Invalid BC raises ValueError."""

    def test_invalid_bc_raises(self):
        rhs = jnp.zeros((NZ, NY, NX))
        with pytest.raises(ValueError, match="Unsupported boundary condition"):
            solve_helmholtz_3d(rhs, DX, DY, DZ, bc_x="invalid")


# ---------------------------------------------------------------------------
# 10. Zero RHS
# ---------------------------------------------------------------------------


class TestZeroRHS:
    """Zero RHS gives zero (or near-zero) solution."""

    def test_zero_rhs_dirichlet(self):
        rhs = jnp.zeros((NZ, NY, NX))
        psi = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "dirichlet", "dirichlet", "dirichlet", 1.0
        )
        assert jnp.allclose(psi, 0.0, atol=1e-12)

    def test_zero_rhs_mixed(self):
        rhs = jnp.zeros((NZ, NY, NX))
        psi = solve_helmholtz_3d(
            rhs, DX, DY, DZ, "periodic", "dirichlet_stag", "neumann_stag", 1.0
        )
        assert jnp.allclose(psi, 0.0, atol=1e-12)
