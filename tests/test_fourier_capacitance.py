"""Tests for the capacitance matrix solver."""

import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx._src.fourier.capacitance import (
    CapacitanceSolver,
    build_capacitance_solver,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NY, NX = 16, 16
DX, DY = 1.0, 1.0


def _make_circle_mask(Ny, Nx, radius_frac=0.35):
    """Create a circular mask inside a rectangle."""
    j, i = np.mgrid[0:Ny, 0:Nx]
    cy, cx = Ny / 2, Nx / 2
    r = np.sqrt((j - cy) ** 2 + (i - cx) ** 2)
    return r < radius_frac * min(Ny, Nx)


@pytest.fixture()
def circle_mask():
    return _make_circle_mask(NY, NX)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildCapacitanceSolver:
    """build_capacitance_solver returns a valid CapacitanceSolver."""

    def test_returns_solver(self, circle_mask):
        solver = build_capacitance_solver(circle_mask, DX, DY)
        assert isinstance(solver, CapacitanceSolver)

    def test_all_ones_mask_raises(self):
        mask = np.ones((NY, NX), dtype=bool)
        with pytest.raises(ValueError, match="No inner-boundary"):
            build_capacitance_solver(mask, DX, DY)

    @pytest.mark.parametrize("bc", ["fft", "dst", "dct"])
    def test_base_bc_options(self, circle_mask, bc):
        solver = build_capacitance_solver(circle_mask, DX, DY, base_bc=bc)
        assert solver.base_bc == bc

    def test_invalid_bc_raises(self, circle_mask):
        build_capacitance_solver(circle_mask, DX, DY, base_bc="fft")
        # Test that _spectral_solve raises for invalid bc
        from spectraldiffx._src.fourier.capacitance import _spectral_solve

        with pytest.raises(ValueError, match="base_bc"):
            _spectral_solve(jnp.zeros((NY, NX)), DX, DY, 0.0, "invalid")


class TestCapacitanceSolverCall:
    """CapacitanceSolver.__call__ produces correct solutions."""

    def test_output_shape(self, circle_mask):
        solver = build_capacitance_solver(circle_mask, DX, DY)
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert psi.shape == (NY, NX)

    def test_zero_rhs_near_zero(self, circle_mask):
        solver = build_capacitance_solver(circle_mask, DX, DY)
        rhs = jnp.zeros((NY, NX))
        psi = solver(rhs)
        assert jnp.allclose(psi, 0.0, atol=1e-8)

    def test_boundary_enforcement(self, circle_mask):
        """Solution should be approximately zero at inner-boundary points."""
        from scipy.ndimage import binary_dilation

        mask_bool = np.asarray(circle_mask, dtype=bool)
        exterior = ~mask_bool
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        dilated = binary_dilation(exterior, structure=struct)
        inner_boundary = mask_bool & dilated
        j_b, i_b = np.where(inner_boundary)

        solver = build_capacitance_solver(circle_mask, DX, DY, base_bc="dst")
        # Use a smooth source inside the mask
        rhs = jnp.ones((NY, NX)) * jnp.array(circle_mask, dtype=float)
        psi = solver(rhs)

        boundary_values = psi[j_b, i_b]
        assert jnp.max(jnp.abs(boundary_values)) < 1e-6

    def test_helmholtz_nonzero_lambda(self, circle_mask):
        """Solver with λ ≠ 0 should produce a different solution."""
        solver_poisson = build_capacitance_solver(circle_mask, DX, DY, lambda_=0.0)
        solver_helm = build_capacitance_solver(circle_mask, DX, DY, lambda_=1.0)
        rhs = jnp.ones((NY, NX)) * jnp.array(circle_mask, dtype=float)
        psi_p = solver_poisson(rhs)
        psi_h = solver_helm(rhs)
        assert not jnp.allclose(psi_p, psi_h, atol=1e-3)

    def test_dst_and_fft_both_enforce_boundary(self, circle_mask):
        """Both DST and FFT bases should enforce zero at boundary points."""
        from scipy.ndimage import binary_dilation

        mask_bool = np.asarray(circle_mask, dtype=bool)
        exterior = ~mask_bool
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        dilated = binary_dilation(exterior, structure=struct)
        inner_boundary = mask_bool & dilated
        j_b, i_b = np.where(inner_boundary)

        rhs = jnp.ones((NY, NX)) * jnp.array(circle_mask, dtype=float)
        for bc in ("fft", "dst"):
            solver = build_capacitance_solver(circle_mask, DX, DY, base_bc=bc)
            psi = solver(rhs)
            boundary_vals = psi[j_b, i_b]
            assert jnp.max(jnp.abs(boundary_vals)) < 1e-6, f"bc={bc}"
