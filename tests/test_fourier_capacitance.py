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
        with pytest.raises(ValueError, match="no dry"):
            build_capacitance_solver(mask, DX, DY)

    def test_all_zeros_mask_raises(self):
        mask = np.zeros((NY, NX), dtype=bool)
        with pytest.raises(ValueError, match="no wet"):
            build_capacitance_solver(mask, DX, DY)

    def test_single_cell_mask_raises(self):
        mask = np.zeros((NY, NX), dtype=bool)
        mask[5, 5] = True
        with pytest.raises(ValueError, match="no interior"):
            build_capacitance_solver(mask, DX, DY)

    @pytest.mark.parametrize("bc", ["fft", "dst", "dct"])
    def test_base_bc_options(self, circle_mask, bc):
        solver = build_capacitance_solver(circle_mask, DX, DY, base_bc=bc)
        assert solver.base_bc == bc

    def test_invalid_bc_raises(self, circle_mask):
        with pytest.raises(ValueError, match="base_bc"):
            build_capacitance_solver(circle_mask, DX, DY, base_bc="invalid")


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


# ---------------------------------------------------------------------------
# Interior PDE residual (gh-87)
#
# The masks and right-hand sides are fixed (seeded), so the tolerances are
# round-off bounds for these small well-conditioned problems.
# ---------------------------------------------------------------------------


def _inner_boundary(mask, periodic):
    """Wet cells 4-connected (wrapping if periodic) to a dry cell."""
    dry = ~mask
    near = np.zeros_like(mask)
    for axis in (0, 1):
        for step in (1, -1):
            shifted = np.roll(dry, step, axis=axis)
            if not periodic:
                edge = [slice(None)] * 2
                edge[axis] = slice(0, 1) if step == 1 else slice(-1, None)
                shifted[tuple(edge)] = False
            near |= shifted
    return mask & near


def _five_point(psi, dx, dy, base_bc):
    """Five-point Laplacian with the rectangle's own boundary treatment."""
    if base_bc == "fft":
        pad = np.pad(psi, 1, mode="wrap")
    elif base_bc == "dst":
        pad = np.pad(psi, 1, mode="constant")
    else:  # dct: cell-centred Neumann, ghost = mirror of the edge cell
        pad = np.pad(psi, 1, mode="symmetric")
    lap_y = (pad[:-2, 1:-1] - 2 * psi + pad[2:, 1:-1]) / dy**2
    lap_x = (pad[1:-1, :-2] - 2 * psi + pad[1:-1, 2:]) / dx**2
    return lap_x + lap_y


def _l_mask():
    mask = np.zeros((20, 24), dtype=bool)
    mask[2:18, 2:10] = True
    mask[10:18, 2:22] = True
    return mask


def _edge_mask():
    """Touches the left and top edges of the rectangle."""
    j, i = np.mgrid[:20, :24]
    return (j - 2.0) ** 2 + (i - 1.0) ** 2 < 8.0**2


MASKS = {
    "circle": lambda: _make_circle_mask(24, 20),
    "L": _l_mask,
    "edge": _edge_mask,
}


@pytest.mark.parametrize("mask_name", sorted(MASKS))
@pytest.mark.parametrize("base_bc", ["fft", "dst", "dct"])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_interior_pde_residual(mask_name, base_bc, lam):
    """∇²ψ − λψ = f on interior cells; ψ = 0 on the inner boundary."""
    mask = MASKS[mask_name]()
    dx, dy = 1.0, 0.7
    f = np.random.default_rng(0).standard_normal(mask.shape)
    psi = np.asarray(
        build_capacitance_solver(mask, dx, dy, lambda_=lam, base_bc=base_bc)(
            jnp.asarray(f)
        )
    )
    boundary = _inner_boundary(mask, periodic=base_bc == "fft")
    interior = mask & ~boundary
    residual = (_five_point(psi, dx, dy, base_bc) - lam * psi - f)[interior]
    assert np.abs(residual).max() < 1e-10
    assert np.abs(psi[~interior]).max() == 0.0


def test_matches_dense_masked_dirichlet_solve(circle_mask):
    """The DST-base result equals a dense solve of the interior equations."""
    mask = np.asarray(circle_mask, dtype=bool)
    interior = mask & ~_inner_boundary(mask, periodic=False)
    idx = np.flatnonzero(interior.ravel())
    n = mask.size
    eye = np.eye(n).reshape(n, *mask.shape)
    A = np.stack([_five_point(e, DX, DY, "dst").ravel() for e in eye], axis=1)
    f = np.random.default_rng(1).standard_normal(mask.shape)
    expected = np.linalg.solve(A[np.ix_(idx, idx)], f.ravel()[idx])
    psi = np.asarray(
        build_capacitance_solver(mask, DX, DY, base_bc="dst")(jnp.asarray(f))
    )
    assert np.allclose(psi.ravel()[idx], expected, atol=1e-12)


def test_solver_is_jittable(circle_mask):
    import equinox as eqx

    solver = build_capacitance_solver(circle_mask, DX, DY, lambda_=0.0)
    rhs = jnp.asarray(np.random.default_rng(2).standard_normal((NY, NX)))
    out = eqx.filter_jit(lambda s, r: s(r))(solver, rhs)
    assert jnp.allclose(out, solver(rhs), atol=1e-12)
