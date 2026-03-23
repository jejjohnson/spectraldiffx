"""Tests for inhomogeneous boundary condition support."""

import jax.numpy as jnp
import pytest

from spectraldiffx._src.fourier.solvers import (
    MixedBCHelmholtzSolver2D,
    modify_rhs_1d,
    modify_rhs_2d,
    solve_helmholtz_2d,
    solve_helmholtz_3d,
    solve_poisson_2d,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DX = 0.1
DY = 0.1
DZ = 0.1


# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------


class TestModifyRhs1D:
    """Test modify_rhs_1d directly."""

    def test_zero_values_returns_unchanged(self):
        rhs = jnp.ones(10)
        result = modify_rhs_1d(rhs, "dirichlet", DX, bc_values=(0.0, 0.0))
        assert jnp.allclose(result, rhs)

    def test_periodic_nonzero_raises(self):
        rhs = jnp.ones(10)
        with pytest.raises(ValueError, match="Periodic"):
            modify_rhs_1d(rhs, "periodic", DX, bc_values=(1.0, 0.0))

    def test_dirichlet_regular_correction(self):
        """Verify DST-I formula: f[0] -= a/dx^2, f[-1] -= b/dx^2."""
        N = 10
        rhs = jnp.zeros(N)
        a, b = 2.0, 3.0
        result = modify_rhs_1d(rhs, "dirichlet", DX, bc_values=(a, b))
        assert jnp.isclose(result[0], -a / DX**2)
        assert jnp.isclose(result[-1], -b / DX**2)
        # Interior unchanged
        assert jnp.allclose(result[1:-1], 0.0)

    def test_dirichlet_stag_correction(self):
        """Verify DST-II formula: f[0] -= 2a/dx^2, f[-1] -= 2b/dx^2."""
        N = 10
        rhs = jnp.zeros(N)
        a, b = 2.0, 3.0
        result = modify_rhs_1d(rhs, "dirichlet_stag", DX, bc_values=(a, b))
        assert jnp.isclose(result[0], -2 * a / DX**2)
        assert jnp.isclose(result[-1], -2 * b / DX**2)

    def test_neumann_regular_correction(self):
        """Verify DCT-I formula: f[0] -= 2*g_L/dx, f[-1] -= 2*g_R/dx."""
        N = 10
        rhs = jnp.zeros(N)
        gL, gR = 1.0, 2.0
        result = modify_rhs_1d(rhs, "neumann", DX, bc_values=(gL, gR))
        assert jnp.isclose(result[0], -2 * gL / DX)
        assert jnp.isclose(result[-1], -2 * gR / DX)

    def test_neumann_stag_correction(self):
        """Verify DCT-II formula: f[0] -= g_L/dx, f[-1] -= g_R/dx."""
        N = 10
        rhs = jnp.zeros(N)
        gL, gR = 1.0, 2.0
        result = modify_rhs_1d(rhs, "neumann_stag", DX, bc_values=(gL, gR))
        assert jnp.isclose(result[0], -gL / DX)
        assert jnp.isclose(result[-1], -gR / DX)


# ---------------------------------------------------------------------------
# 2D Dirichlet — quadratic solution (exact for FD2)
# ---------------------------------------------------------------------------


class TestDirichlet2DQuadratic:
    """psi(x,y) = x^2 + y^2 on regular grid with Dirichlet BCs.

    nabla^2 psi = 4. FD2 is exact for degree-2 polynomials, so the
    solver should recover psi to machine precision.
    """

    def _setup(self, Nx, Ny, dx, dy):
        # Regular grid: interior points at dx, 2*dx, ..., Nx*dx
        # Boundary at x=0 and x=(Nx+1)*dx
        x = jnp.arange(1, Nx + 1) * dx
        y = jnp.arange(1, Ny + 1) * dy
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        psi_exact = X**2 + Y**2
        rhs = 4.0 * jnp.ones((Ny, Nx))

        Lx = (Nx + 1) * dx
        Ly = (Ny + 1) * dy

        # Boundary values: psi at the boundary points
        y_full = jnp.arange(1, Ny + 1) * dy
        x_full = jnp.arange(1, Nx + 1) * dx
        bc_x_left = y_full**2  # psi(0, y) = y^2
        bc_x_right = Lx**2 + y_full**2  # psi(Lx, y) = Lx^2 + y^2
        bc_y_bottom = x_full**2  # psi(x, 0) = x^2
        bc_y_top = x_full**2 + Ly**2  # psi(x, Ly) = x^2 + Ly^2

        return psi_exact, rhs, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top

    def test_quadratic_exact(self):
        Nx, Ny = 16, 12
        dx, dy = 0.1, 0.1
        psi_exact, rhs, xl, xr, yb, yt = self._setup(Nx, Ny, dx, dy)
        psi_got = solve_helmholtz_2d(
            rhs,
            dx,
            dy,
            bc_x="dirichlet",
            bc_y="dirichlet",
            bc_x_values=(xl, xr),
            bc_y_values=(yb, yt),
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)

    def test_quadratic_poisson(self):
        """solve_poisson_2d convenience wrapper."""
        Nx, Ny = 16, 12
        dx, dy = 0.1, 0.1
        psi_exact, rhs, xl, xr, yb, yt = self._setup(Nx, Ny, dx, dy)
        psi_got = solve_poisson_2d(
            rhs,
            dx,
            dy,
            bc_x="dirichlet",
            bc_y="dirichlet",
            bc_x_values=(xl, xr),
            bc_y_values=(yb, yt),
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# 2D Dirichlet — linear solution (exact, simpler)
# ---------------------------------------------------------------------------


class TestDirichlet2DLinear:
    """psi(x,y) = 3x + 2y + 1 on regular grid with Dirichlet BCs.

    nabla^2 psi = 0. Linear solution is trivially exact.
    """

    def test_linear_exact(self):
        Nx, Ny = 16, 12
        dx, dy = 0.1, 0.1
        x = jnp.arange(1, Nx + 1) * dx
        y = jnp.arange(1, Ny + 1) * dy
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        psi_exact = 3.0 * X + 2.0 * Y + 1.0

        Lx = (Nx + 1) * dx
        Ly = (Ny + 1) * dy

        # Helmholtz with lambda=1.0 to avoid Poisson null-mode issues
        lam = 1.0
        rhs = -lam * psi_exact  # nabla^2 psi = 0, so rhs = (0 - lambda) * psi

        y_arr = jnp.arange(1, Ny + 1) * dy
        x_arr = jnp.arange(1, Nx + 1) * dx

        psi_got = solve_helmholtz_2d(
            rhs,
            dx,
            dy,
            bc_x="dirichlet",
            bc_y="dirichlet",
            lambda_=lam,
            bc_x_values=(
                3.0 * 0.0 + 2.0 * y_arr + 1.0,  # psi(0, y)
                3.0 * Lx + 2.0 * y_arr + 1.0,  # psi(Lx, y)
            ),
            bc_y_values=(
                3.0 * x_arr + 2.0 * 0.0 + 1.0,  # psi(x, 0)
                3.0 * x_arr + 2.0 * Ly + 1.0,  # psi(x, Ly)
            ),
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# 2D Dirichlet staggered — quadratic solution
# ---------------------------------------------------------------------------


class TestDirichletStag2DConvergence:
    """psi(x,y) = sin(pi*x)*sin(pi*y) on staggered grid with Dirichlet BCs.

    Cell centres at (i+0.5)*dx. Boundary at x=0 and x=Nx*dx.
    The staggered ghost-point interpolation gives O(h^2) accuracy.
    """

    def test_convergence(self):
        errors = []
        for N in [16, 32, 64]:
            dx = 1.0 / N
            x = (jnp.arange(N) + 0.5) * dx
            y = (jnp.arange(N) + 0.5) * dx
            X, Y = jnp.meshgrid(x, y, indexing="xy")
            Lx = 1.0

            # psi = sin(pi*x)*sin(pi*y) + x + y (non-trivial boundary values)
            psi_exact2 = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) + X + Y
            rhs2 = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

            y_arr = (jnp.arange(N) + 0.5) * dx
            x_arr = (jnp.arange(N) + 0.5) * dx

            psi_got = solve_poisson_2d(
                rhs2,
                dx,
                dx,
                bc_x="dirichlet_stag",
                bc_y="dirichlet_stag",
                bc_x_values=(
                    jnp.sin(jnp.pi * 0.0) * jnp.sin(jnp.pi * y_arr) + 0.0 + y_arr,
                    jnp.sin(jnp.pi * Lx) * jnp.sin(jnp.pi * y_arr) + Lx + y_arr,
                ),
                bc_y_values=(
                    jnp.sin(jnp.pi * x_arr) * jnp.sin(jnp.pi * 0.0) + x_arr + 0.0,
                    jnp.sin(jnp.pi * x_arr) * jnp.sin(jnp.pi * Lx) + x_arr + Lx,
                ),
            )
            err = float(jnp.max(jnp.abs(psi_got - psi_exact2)))
            errors.append(err)

        # O(h^2) convergence
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        assert ratio1 > 3.0, f"Expected ~4x, got {ratio1:.2f}"
        assert ratio2 > 3.0, f"Expected ~4x, got {ratio2:.2f}"


# ---------------------------------------------------------------------------
# 2D Neumann staggered — quadratic solution
# ---------------------------------------------------------------------------


class TestNeumannStag2DQuadratic:
    """psi(x,y) = x^2 + y^2 on staggered grid with Neumann BCs (DCT-II).

    dpsi/dx = 2x, dpsi/dy = 2y.
    Outward normal convention: g_L = -dpsi/dx at left, g_R = dpsi/dx at right.
    """

    def test_quadratic_helmholtz(self):
        Nx, Ny = 16, 12
        dx, dy = 0.1, 0.1
        x = (jnp.arange(Nx) + 0.5) * dx
        y = (jnp.arange(Ny) + 0.5) * dy
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        psi_exact = X**2 + Y**2

        Lx = Nx * dx
        Ly = Ny * dy

        # Helmholtz to avoid null mode
        lam = 1.0
        rhs = 4.0 * jnp.ones((Ny, Nx)) - lam * psi_exact

        # Outward normal derivatives:
        # Left (x=0): n = -x_hat, so dpsi/dn = -dpsi/dx = -2*0 = 0
        # Right (x=Lx): n = +x_hat, so dpsi/dn = dpsi/dx = 2*Lx
        # Bottom (y=0): n = -y_hat, so dpsi/dn = -dpsi/dy = -2*0 = 0
        # Top (y=Ly): n = +y_hat, so dpsi/dn = dpsi/dy = 2*Ly
        psi_got = solve_helmholtz_2d(
            rhs,
            dx,
            dy,
            bc_x="neumann_stag",
            bc_y="neumann_stag",
            lambda_=lam,
            bc_x_values=(
                jnp.zeros(Ny),  # dpsi/dn at x=0
                2.0 * Lx * jnp.ones(Ny),  # dpsi/dn at x=Lx
            ),
            bc_y_values=(
                jnp.zeros(Nx),  # dpsi/dn at y=0
                2.0 * Ly * jnp.ones(Nx),  # dpsi/dn at y=Ly
            ),
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# Mixed per-axis: Dirichlet in x, Neumann staggered in y
# ---------------------------------------------------------------------------


class TestMixedInhomogeneous2D:
    """Dirichlet in x (regular) + Neumann staggered in y, quadratic solution."""

    def test_mixed_dirichlet_neumann(self):
        Nx, Ny = 16, 12
        dx, dy = 0.1, 0.1
        # Regular grid in x: interior at dx, 2*dx, ..., Nx*dx
        # Staggered grid in y: centres at (j+0.5)*dy
        x = jnp.arange(1, Nx + 1) * dx
        y = (jnp.arange(Ny) + 0.5) * dy
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        psi_exact = X**2 + Y**2

        Lx = (Nx + 1) * dx
        Ly = Ny * dy

        lam = 1.0
        rhs = 4.0 * jnp.ones((Ny, Nx)) - lam * psi_exact

        y_arr = (jnp.arange(Ny) + 0.5) * dy

        psi_got = solve_helmholtz_2d(
            rhs,
            dx,
            dy,
            bc_x="dirichlet",
            bc_y="neumann_stag",
            lambda_=lam,
            bc_x_values=(
                y_arr**2,  # psi(0, y) = y^2
                Lx**2 + y_arr**2,  # psi(Lx, y)
            ),
            bc_y_values=(
                jnp.zeros(Nx),  # dpsi/dn at y=0 = 0
                2.0 * Ly * jnp.ones(Nx),  # dpsi/dn at y=Ly = 2*Ly
            ),
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# O(h^2) convergence for smooth solution
# ---------------------------------------------------------------------------


class TestConvergence:
    """Verify O(h^2) convergence for a smooth non-polynomial solution."""

    def test_dirichlet_convergence(self):
        """Solve with sin*sin solution, verify error decreases as O(h^2)."""
        errors = []
        resolutions = [16, 32, 64]

        for N in resolutions:
            dx = 1.0 / (N + 1)
            x = jnp.arange(1, N + 1) * dx
            y = jnp.arange(1, N + 1) * dx
            X, Y = jnp.meshgrid(x, y, indexing="xy")

            # psi = sin(pi*x) * sin(pi*y) + x + y
            psi_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) + X + Y
            # nabla^2 psi = -2*pi^2 * sin(pi*x) * sin(pi*y)
            rhs = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

            Lx = 1.0
            # Boundary values
            y_arr = jnp.arange(1, N + 1) * dx
            x_arr = jnp.arange(1, N + 1) * dx

            psi_got = solve_poisson_2d(
                rhs,
                dx,
                dx,
                bc_x="dirichlet",
                bc_y="dirichlet",
                bc_x_values=(
                    jnp.sin(jnp.pi * 0.0) * jnp.sin(jnp.pi * y_arr)
                    + 0.0
                    + y_arr,  # x=0
                    jnp.sin(jnp.pi * Lx) * jnp.sin(jnp.pi * y_arr) + Lx + y_arr,  # x=Lx
                ),
                bc_y_values=(
                    jnp.sin(jnp.pi * x_arr) * jnp.sin(jnp.pi * 0.0)
                    + x_arr
                    + 0.0,  # y=0
                    jnp.sin(jnp.pi * x_arr) * jnp.sin(jnp.pi * Lx) + x_arr + Lx,  # y=Ly
                ),
            )
            err = jnp.max(jnp.abs(psi_got - psi_exact))
            errors.append(float(err))

        # Check O(h^2) convergence: ratio should be ~4 when h halves
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        assert ratio1 > 3.0, f"Expected ~4x improvement, got {ratio1:.2f}"
        assert ratio2 > 3.0, f"Expected ~4x improvement, got {ratio2:.2f}"


# ---------------------------------------------------------------------------
# 3D inhomogeneous — quadratic
# ---------------------------------------------------------------------------


class TestInhomogeneous3D:
    """3D inhomogeneous Dirichlet with quadratic solution."""

    def test_quadratic_3d(self):
        Nx, Ny, Nz = 8, 6, 5
        dx, dy, dz = 0.1, 0.1, 0.1
        x = jnp.arange(1, Nx + 1) * dx
        y = jnp.arange(1, Ny + 1) * dy
        z = jnp.arange(1, Nz + 1) * dz
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="xy")
        # Reorder to (Nz, Ny, Nx)
        X = X.transpose(2, 0, 1)
        Y = Y.transpose(2, 0, 1)
        Z = Z.transpose(2, 0, 1)
        psi_exact = X**2 + Y**2 + Z**2
        rhs = 6.0 * jnp.ones((Nz, Ny, Nx))

        Lx = (Nx + 1) * dx
        Ly = (Ny + 1) * dy
        Lz = (Nz + 1) * dz

        # Build face arrays
        # x-faces: shape (Nz, Ny)
        y2d_x, z2d_x = jnp.meshgrid(
            jnp.arange(1, Ny + 1) * dy, jnp.arange(1, Nz + 1) * dz
        )
        xl = y2d_x**2 + z2d_x**2  # psi(0, y, z)
        xr = Lx**2 + y2d_x**2 + z2d_x**2

        # y-faces: shape (Nz, Nx)
        x2d_y, z2d_y = jnp.meshgrid(
            jnp.arange(1, Nx + 1) * dx, jnp.arange(1, Nz + 1) * dz
        )
        yb = x2d_y**2 + z2d_y**2
        yt = x2d_y**2 + Ly**2 + z2d_y**2

        # z-faces: shape (Ny, Nx)
        x2d_z, y2d_z = jnp.meshgrid(
            jnp.arange(1, Nx + 1) * dx, jnp.arange(1, Ny + 1) * dy
        )
        zb = x2d_z**2 + y2d_z**2
        zf = x2d_z**2 + y2d_z**2 + Lz**2

        psi_got = solve_helmholtz_3d(
            rhs,
            dx,
            dy,
            dz,
            bc_x="dirichlet",
            bc_y="dirichlet",
            bc_z="dirichlet",
            bc_x_values=(xl, xr),
            bc_y_values=(yb, yt),
            bc_z_values=(zb, zf),
        )
        assert jnp.allclose(psi_got, psi_exact, atol=1e-9)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Default bc_values=(None, None) doesn't change behavior."""

    def test_2d_default_unchanged(self):
        """solve_helmholtz_2d with default bc_values matches without them."""
        Nx, Ny = 16, 12
        rhs = jnp.sin(jnp.arange(Ny)[:, None]) * jnp.cos(jnp.arange(Nx)[None, :])
        ref = solve_helmholtz_2d(rhs, DX, DY, "dirichlet", "dirichlet", 1.0)
        got = solve_helmholtz_2d(
            rhs,
            DX,
            DY,
            "dirichlet",
            "dirichlet",
            1.0,
            bc_x_values=(None, None),
            bc_y_values=(None, None),
        )
        assert jnp.allclose(got, ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Module class
# ---------------------------------------------------------------------------


class TestModuleClass:
    """MixedBCHelmholtzSolver2D with inhomogeneous bc_values."""

    def test_matches_functional(self):
        Nx, Ny = 16, 12
        dx, dy = 0.1, 0.1
        x = jnp.arange(1, Nx + 1) * dx
        y = jnp.arange(1, Ny + 1) * dy
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        psi_exact = X**2 + Y**2
        rhs = 4.0 * jnp.ones((Ny, Nx))

        Lx = (Nx + 1) * dx
        Ly = (Ny + 1) * dy
        y_arr = jnp.arange(1, Ny + 1) * dy
        x_arr = jnp.arange(1, Nx + 1) * dx

        xl = y_arr**2
        xr = Lx**2 + y_arr**2
        yb = x_arr**2
        yt = x_arr**2 + Ly**2

        solver = MixedBCHelmholtzSolver2D(
            dx=dx, dy=dy, bc_x="dirichlet", bc_y="dirichlet"
        )
        psi_got = solver(rhs, bc_x_values=(xl, xr), bc_y_values=(yb, yt))
        assert jnp.allclose(psi_got, psi_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Validation errors for invalid combinations."""

    def test_modify_rhs_2d_with_periodic_and_values(self):
        """Cannot pass non-None values for periodic axis."""
        rhs = jnp.zeros((10, 10))
        with pytest.raises(KeyError):
            # periodic is not in _BC_RHS_FORMULAS
            modify_rhs_2d(
                rhs,
                "periodic",
                "dirichlet",
                DX,
                DY,
                bc_x_values=(jnp.ones(10), None),
            )
