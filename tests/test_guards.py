"""Input guards on the Fourier transforms, grids and solvers (gh-94)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraldiffx as sdx
from spectraldiffx._src.fourier.eigenvalues import dst1_eigenvalues, fft_eigenvalues


@pytest.mark.parametrize(
    "call",
    [
        lambda: sdx.dct(jnp.ones(1), type=1),
        lambda: sdx.idct(jnp.ones(1), type=1),
        lambda: sdx.dct(jnp.ones(1), type=1, norm="ortho"),
        lambda: sdx.dctn(jnp.ones((1, 4)), type=1, axes=[0]),
        lambda: sdx.idctn(jnp.ones((4, 1)), type=1),
    ],
)
def test_dct1_needs_two_points(call):
    with pytest.raises(ValueError, match="DCT-I requires N >= 2"):
        call()


def test_dct1_length_check_only_on_transformed_axes():
    assert sdx.dctn(jnp.ones((1, 4)), type=1, axes=[1]).shape == (1, 4)


@pytest.mark.parametrize("fn", [sdx.idctn, sdx.idstn, sdx.dctn, sdx.dstn])
def test_nd_type_validated_before_axis_loop(fn):
    with pytest.raises(ValueError, match="type must be 1, 2, 3, or 4"):
        fn(jnp.ones(3), type=7, axes=[])


@pytest.mark.parametrize("fn", [sdx.dct, sdx.idct, sdx.dst, sdx.idst])
def test_bool_type_rejected(fn):
    with pytest.raises(ValueError, match="got True"):
        fn(jnp.ones(4), type=True)


def test_padding_dealias_rejected():
    with pytest.raises(ValueError, match="dealias must be"):
        sdx.FourierGrid1D.from_N_L(12, 1.0, dealias="padding")


@pytest.mark.parametrize(
    "make",
    [
        lambda: sdx.FourierGrid1D(N=8, L=2.0, dx=1.0),
        lambda: sdx.FourierGrid2D(Nx=8, Ny=8, Lx=1.0, Ly=1.0, dx=1.0, dy=0.125),
        lambda: sdx.FourierGrid3D(
            Nz=4, Ny=4, Nx=4, Lz=1.0, Ly=1.0, Lx=1.0, dz=0.25, dy=0.25, dx=1.0
        ),
    ],
    ids=["1D", "2D", "3D"],
)
def test_inconsistent_plain_constructor_rejected(make):
    with pytest.raises(ValueError, match="inconsistency"):
        make()


def test_grid_can_be_traced():
    """The consistency check is skipped for traced values."""

    @jax.jit
    def length(L):
        return sdx.FourierGrid1D(N=8, L=L, dx=L / 8).dx

    assert float(length(2.0)) == pytest.approx(0.25)


def test_resonant_lambda_raises_2d():
    n = 8
    lam = float(dst1_eigenvalues(n, 1.0)[0] + dst1_eigenvalues(n, 1.0)[2])
    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        sdx.solve_helmholtz_dst(jnp.ones((n, n)), 1.0, 1.0, lam)


def test_resonant_lambda_raises_1d_instead_of_zeroing():
    n = 8
    lam = float(fft_eigenvalues(n, 1.0)[1])
    rhs = jnp.cos(2 * np.pi * jnp.arange(n) / n)
    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        sdx.solve_helmholtz_fft_1d(rhs, 1.0, lam)


def test_null_mode_still_projected_1d():
    rhs = jnp.cos(2 * np.pi * jnp.arange(8) / 8) + 3.0
    psi = sdx.solve_helmholtz_fft_1d(rhs, 1.0)
    assert bool(jnp.all(jnp.isfinite(psi)))
    assert abs(float(jnp.mean(psi))) < 1e-12


def test_solver_class_rejects_negative_alpha():
    grid = sdx.FourierGrid2D.from_N_L(Nx=8, Ny=8, Lx=2 * np.pi, Ly=2 * np.pi)
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        sdx.SpectralHelmholtzSolver2D(grid).solve(jnp.ones((8, 8)), alpha=-1.0)
