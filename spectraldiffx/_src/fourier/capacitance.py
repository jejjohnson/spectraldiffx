"""Capacitance matrix solver for masked/irregular domains.

Extends a fast rectangular spectral solver (FFT/DST/DCT) to domains that are
subsets of a rectangle (e.g. ocean basins with land masks) using the classic
capacitance-matrix method (Buzbee, Golub & Nielson 1970).

The linear algebra lives in gaussx: the rectangular finite-difference
Helmholtz operator is a :class:`gaussx.DiagonalisedOperator` (its eigenbasis
is the FFT / DST-I / DCT-II), the masked problem is a
:class:`gaussx.MaskedOperator` restricted to the interior cells, and
``gaussx.solve`` applies the precomputed capacitance correction. This module
only classifies the grid cells and reshapes fields.

Reference: Buzbee, Golub & Nielson (1970), "On Direct Methods for Solving
Poisson's Equations", SIAM J. Numer. Anal.
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np

from .eigenvalues import dct2_eigenvalues, dst1_eigenvalues, fft_eigenvalues
from .transforms import dctn, dstn, idctn, idstn

_BASE_BCS = ("fft", "dst", "dct")


# ---------------------------------------------------------------------------
# Rectangular base operators (orthonormal transforms, FD2 eigenvalues)
# ---------------------------------------------------------------------------


def _dst1(x: Array) -> Array:
    """Orthonormal 2-D DST-I (symmetric: its own inverse)."""
    return dstn(x, type=1, axes=[0, 1], norm="ortho")


def _idst1(c: Array) -> Array:
    return idstn(c, type=1, axes=[0, 1], norm="ortho")


def _dct2(x: Array) -> Array:
    """Orthonormal 2-D DCT-II."""
    return dctn(x, type=2, axes=[0, 1], norm="ortho")


def _idct2(c: Array) -> Array:
    return idctn(c, type=2, axes=[0, 1], norm="ortho")


def _base_operator(
    shape: tuple[int, int], dx: float, dy: float, lambda_: float, base_bc: str
) -> gaussx.DiagonalisedOperator:
    """Rectangular five-point operator ∇² − λ as a diagonalised operator.

    The eigenvalues are the FD2 eigenvalues used by ``solve_helmholtz_fft`` /
    ``_dst`` / ``_dct`` (periodic, Dirichlet DST-I, Neumann DCT-II), so the
    base solve matches those functions; the stencil is local (5-point),
    which is what makes the one-cell coupling ring exact.

        Λ[j, i] = λy_j + λx_i − λ
    """
    ny, nx = shape
    eig_fn = {"fft": fft_eigenvalues, "dst": dst1_eigenvalues, "dct": dct2_eigenvalues}
    eig = eig_fn[base_bc](ny, dy)[:, None] + eig_fn[base_bc](nx, dx)[None, :] - lambda_
    if base_bc == "fft":
        return gaussx.circulant_from_symbol(eig)
    forward, inverse = (_dst1, _idst1) if base_bc == "dst" else (_dct2, _idct2)
    return gaussx.DiagonalisedOperator(eig, forward, inverse, shape, normal=True)


# ---------------------------------------------------------------------------
# Capacitance matrix solver
# ---------------------------------------------------------------------------


class CapacitanceSolver(eqx.Module):
    """Spectral Poisson/Helmholtz solver for masked irregular domains.

    Solves ``(∇² − λ) ψ = f`` on the wet cells of ``mask`` with ``ψ = 0`` on
    the *inner boundary* (wet cells adjacent to a dry cell) and outside the
    mask, using the five-point finite-difference Laplacian.

    Method (Buzbee, Golub & Nielson 1970, via gaussx)
    --------------------------------------------------
    Let ``B`` be the rectangular operator (periodic / Dirichlet / Neumann by
    ``base_bc``), ``I`` the interior cells (wet, not on the inner boundary)
    and ``C`` the inner-boundary cells, which are exactly the cells outside
    ``I`` that the stencil of ``I`` reaches. The masked problem is
    ``B[I][:, I] ψ_I = f_I``. The capacitance method solves it with two fast
    rectangular solves per call: find ``y`` with ``y[C] = 0`` and
    ``(B y)_k = f_k`` for every ``k ∉ C`` (point sources on ``C`` absorb the
    constraints, via an ``|C| × |C|`` capacitance matrix factorised once at
    construction). For a singular base (``fft``/``dct`` with λ = 0) the
    constant null vector is included in the capacitance system, so the PDE
    holds exactly in the interior (gh-87).

    Construct with :func:`build_capacitance_solver`.

    Attributes
    ----------
    operator : gaussx.MaskedOperator
        ``B[I][:, I]`` with its precomputed capacitance factorisation.
    interior_indices : Int[Array, "Ni"]
        Flat (row-major) indices of the interior cells ``I``.
    shape : tuple[int, int]
        Grid shape ``(Ny, Nx)``.
    dx, dy : float
        Grid spacings (static; baked into ``operator``).
    lambda_ : float
        Helmholtz parameter (static).
    base_bc : str
        Rectangular base: ``"fft"``, ``"dst"`` or ``"dct"`` (static).
    """

    operator: gaussx.MaskedOperator
    interior_indices: Int[Array, " Ni"]
    shape: tuple[int, int] = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    lambda_: float = eqx.field(static=True)
    base_bc: str = eqx.field(static=True)

    def __call__(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − λ)ψ = rhs on the masked domain.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side on the full rectangular grid. Only the interior
            cells are used; values on the inner boundary and outside the
            mask are ignored.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ on the full grid: the masked solve on the interior,
            exactly zero on the inner boundary and outside the mask.
        """
        ny, nx = self.shape
        f_interior = rhs.reshape(ny * nx)[self.interior_indices]
        psi_interior = gaussx.solve(self.operator, f_interior)
        psi = jnp.zeros(ny * nx, dtype=psi_interior.dtype)
        return psi.at[self.interior_indices].set(psi_interior).reshape(ny, nx)


def build_capacitance_solver(
    mask: np.ndarray,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    base_bc: str = "fft",
) -> CapacitanceSolver:
    """Pre-compute the capacitance factorisation and return a solver.

    1. **Classify cells** — the inner boundary is every wet cell that is
       4-connected to a dry cell; with ``base_bc="fft"`` neighbours wrap
       around the periodic rectangle. The remaining wet cells are the
       interior unknowns.
    2. **Delegate to gaussx** — ``gaussx.MaskedOperator`` over the
       rectangular base operator, with the inner boundary as its coupling
       set, factorises the ``N_b × N_b`` capacitance matrix once.

    Complexity
    ----------
    * Offline (this function): ``N_b`` rectangular solves,
      O(N_b · Ny·Nx · log(Ny·Nx)) time, O(N_b² + Ny·Nx) memory.
    * Online (``CapacitanceSolver.__call__``): two rectangular solves plus an
      O(N_b²) back-substitution.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (Ny, Nx)
        Physical domain mask. ``True`` = wet (ocean/fluid), ``False`` = dry.
    dx, dy : float
        Grid spacings in x and y.
    lambda_ : float
        Helmholtz parameter λ. ``0.0`` gives Poisson.
    base_bc : {"fft", "dst", "dct"}
        Rectangular base operator: periodic, Dirichlet (DST-I) or Neumann
        (DCT-II) on the enclosing rectangle.

    Returns
    -------
    CapacitanceSolver
        Callable equinox Module with the factorisation baked in.

    Raises
    ------
    ValueError
        For an unknown ``base_bc``, a mask with no wet cells, a mask with no
        dry cells (nothing to correct: use the rectangular solver directly),
        or a mask whose wet cells are all on the inner boundary (no interior
        unknowns).
    """
    if base_bc not in _BASE_BCS:
        raise ValueError(f"base_bc must be 'fft', 'dst', or 'dct'; got {base_bc!r}")
    wet = np.asarray(mask, dtype=bool)
    if not wet.any():
        raise ValueError("The mask has no wet (True) cells.")
    if wet.all():
        raise ValueError(
            "The mask has no dry (False) cells, so there is no inner boundary; "
            f"use solve_helmholtz_{base_bc} directly."
        )
    periodic = base_bc == "fft"

    # Wet cells reached by the stencil of a dry cell = inner boundary.
    boundary = np.zeros(wet.size, dtype=bool)
    boundary[
        np.asarray(gaussx.grid_coupling_indices(jnp.asarray(~wet), periodic=periodic))
    ] = True
    interior = wet & ~boundary.reshape(wet.shape)
    if not interior.any():
        raise ValueError(
            "Every wet cell lies on the inner boundary, so there are no interior "
            "unknowns (the domain is at most one cell wide)."
        )

    flat = jnp.asarray(interior.ravel())
    operator = gaussx.MaskedOperator(
        _base_operator(wet.shape, dx, dy, lambda_, base_bc),
        flat,
        flat,
        coupling_indices=gaussx.grid_coupling_indices(
            jnp.asarray(interior), periodic=periodic
        ),
    )
    return CapacitanceSolver(
        operator=operator,
        interior_indices=jnp.asarray(np.flatnonzero(interior.ravel())),
        shape=wet.shape,
        dx=float(dx),
        dy=float(dy),
        lambda_=float(lambda_),
        base_bc=base_bc,
    )
