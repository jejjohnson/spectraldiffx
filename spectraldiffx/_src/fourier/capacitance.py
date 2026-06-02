"""Capacitance matrix solver for masked/irregular domains.

Extends a fast rectangular spectral solver (DST/DCT/FFT) to domains that are
subsets of a rectangle (e.g. ocean basins with land masks) using the classic
Sherman-Morrison correction via boundary Green's functions.

The generic linear algebra (Green's functions, capacitance inverse, and the
low-rank correction) lives in :class:`gaussx.CapacitanceSolver`. This module
keeps the spectral-specific parts: extracting the inner-boundary points from a
mask, building the rectangular base solve, and reshaping/masking fields. The
public API (:class:`CapacitanceSolver`, :func:`build_capacitance_solver`) is
unchanged.

``build_capacitance_solver`` performs a one-time offline precomputation
(N_b rectangular solves, where N_b = number of irregular-boundary points).
The returned ``CapacitanceSolver`` callable is then cheap to evaluate for any
right-hand side.

Reference: Buzbee, Golub & Nielson (1970), "On Direct Methods for Solving
Poisson's Equations", SIAM J. Numer. Anal.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
from gaussx import CapacitanceSolver as _GaussxCapacitanceSolver
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from .solvers import solve_helmholtz_dct, solve_helmholtz_dst, solve_helmholtz_fft

# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

_HELMHOLTZ_DISPATCH: dict[str, Callable] = {
    "fft": solve_helmholtz_fft,
    "dst": solve_helmholtz_dst,
    "dct": solve_helmholtz_dct,
}


def _spectral_solve(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
) -> Float[Array, "Ny Nx"]:
    """Dispatch to the rectangular spectral Helmholtz solver for the given BC type.

    Maps *bc* ∈ {"fft", "dst", "dct"} to the corresponding
    ``solve_helmholtz_*`` function and calls it with the provided arguments.
    """
    solver = _HELMHOLTZ_DISPATCH.get(bc)
    if solver is None:
        raise ValueError(f"base_bc must be 'fft', 'dst', or 'dct'; got {bc!r}")
    return solver(rhs, dx, dy, lambda_)


# ---------------------------------------------------------------------------
# Capacitance matrix solver
# ---------------------------------------------------------------------------


class CapacitanceSolver(eqx.Module):
    """Spectral Poisson/Helmholtz solver for masked irregular domains.

    Uses the **capacitance matrix method** (Buzbee, Golub & Nielson 1970) to
    extend a fast rectangular spectral solver to a domain defined by a binary
    mask.

    The algorithm (Buzbee, Golub & Nielson 1970):

    1. Solve the PDE on the **full rectangle** using a fast spectral solver
       (DST/DCT/FFT), ignoring the mask.  Call this ``u``.
    2. ``u`` generally violates ψ = 0 at inner-boundary points.  Correct it:
       ``ψ = u − Σ_k α_k g_k``, where ``g_k`` are precomputed Green's
       functions (rectangular-domain response to δ-sources at each
       boundary point b_k).
    3. The coefficients α are found by requiring ψ(b_k) = 0 at all
       N_b boundary points, giving the linear system ``C α = u[B]``
       where ``C[k,l] = g_l(b_k)`` is the **capacitance matrix**.

    The capacitance correction is delegated to :class:`gaussx.CapacitanceSolver`;
    this class adds the field reshaping and the exterior masking. Construct with
    :func:`build_capacitance_solver`.

    Attributes
    ----------
    solver : gaussx.CapacitanceSolver
        The generic capacitance solver operating on flat vectors.
    mask : Float[Array, "Ny Nx"]
        Domain mask (1.0 = interior, 0.0 = exterior).  Applied to the output
        so that values outside the physical domain are exactly zero.
    shape : tuple[int, int]
        Grid shape ``(Ny, Nx)``.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.
    base_bc : str
        Spectral solver used as the rectangular base.
    """

    solver: _GaussxCapacitanceSolver
    mask: Float[Array, "Ny Nx"]
    shape: tuple[int, int] = eqx.field(static=True)
    dx: float
    dy: float
    lambda_: float = eqx.field(static=True)
    base_bc: str = eqx.field(static=True)

    def __call__(
        self,
        rhs: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − λ)ψ = rhs on the masked domain.

        The generic solver enforces ψ = 0 at the inner-boundary points and
        returns the corrected field; this method reshapes between fields and
        flat vectors and zeroes the exterior via the mask.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side on the full rectangular grid.
            Values outside the physical domain (mask = False) are ignored.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ on the full rectangular grid.  Exactly zero outside
            the mask, and ψ ≈ 0 at inner-boundary points.
        """
        ny, nx = self.shape
        # Honor the documented contract: exterior (mask = False) values are
        # ignored by zeroing them before the base solve.
        rhs_masked = rhs * self.mask
        psi_flat = self.solver(rhs_masked.reshape(ny * nx))
        return psi_flat.reshape(ny, nx) * self.mask


def build_capacitance_solver(
    mask: np.ndarray,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    base_bc: str = "fft",
) -> CapacitanceSolver:
    """Pre-compute the capacitance matrix and return a ready-to-use solver.

    Offline algorithm (Buzbee, Golub & Nielson 1970):

    1. **Detect inner boundary** — find the N_b mask-interior cells that are
       4-connected to at least one exterior cell (using ``scipy.ndimage.binary_dilation``
       with a cross-shaped structuring element).
    2. **Delegate to gaussx** — :class:`gaussx.CapacitanceSolver` computes the
       Green's functions (one rectangular base solve per boundary point), the
       capacitance matrix, and its inverse.

    Complexity
    ----------
    * Offline (this function):  O(N_b · Ny · Nx · log(Ny·Nx))  time,
      O(N_b · Ny · Nx)  memory for the Green's function matrix.
    * Online (``CapacitanceSolver.__call__``):  O(N_b² + Ny · Nx · log(Ny·Nx))
      time per solve.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (Ny, Nx)
        Physical domain mask.  ``True`` = interior (ocean/fluid),
        ``False`` = exterior (land/walls).
        Inner-boundary points are computed as wet (``True``) cells that are
        4-connected to at least one dry (``False``) cell.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Use ``0.0`` for pure Poisson.
    base_bc : {"fft", "dst", "dct"}
        Rectangular spectral solver used as the base.

    Returns
    -------
    CapacitanceSolver
        A callable equinox Module wrapping the precomputed gaussx solver.

    Raises
    ------
    ValueError
        If the mask has no inner-boundary points (e.g. all-ones mask).
    """
    from scipy.ndimage import binary_dilation

    mask_bool = np.asarray(mask, dtype=bool)
    ny, nx = mask_bool.shape

    # Inner-boundary: mask-interior cells adjacent to at least one exterior cell
    exterior = ~mask_bool
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = binary_dilation(exterior, structure=struct)
    inner_boundary = mask_bool & dilated

    j_b, i_b = np.where(inner_boundary)
    n_b = len(j_b)
    if n_b == 0:
        raise ValueError(
            "No inner-boundary points found.  Check that the mask has a "
            "non-trivial interior/exterior structure."
        )

    boundary_indices = jnp.asarray(j_b * nx + i_b)

    def base_solve(rhs_flat: Float[Array, " n"]) -> Float[Array, " n"]:
        rhs_2d = rhs_flat.reshape(ny, nx)
        return _spectral_solve(rhs_2d, dx, dy, lambda_, base_bc).reshape(ny * nx)

    gaussx_solver = _GaussxCapacitanceSolver(base_solve, boundary_indices, ny * nx)

    return CapacitanceSolver(
        solver=gaussx_solver,
        mask=jnp.array(mask_bool, dtype=float),
        shape=(ny, nx),
        dx=float(dx),
        dy=float(dy),
        lambda_=float(lambda_),
        base_bc=base_bc,
    )
