"""Capacitance matrix solver for masked/irregular domains.

Extends a fast rectangular spectral solver (DST/DCT/FFT) to domains that are
subsets of a rectangle (e.g. ocean basins with land masks) using the classic
Sherman-Morrison correction via boundary Green's functions.

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

    Construct with :func:`build_capacitance_solver`.

    Attributes
    ----------
    _C_inv : Float[Array, "Nb Nb"]
        Pre-inverted capacitance matrix.
    _green_flat : Float[Array, "Nb NyNx"]
        Green's functions (one row per boundary point), stored flat.
    _j_b : Array
        Row indices of inner-boundary points.
    _i_b : Array
        Column indices of inner-boundary points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.
    base_bc : str
        Spectral solver used as the rectangular base.
    """

    _C_inv: Float[Array, "Nb Nb"]
    _green_flat: Float[Array, "Nb NyNx"]
    _j_b: Array
    _i_b: Array
    dx: float
    dy: float
    lambda_: float = eqx.field(static=True)
    base_bc: str = eqx.field(static=True)

    def __call__(
        self,
        rhs: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − λ)ψ = rhs on the masked domain.

        Given the precomputed capacitance matrix C⁻¹ and Green's functions
        G, the online solve proceeds in four steps:

        1. Rectangular solve:   u = L_rect⁻¹ rhs              [Ny, Nx]
        2. Sample boundary:     u_B = u[j_b, i_b]             [N_b]
        3. Correction weights:  α = C⁻¹ · u_B                 [N_b]
        4. Subtract correction: ψ = u − Σ_k α_k g_k          [Ny, Nx]

        The result satisfies ψ(b_k) ≈ 0 at all N_b inner-boundary points
        and (∇² − λ)ψ = rhs at interior points.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side on the full rectangular grid.
            Values outside the physical domain (mask = False) are ignored.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ on the full rectangular grid.  Satisfies ψ ≈ 0 at
            inner-boundary points and outside the mask.
        """
        Ny, Nx = rhs.shape
        # Step 1: rectangular spectral solve
        u = _spectral_solve(rhs, self.dx, self.dy, self.lambda_, self.base_bc)
        # Step 2: values of u at inner-boundary points
        u_b = u[self._j_b, self._i_b]  # [Nb]
        # Step 3: correction coefficients  alpha = C^{-1} u_b
        alpha = self._C_inv @ u_b  # [Nb]
        # Step 4: correction field  sum_k alpha_k g_k
        correction = (self._green_flat.T @ alpha).reshape(Ny, Nx)  # [Ny, Nx]
        return u - correction


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
    2. **Compute Green's functions** — for each boundary point b_k, solve
       ``L_rect g_k = e_{b_k}`` on the full rectangle using the base spectral
       solver.  Stores G as a [N_b, Ny*Nx] matrix.
    3. **Build capacitance matrix** — ``C[k, l] = g_l(b_k)``, i.e. the
       response at boundary point k due to a unit source at boundary point l.
       Shape: [N_b, N_b].
    4. **Invert** — ``C⁻¹ = inv(C)`` via dense NumPy (offline, not JIT-traced).

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
        A callable equinox Module with all precomputed arrays baked in.

    Raises
    ------
    ValueError
        If the mask has no inner-boundary points (e.g. all-ones mask).
    """
    from scipy.ndimage import binary_dilation

    mask_bool = np.asarray(mask, dtype=bool)
    Ny, Nx = mask_bool.shape

    # Inner-boundary: mask-interior cells adjacent to at least one exterior cell
    exterior = ~mask_bool
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = binary_dilation(exterior, structure=struct)
    inner_boundary = mask_bool & dilated

    j_b, i_b = np.where(inner_boundary)
    N_b = len(j_b)
    if N_b == 0:
        raise ValueError(
            "No inner-boundary points found.  Check that the mask has a "
            "non-trivial interior/exterior structure."
        )

    # Helper: one rectangular spectral solve (numpy interface)
    def _base_solve_np(f_2d: np.ndarray) -> np.ndarray:
        f_jax = jnp.array(f_2d, dtype=float)
        result = _spectral_solve(f_jax, dx, dy, lambda_, base_bc)
        return np.array(result)

    # Green's functions: G[k] = solution to L_rect g_k = e_{b_k}
    green = np.zeros((N_b, Ny, Nx), dtype=float)
    for k in range(N_b):
        e_k = np.zeros((Ny, Nx), dtype=float)
        e_k[j_b[k], i_b[k]] = 1.0
        green[k] = _base_solve_np(e_k)

    # Capacitance matrix C[k, l] = green[l] evaluated at boundary point b_k
    C = green[:, j_b, i_b].T  # [N_b, N_b]
    C_inv = np.linalg.inv(C)

    return CapacitanceSolver(
        _C_inv=jnp.array(C_inv),
        _green_flat=jnp.array(green.reshape(N_b, Ny * Nx)),
        _j_b=jnp.array(j_b),
        _i_b=jnp.array(i_b),
        dx=float(dx),
        dy=float(dy),
        lambda_=float(lambda_),
        base_bc=base_bc,
    )
