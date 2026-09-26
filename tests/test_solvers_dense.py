"""Spectral Helmholtz solvers against dense finite-difference matrices (gh-119).

Each per-axis boundary condition has a dense 1-D second-order (FD2)
Laplacian built from its ghost-point closure. On an axis with spacing h the
interior rows are [1, -2, 1] / h², and the boundary rows are closed as follows:

==========================================  ==========================
BC (left, right)                            boundary row closure
==========================================  ==========================
``periodic``                                wrap-around neighbour
``dirichlet`` (vertex, ghost = 0)           [-2, 1]
``dirichlet_stag`` (cell, ghost = -ψ₀)      [-3, 1]
``neumann`` (vertex, mirror ghost ψ₋₁ = ψ₁)  [-2, 2]
``neumann_stag`` (cell, ghost = ψ₀)         [-1, 1]
==========================================  ==========================

The mixed tuples combine the two sides. These matrices are the reference for
both checks below:

* ``test_fd2_eigenpairs``: the transform basis diagonalises the matrix with
  the eigenvalues the package uses, pairing each basis vector with its
  eigenvalue, not just matching the sorted spectrum.
* ``test_solve_helmholtz_2d_matches_dense`` / ``_3d_``: the solver agrees
  with a dense Kronecker-sum solve.

Singular pairs (every axis periodic or Neumann, with λ = 0) are checked by
the residual on a right-hand side in the range of the operator, which does
not depend on the gauge the solver picks. The dense sweep is also the
reference for the planned kernel consolidation (#106).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx import solve_helmholtz_2d, solve_helmholtz_3d
from spectraldiffx._src.fourier.solvers import _BC_DISPATCH, _inverse_1d

BCS = list(_BC_DISPATCH)
_NULL_BCS = {"periodic", "neumann", "neumann_stag"}


def _bc_id(bc) -> str:
    return bc if isinstance(bc, str) else "-".join(bc)


def dense_laplacian_1d(bc, n: int, h: float) -> np.ndarray:
    """Dense FD2 Laplacian for one axis, with the closure in the table above."""
    A = -2.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
    left, right = (bc, bc) if isinstance(bc, str) else bc
    if left == "periodic":
        A[0, -1] += 1.0
        A[-1, 0] += 1.0
    else:
        A[0, 0] += {"dirichlet_stag": -1.0, "neumann_stag": 1.0}.get(left, 0.0)
        A[-1, -1] += {"dirichlet_stag": -1.0, "neumann_stag": 1.0}.get(right, 0.0)
        if left == "neumann":
            A[0, 1] = 2.0
        if right == "neumann":
            A[-1, -2] = 2.0
    return A / h**2


def dense_operator(bcs, ns, hs, lam) -> np.ndarray:
    """Kronecker sum over axes (row-major: axis 0 slowest) minus λI."""
    total = int(np.prod(ns))
    A = -lam * np.eye(total)
    for axis, (bc, n, h) in enumerate(zip(bcs, ns, hs, strict=True)):
        left = int(np.prod(ns[:axis]))
        right = int(np.prod(ns[axis + 1 :]))
        A += np.kron(np.kron(np.eye(left), dense_laplacian_1d(bc, n, h)), np.eye(right))
    return A


def _is_singular(bcs, lam) -> bool:
    return lam == 0.0 and all(isinstance(bc, str) and bc in _NULL_BCS for bc in bcs)


def _check_against_dense(solve, bcs, ns, hs, lam, seed):
    """Compare ``solve(f)`` with the dense solve, or check the residual."""
    rng = np.random.default_rng(seed)
    A = dense_operator(bcs, ns, hs, lam)
    scale = np.abs(A).max()
    if _is_singular(bcs, lam):
        f = A @ rng.standard_normal(A.shape[0])  # in the range of A
        psi = np.asarray(solve(jnp.asarray(f.reshape(ns)))).reshape(-1)
        residual = np.abs(A @ psi - f).max() / (scale * np.abs(psi).max())
        assert residual < 1e-12
    else:
        f = rng.standard_normal(A.shape[0])
        psi = np.asarray(solve(jnp.asarray(f.reshape(ns)))).reshape(-1)
        expected = np.linalg.solve(A, f)
        assert np.abs(psi - expected).max() < 1e-10 * np.abs(expected).max()


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9, 16])
@pytest.mark.parametrize("bc", BCS, ids=_bc_id)
def test_fd2_eigenpairs(bc, n):
    """Inverse-transform basis vector k is an eigenvector with eigenvalue λ_k."""
    family, type_, eig_fn, _ = _BC_DISPATCH[bc]
    h = 0.7
    A = dense_laplacian_1d(bc, n, h)
    lam = np.asarray(eig_fn(n, h))
    V = np.stack(
        [np.asarray(_inverse_1d(jnp.eye(n)[k], family, type_, 0)) for k in range(n)],
        axis=1,
    )
    scale = np.abs(lam).max()
    assert np.abs(A @ V - V * lam[None, :]).max() < 1e-12 * scale
    dense_spectrum = np.sort(np.linalg.eigvals(A).real)
    assert np.abs(dense_spectrum - np.sort(lam)).max() < 1e-12 * scale


# Two shapes per pair: (even, odd) and (odd, even), anisotropic spacing.
_SHAPES_2D = [((6, 7), (0.9, 0.6)), ((5, 8), (0.6, 0.9))]


@pytest.mark.parametrize("lam", [0.0, 0.7])
@pytest.mark.parametrize("shape_h", _SHAPES_2D, ids=["even-odd", "odd-even"])
@pytest.mark.parametrize(
    ("bc_y", "bc_x"),
    [
        # The 9 same-BC pairs run in the fast lane; the 72 mixed pairs are
        # `slow` (the full sweep takes well over 3 s).
        pytest.param(
            bc_y,
            bc_x,
            id=f"{_bc_id(bc_y)}-{_bc_id(bc_x)}",
            marks=() if bc_y == bc_x else pytest.mark.slow,
        )
        for bc_y in BCS
        for bc_x in BCS
    ],
)
def test_solve_helmholtz_2d_matches_dense(bc_y, bc_x, shape_h, lam):
    (ny, nx), (dy, dx) = shape_h

    def solve(f):
        return solve_helmholtz_2d(f, dx, dy, bc_x=bc_x, bc_y=bc_y, lambda_=lam)

    _check_against_dense(solve, (bc_y, bc_x), (ny, nx), (dy, dx), lam, seed=0)


# Each transform family (FFT, DST-I..IV, DCT-I..IV) appears at least once,
# on axes of different lengths and spacings.
_COMBOS_3D = [
    ("periodic", "dirichlet", "neumann"),
    ("dirichlet_stag", "neumann_stag", "periodic"),
    (("dirichlet_stag", "neumann_stag"), "periodic", ("dirichlet", "neumann")),
    (("neumann_stag", "dirichlet_stag"), ("neumann", "dirichlet"), "dirichlet"),
    ("neumann", "neumann_stag", "periodic"),
    ("periodic", "periodic", "periodic"),
    ("neumann_stag", ("dirichlet", "neumann"), "dirichlet_stag"),
    (("neumann", "dirichlet"), "dirichlet", ("dirichlet_stag", "neumann_stag")),
]


@pytest.mark.parametrize("lam", [0.0, 0.7])
@pytest.mark.parametrize(
    "bcs", _COMBOS_3D, ids=["-".join(_bc_id(b) for b in c) for c in _COMBOS_3D]
)
def test_solve_helmholtz_3d_matches_dense(bcs, lam):
    bc_z, bc_y, bc_x = bcs
    ns, hs = (4, 5, 6), (1.1, 0.8, 0.6)

    def solve(f):
        return solve_helmholtz_3d(
            f, hs[2], hs[1], hs[0], bc_x=bc_x, bc_y=bc_y, bc_z=bc_z, lambda_=lam
        )

    _check_against_dense(solve, bcs, ns, hs, lam, seed=1)
