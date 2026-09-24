# ============================================================================
# Chebyshev Derivative Operators
# ============================================================================

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from .grid import (
    ChebyshevGrid1D,
    ChebyshevGrid2D,
    ChebyshevGrid3D,
    _transform_along_axis,
)
from .transforms import (
    chebyshev_antiderivative_coeffs,
    chebyshev_derivative_coeffs,
    chebyshev_integral_coeffs,
)

DerivativeMethod = Literal["matrix", "fft"]


def _check_method(method: str) -> None:
    """Validate the ``method`` field of a derivative operator."""
    if method not in ("matrix", "fft"):
        raise ValueError(f"method must be 'matrix' or 'fft', got {method!r}")


def _diff_along_axis(
    u: Array,
    D: Array,
    N: int,
    L: float,
    node_type: str,
    axis: int,
    order: int,
    method: str,
) -> Array:
    """Apply ∂ᵒʳᵈᵉʳ/∂ξᵒʳᵈᵉʳ along one axis of an N-d nodal field.

    method = "matrix":  u ← D ×_axis u, repeated ``order`` times — O(n²)
                        per 1D line.
    method = "fft":     DCT → coefficient recurrence (``order`` times) →
                        inverse DCT — O(n log n) per 1D line.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return u
    if method == "fft":
        a = _transform_along_axis(u, N, node_type, False, axis)
        a = jnp.moveaxis(a, axis, -1)
        a = chebyshev_derivative_coeffs(a, L=L, order=order)
        a = jnp.moveaxis(a, -1, axis)
        return _transform_along_axis(a, N, node_type, True, axis)
    out = u
    for _ in range(order):
        out = jnp.moveaxis(jnp.tensordot(D, out, axes=(1, axis)), 0, axis)
    return out


# Array-shape aliases used across this module:
#   "Npts"  — number of 1D Chebyshev nodes (N+1 for Gauss-Lobatto, N for Gauss)
#   "Nypts Nxpts" — 2D tensor-product grid (y-fast inner axis is x)


class ChebyshevDerivative1D(eqx.Module):
    """1D Chebyshev derivative operator using the precomputed differentiation matrix.

    Mathematical Formulation
    ------------------------
    For a function u(x) sampled at Chebyshev nodes xⱼ on [−L, L]:

        (du/dx)ⱼ = Σₖ D_{jk} uₖ

    where D is the (N+1)×(N+1) (Gauss–Lobatto) or N×N (Gauss) differentiation
    matrix precomputed in :class:`ChebyshevGrid1D`.  Higher-order derivatives
    are matrix powers:

        d²u/dx² = D · D · u = D² · u

    Two evaluation strategies are available via ``method``:

        "matrix" (default) : Dⁿ·u by dense mat-vecs — O(N²) per derivative,
                             fastest for small N on accelerators.
        "fft"              : DCT → coefficient recurrence → inverse DCT —
                             O(N log N).  Asymptotically cheaper, but a
                             BLAS mat-vec usually wins in practice for
                             N ≲ 10³ (on CPU, "matrix" was ~4× faster at
                             N = 512 for a batch of 256 fields); benchmark
                             for your hardware and N.

    Both agree to round-off (they are the same polynomial interpolant).

    Attributes
    ----------
    grid : ChebyshevGrid1D
        1D Chebyshev grid carrying the differentiation matrix D.
    method : {"matrix", "fft"}
        Evaluation strategy (static).

    Examples
    --------
    Derivative of sin(πx) on [−1, 1]:

    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    >>> deriv = ChebyshevDerivative1D(grid=grid)
    >>> u = jnp.sin(jnp.pi * grid.x)
    >>> du_dx = deriv(u)  # ≈ π cos(πx)
    >>> d2u_dx2 = deriv(u, order=2)  # ≈ −π² sin(πx)
    """

    grid: ChebyshevGrid1D
    method: DerivativeMethod = eqx.field(default="matrix", static=True)

    def __check_init__(self) -> None:
        _check_method(self.method)

    def __call__(self, u: Num[Array, "Npts"], order: int = 1) -> Float[Array, "Npts"]:
        """Apply the n-th derivative Dⁿ to a nodal field.

        Parameters
        ----------
        u : Float[Array, "Npts"]
            Nodal values at Chebyshev nodes (Npts = N+1 for GL, N for Gauss).
        order : int
            Derivative order (≥ 0).  ``order=0`` returns a copy of ``u``.

        Returns
        -------
        Float[Array, "Npts"]
            n-th derivative at the Chebyshev nodes.
        """
        g = self.grid
        return _diff_along_axis(u, g.D, g.N, g.L, g.node_type, 0, order, self.method)

    def gradient(self, u: Num[Array, "Npts"]) -> Float[Array, "Npts"]:
        """First derivative ``du/dx`` at Chebyshev nodes."""
        return self(u, order=1)

    def laplacian(self, u: Num[Array, "Npts"]) -> Float[Array, "Npts"]:
        """Second derivative ``d²u/dx²`` at Chebyshev nodes."""
        return self(u, order=2)

    def integrate(self, u: Num[Array, "Npts"]) -> Float[Array, ""]:
        """Definite integral ∫_{−L}^{L} u(x) dx.

        Computed from the Chebyshev coefficients,

            ∫ u dx = L Σ_{k even} 2 aₖ / (1 − k²)

        which on Gauss–Lobatto nodes is exactly Clenshaw–Curtis quadrature.
        """
        return chebyshev_integral_coeffs(self.grid.transform(u), L=self.grid.L)

    def antiderivative(self, u: Num[Array, "Npts"]) -> Float[Array, "Npts"]:
        """Indefinite integral U(x) = ∫_{−L}^{x} u(s) ds at the nodes.

        See :func:`chebyshev_antiderivative_coeffs` for the recurrence.
        """
        a = self.grid.transform(u)
        B = chebyshev_antiderivative_coeffs(a, L=self.grid.L)
        return self.grid.transform(B, inverse=True)


class ChebyshevDerivative2D(eqx.Module):
    """2D Chebyshev derivative operators on [−Lx, Lx] × [−Ly, Ly].

    Mathematical Formulation
    ------------------------
    For u(x, y) on a (Nypts, Nxpts) grid with differentiation matrices Dx, Dy
    stored on the grid:

        (∂u/∂x)[j, i] = (u · Dxᵀ)[j, i]   # applied along axis 1 (x)
        (∂u/∂y)[j, i] = (Dy · u)[j, i]    # applied along axis 0 (y)

    The scalar Laplacian and 2D divergence/curl follow directly:

        ∇²u   = ∂²u/∂x² + ∂²u/∂y²
        ∇·V  = ∂vₓ/∂x + ∂vᵧ/∂y
        (∇×V)_z = ∂vᵧ/∂x − ∂vₓ/∂y

    Attributes
    ----------
    grid : ChebyshevGrid2D
        2D Chebyshev grid carrying Dx, Dy and the precomputed Dx², Dy².
    method : {"matrix", "fft"}
        Evaluation strategy (static).  "matrix" contracts each axis with
        the 1D differentiation matrix (O(N³) per 2D derivative); "fft"
        uses DCT + coefficient recurrence along each axis
        (O(N² log N)).  See :class:`ChebyshevDerivative1D`.

    Examples
    --------
    Laplacian of u(x, y) = sin(πx)·sin(πy) on [−1, 1]²:

    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid2D.from_N_L(Nx=24, Ny=24, Lx=1.0, Ly=1.0)
    >>> deriv = ChebyshevDerivative2D(grid=grid)
    >>> X, Y = grid.X
    >>> u = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    >>> lap_u = deriv.laplacian(u)  # ≈ −2π² u

    Divergence of V = (y, −x) (should be ~0):

    >>> vx, vy = Y, -X
    >>> div = deriv.divergence(vx, vy)  # ≈ 0
    """

    grid: ChebyshevGrid2D
    method: DerivativeMethod = eqx.field(default="matrix", static=True)

    def __check_init__(self) -> None:
        _check_method(self.method)

    def _dx(self, u: Array, order: int = 1) -> Array:
        """∂ᵒʳᵈᵉʳu/∂xᵒʳᵈᵉʳ (axis 1)."""
        g = self.grid
        if order == 2 and self.method == "matrix":
            return u @ g.Dx2.T  # precomputed Dx², one matmul
        return _diff_along_axis(u, g.Dx, g.Nx, g.Lx, g.node_type, 1, order, self.method)

    def _dy(self, u: Array, order: int = 1) -> Array:
        """∂ᵒʳᵈᵉʳu/∂yᵒʳᵈᵉʳ (axis 0)."""
        g = self.grid
        if order == 2 and self.method == "matrix":
            return g.Dy2 @ u  # precomputed Dy², one matmul
        return _diff_along_axis(u, g.Dy, g.Ny, g.Ly, g.node_type, 0, order, self.method)

    def gradient(
        self, u: Num[Array, "Nypts Nxpts"]
    ) -> tuple[Float[Array, "Nypts Nxpts"], Float[Array, "Nypts Nxpts"]]:
        """Partial derivatives (∂u/∂x, ∂u/∂y) of a 2D nodal field."""
        return self._dx(u), self._dy(u)

    def laplacian(self, u: Num[Array, "Nypts Nxpts"]) -> Float[Array, "Nypts Nxpts"]:
        """2D Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y².

        With ``method="matrix"`` this uses the precomputed Dx² and Dy² from
        the grid, so the per-call cost is two matrix–matrix multiplies and
        an add (no O(N³) recomputation of D²).
        """
        return self._dx(u, 2) + self._dy(u, 2)

    def divergence(
        self,
        vx: Num[Array, "Nypts Nxpts"],
        vy: Num[Array, "Nypts Nxpts"],
    ) -> Float[Array, "Nypts Nxpts"]:
        """Cartesian divergence ∇·V = ∂vₓ/∂x + ∂vᵧ/∂y."""
        return self._dx(vx) + self._dy(vy)

    def curl(
        self,
        vx: Num[Array, "Nypts Nxpts"],
        vy: Num[Array, "Nypts Nxpts"],
    ) -> Float[Array, "Nypts Nxpts"]:
        """Scalar curl ζ = ∂vᵧ/∂x − ∂vₓ/∂y (z-component of ∇×V).

        This is also the relative vorticity of the velocity field (vₓ, vᵧ).
        """
        return self._dx(vy) - self._dy(vx)

    def advection_scalar(
        self,
        vx: Num[Array, "Nypts Nxpts"],
        vy: Num[Array, "Nypts Nxpts"],
        q: Num[Array, "Nypts Nxpts"],
    ) -> Float[Array, "Nypts Nxpts"]:
        """Scalar advection (V·∇)q = vₓ·∂q/∂x + vᵧ·∂q/∂y."""
        dq_dx, dq_dy = self.gradient(q)
        return vx * dq_dx + vy * dq_dy

    def biharmonic(self, u: Num[Array, "Nypts Nxpts"]) -> Float[Array, "Nypts Nxpts"]:
        """Biharmonic operator ∇⁴u = ∇²(∇²u)."""
        return self.laplacian(self.laplacian(u))

    def hyperviscosity(
        self,
        u: Num[Array, "Nypts Nxpts"],
        nu: float,
        order: int = 2,
    ) -> Float[Array, "Nypts Nxpts"]:
        """Hyperviscous tendency (−1)ⁿ⁺¹ ν ∇²ⁿ u.

        The sign makes the operator dissipative for every n (it matches
        :meth:`SpectralDerivative2D.hyperviscosity`).  No boundary
        conditions are imposed: this is the raw collocation operator.

        Parameters
        ----------
        u : Num[Array, "Nypts Nxpts"]
            Nodal field.
        nu : float
            Hyperviscosity coefficient (≥ 0).
        order : int
            n ≥ 1 (1 = Laplacian diffusion, 2 = biharmonic).
        """
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if nu < 0:
            raise ValueError(f"nu must be >= 0, got {nu}")
        out = u
        for _ in range(order):
            out = self.laplacian(out)
        return (-1.0) ** (order + 1) * nu * out

    def vector_laplacian(
        self,
        vx: Num[Array, "Nypts Nxpts"],
        vy: Num[Array, "Nypts Nxpts"],
    ) -> tuple[Float[Array, "Nypts Nxpts"], Float[Array, "Nypts Nxpts"]]:
        """Cartesian vector Laplacian ∇²V = (∇²vₓ, ∇²vᵧ)."""
        return self.laplacian(vx), self.laplacian(vy)

    def velocity_from_streamfunction(
        self, psi: Num[Array, "Nypts Nxpts"]
    ) -> tuple[Float[Array, "Nypts Nxpts"], Float[Array, "Nypts Nxpts"]]:
        """Non-divergent velocity from a streamfunction.

            u = −∂ψ/∂y,   v = ∂ψ/∂x

        (same sign convention as :meth:`SpectralDerivative2D.velocity_from_streamfunction`),
        so that ∇·(u, v) = 0 and ζ = ∂v/∂x − ∂u/∂y = ∇²ψ.
        """
        return -self._dy(psi), self._dx(psi)

    def jacobian(
        self,
        f: Num[Array, "Nypts Nxpts"],
        g: Num[Array, "Nypts Nxpts"],
    ) -> Float[Array, "Nypts Nxpts"]:
        """Jacobian J(f, g) = ∂f/∂x·∂g/∂y − ∂f/∂y·∂g/∂x.

        Derivatives are spectral; the products are taken pointwise at the
        collocation nodes (no dealiasing — combine with
        :func:`cheb_dealias_product` if aliasing matters).  With ψ the
        streamfunction, J(ψ, q) = u·∇q is the advection of q.
        """
        df_dx, df_dy = self.gradient(f)
        dg_dx, dg_dy = self.gradient(g)
        return df_dx * dg_dy - df_dy * dg_dx

    def integrate(self, u: Num[Array, "Nypts Nxpts"]) -> Float[Array, ""]:
        """Definite integral ∫∫ u dx dy over [−Lx, Lx] × [−Ly, Ly].

        Tensor-product Chebyshev quadrature from the 2D coefficients
        (Clenshaw–Curtis on Gauss–Lobatto nodes).
        """
        g = self.grid
        a = g.transform(u)  # (ky, kx)
        ax = chebyshev_integral_coeffs(a, L=g.Lx)  # integrate x → (ky,)
        return chebyshev_integral_coeffs(ax, L=g.Ly)


class ChebyshevDerivative3D(eqx.Module):
    """3D Chebyshev derivative operators on [−Lz, Lz] × [−Ly, Ly] × [−Lx, Lx].

    Arrays have shape (Nzpts, Nypts, Nxpts) with axis order (z, y, x),
    mirroring :class:`SpectralDerivative3D`; vector-valued methods take and
    return components in the same (z, y, x) order.

    Mathematical Formulation
    ------------------------
    Each partial derivative contracts one axis with the 1D matrix:

        ∂u/∂x = u ×₂ Dx,   ∂u/∂y = u ×₁ Dy,   ∂u/∂z = u ×₀ Dz

    so a derivative costs O(Nx·Ny·Nz·N) ("matrix") or
    O(Nx·Ny·Nz·log N) ("fft").

    Attributes
    ----------
    grid : ChebyshevGrid3D
        3D Chebyshev grid.
    method : {"matrix", "fft"}
        Evaluation strategy (static).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid3D.from_N_L(Nx=12, Ny=12, Nz=12, Lx=1.0, Ly=1.0, Lz=1.0)
    >>> deriv = ChebyshevDerivative3D(grid=grid)
    >>> Z, Y, X = grid.X
    >>> u = jnp.sin(X) * jnp.cos(Y) * jnp.exp(Z)
    >>> lap = deriv.laplacian(u)  # ≈ −u
    """

    grid: ChebyshevGrid3D
    method: DerivativeMethod = eqx.field(default="matrix", static=True)

    def __check_init__(self) -> None:
        _check_method(self.method)

    def _d(self, u: Array, axis: int, order: int = 1) -> Array:
        """∂ᵒʳᵈᵉʳu along ``axis`` (0 = z, 1 = y, 2 = x)."""
        g = self.grid
        D, D2, N, L = {
            0: (g.Dz, g.Dz2, g.Nz, g.Lz),
            1: (g.Dy, g.Dy2, g.Ny, g.Ly),
            2: (g.Dx, g.Dx2, g.Nx, g.Lx),
        }[axis]
        if order == 2 and self.method == "matrix":
            return jnp.moveaxis(jnp.tensordot(D2, u, axes=(1, axis)), 0, axis)
        return _diff_along_axis(u, D, N, L, g.node_type, axis, order, self.method)

    def gradient(
        self, u: Num[Array, "Nzpts Nypts Nxpts"]
    ) -> tuple[
        Float[Array, "Nzpts Nypts Nxpts"],
        Float[Array, "Nzpts Nypts Nxpts"],
        Float[Array, "Nzpts Nypts Nxpts"],
    ]:
        """Gradient (∂u/∂z, ∂u/∂y, ∂u/∂x)."""
        return self._d(u, 0), self._d(u, 1), self._d(u, 2)

    def divergence(
        self,
        vz: Num[Array, "Nzpts Nypts Nxpts"],
        vy: Num[Array, "Nzpts Nypts Nxpts"],
        vx: Num[Array, "Nzpts Nypts Nxpts"],
    ) -> Float[Array, "Nzpts Nypts Nxpts"]:
        """Divergence ∇·V = ∂v_z/∂z + ∂vᵧ/∂y + ∂vₓ/∂x."""
        return self._d(vz, 0) + self._d(vy, 1) + self._d(vx, 2)

    def curl(
        self,
        vz: Num[Array, "Nzpts Nypts Nxpts"],
        vy: Num[Array, "Nzpts Nypts Nxpts"],
        vx: Num[Array, "Nzpts Nypts Nxpts"],
    ) -> tuple[
        Float[Array, "Nzpts Nypts Nxpts"],
        Float[Array, "Nzpts Nypts Nxpts"],
        Float[Array, "Nzpts Nypts Nxpts"],
    ]:
        """Curl ω = ∇×V returned as (ω_z, ω_y, ω_x).

        ω_z = ∂vᵧ/∂x − ∂vₓ/∂y
        ω_y = ∂vₓ/∂z − ∂v_z/∂x
        ω_x = ∂v_z/∂y − ∂vᵧ/∂z
        """
        wz = self._d(vy, 2) - self._d(vx, 1)
        wy = self._d(vx, 0) - self._d(vz, 2)
        wx = self._d(vz, 1) - self._d(vy, 0)
        return wz, wy, wx

    def laplacian(
        self, u: Num[Array, "Nzpts Nypts Nxpts"]
    ) -> Float[Array, "Nzpts Nypts Nxpts"]:
        """Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²."""
        return self._d(u, 0, 2) + self._d(u, 1, 2) + self._d(u, 2, 2)

    def biharmonic(
        self, u: Num[Array, "Nzpts Nypts Nxpts"]
    ) -> Float[Array, "Nzpts Nypts Nxpts"]:
        """Biharmonic operator ∇⁴u = ∇²(∇²u)."""
        return self.laplacian(self.laplacian(u))

    def hyperviscosity(
        self,
        u: Num[Array, "Nzpts Nypts Nxpts"],
        nu: float,
        order: int = 2,
    ) -> Float[Array, "Nzpts Nypts Nxpts"]:
        """Hyperviscous tendency (−1)ⁿ⁺¹ ν ∇²ⁿ u (dissipative for every n ≥ 1)."""
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if nu < 0:
            raise ValueError(f"nu must be >= 0, got {nu}")
        out = u
        for _ in range(order):
            out = self.laplacian(out)
        return (-1.0) ** (order + 1) * nu * out

    def vector_laplacian(
        self,
        vz: Num[Array, "Nzpts Nypts Nxpts"],
        vy: Num[Array, "Nzpts Nypts Nxpts"],
        vx: Num[Array, "Nzpts Nypts Nxpts"],
    ) -> tuple[
        Float[Array, "Nzpts Nypts Nxpts"],
        Float[Array, "Nzpts Nypts Nxpts"],
        Float[Array, "Nzpts Nypts Nxpts"],
    ]:
        """Cartesian vector Laplacian ∇²V = (∇²v_z, ∇²vᵧ, ∇²vₓ)."""
        return self.laplacian(vz), self.laplacian(vy), self.laplacian(vx)

    def velocity_from_streamfunction(
        self, psi: Num[Array, "Nzpts Nypts Nxpts"]
    ) -> tuple[Float[Array, "Nzpts Nypts Nxpts"], Float[Array, "Nzpts Nypts Nxpts"]]:
        """Horizontal velocity (u, v) = (−∂ψ/∂y, ∂ψ/∂x) at every level."""
        return -self._d(psi, 1), self._d(psi, 2)

    def jacobian(
        self,
        f: Num[Array, "Nzpts Nypts Nxpts"],
        g: Num[Array, "Nzpts Nypts Nxpts"],
    ) -> Float[Array, "Nzpts Nypts Nxpts"]:
        """Horizontal Jacobian J(f, g) = ∂f/∂x·∂g/∂y − ∂f/∂y·∂g/∂x at every level."""
        return self._d(f, 2) * self._d(g, 1) - self._d(f, 1) * self._d(g, 2)

    def integrate(self, u: Num[Array, "Nzpts Nypts Nxpts"]) -> Float[Array, ""]:
        """Definite integral ∫∫∫ u dV over the box (tensor-product quadrature)."""
        g = self.grid
        a = g.transform(u)  # (kz, ky, kx)
        a = chebyshev_integral_coeffs(a, L=g.Lx)  # → (kz, ky)
        a = chebyshev_integral_coeffs(a, L=g.Ly)  # → (kz,)
        return chebyshev_integral_coeffs(a, L=g.Lz)
