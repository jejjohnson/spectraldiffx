# ============================================================================
# Chebyshev Spectral Filters
# ============================================================================

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Num

from .grid import ChebyshevGrid1D, ChebyshevGrid2D

# Array-shape aliases:
#   "Npts"         — number of 1D Chebyshev nodes (N+1 for GL, N for Gauss)
#   "Nypts Nxpts"  — 2D tensor-product grid


def _coeff_dtype(a: Num[Array, ...]) -> type:
    """Return the real floating dtype of a coefficient array.

    Falls back to ``jnp.float32`` for non-floating dtypes so filter kernels
    always have a well-defined real dtype (e.g. for integer input).
    """
    return a.dtype if jnp.issubdtype(a.dtype, jnp.floating) else jnp.float32


class ChebyshevFilter1D(eqx.Module):
    """1D Chebyshev spectral filter for smoothing and numerical stabilisation.

    Mathematical Formulation
    ------------------------
    Filters are applied as a multiplicative mask in Chebyshev-coefficient space:

        ãₖ = F(k) · aₖ

    where aₖ are Chebyshev coefficients of u(x) = Σₖ aₖ Tₖ(x/L).

    Exponential filter
        F(k) = exp(−α (k/kₘₐₓ)ᵖ)          # kₘₐₓ = N for GL, N−1 for Gauss

    Hyperviscosity filter
        F(k) = exp(−ν_h kᵖ Δt)

    Attributes
    ----------
    grid : ChebyshevGrid1D
        Underlying 1D Chebyshev grid (provides the forward/inverse transform).

    Examples
    --------
    Smooth a noisy field with the default machine-epsilon exponential filter:

    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    >>> flt = ChebyshevFilter1D(grid=grid)
    >>> u = jnp.sin(jnp.pi * grid.x) + 1e-2 * jnp.cos(31 * jnp.pi * grid.x)
    >>> u_smooth = flt.exponential_filter(u)  # high mode is suppressed
    >>> u_damped = flt.hyperviscosity(u, nu_hyper=1e-4, dt=0.01, power=4)
    """

    grid: ChebyshevGrid1D

    def exponential_filter(
        self,
        u: Num[Array, Npts],
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Num[Array, Npts]:
        """Apply an exponential cut-off filter in Chebyshev mode space.

            F(k) = exp(−α (k/kₘₐₓ)ᵖ)

        Near unity for low k and falls off sharply near kₘₐₓ.  The default
        α = 36 gives F(kₘₐₓ) ≈ exp(−36) ≈ 2·10⁻¹⁶ (≈ double-precision ε).

        Parameters
        ----------
        u : Num[Array, "Npts"]
            Physical-space field, or Chebyshev coefficients if ``spectral=True``.
        alpha : float
            Damping strength. Default 36.0.
        power : int
            Sharpening exponent (even integer). Default 16.  Must be > 0.
        spectral : bool
            If ``True``, ``u`` is treated as Chebyshev coefficients.

        Returns
        -------
        Num[Array, "Npts"]
            Filtered field (physical if ``spectral=False``, else coefficients).
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        a = u if spectral else self.grid.transform(u)
        N = self.grid.N
        n_modes = N + 1 if self.grid.node_type == "gauss-lobatto" else N
        # Normalise by the highest mode index so F(kₘₐₓ) = exp(−α).
        k_max = max(1, n_modes - 1)
        real_dtype = _coeff_dtype(a)
        k = jnp.arange(n_modes, dtype=real_dtype)
        filter_mask = jnp.exp(-alpha * (k / k_max) ** power)
        a_f = a * filter_mask
        return a_f if spectral else self.grid.transform(a_f, inverse=True)

    def hyperviscosity(
        self,
        u: Num[Array, Npts],
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Num[Array, Npts]:
        """Apply hyperviscous damping in Chebyshev mode space.

            F(k) = exp(−ν_h kᵖ Δt)

        Simulates high-order diffusion: ∂u/∂t = (−1)^(p/2) ν_h ∂ᵖu/∂xᵖ.

        Parameters
        ----------
        u : Num[Array, "Npts"]
            Physical-space field or Chebyshev coefficients.
        nu_hyper : float
            Hyperviscosity coefficient (≥ 0).
        dt : float
            Time step for the damping (≥ 0).
        power : int
            Diffusion order.  Default 4 (biharmonic).  Must be > 0.
        spectral : bool
            If ``True``, ``u`` is treated as Chebyshev coefficients.

        Returns
        -------
        Num[Array, "Npts"]
            Damped field or coefficients.
        """
        if nu_hyper < 0:
            raise ValueError(f"nu_hyper must be >= 0, got {nu_hyper}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        a = u if spectral else self.grid.transform(u)
        N = self.grid.N
        n_modes = N + 1 if self.grid.node_type == "gauss-lobatto" else N
        real_dtype = _coeff_dtype(a)
        k = jnp.arange(n_modes, dtype=real_dtype)
        filter_mask = jnp.exp(-nu_hyper * k**power * dt)
        a_f = a * filter_mask
        return a_f if spectral else self.grid.transform(a_f, inverse=True)


class ChebyshevFilter2D(eqx.Module):
    """2D Chebyshev spectral filter on [−Lx, Lx] × [−Ly, Ly].

    Applies separable 1D exponential or hyperviscosity kernels:

        F(kₓ, kᵧ) = Fₓ(kₓ) · Fᵧ(kᵧ)

    Attributes
    ----------
    grid : ChebyshevGrid2D
        Underlying 2D Chebyshev grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid2D.from_N_L(Nx=24, Ny=24, Lx=1.0, Ly=1.0)
    >>> flt = ChebyshevFilter2D(grid=grid)
    >>> X, Y = grid.X
    >>> u = jnp.sin(2 * jnp.pi * X) * jnp.cos(3 * jnp.pi * Y)
    >>> u_smooth = flt.exponential_filter(u, alpha=20.0, power=8)
    """

    grid: ChebyshevGrid2D

    def exponential_filter(
        self,
        u: Num[Array, "Nypts Nxpts"],
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Num[Array, "Nypts Nxpts"]:
        """Separable 2D exponential filter.

            F(kₓ, kᵧ) = exp(−α (kₓ/kₓ_max)ᵖ) · exp(−α (kᵧ/kᵧ_max)ᵖ)

        Parameters
        ----------
        u : Num[Array, "Nypts Nxpts"]
            Physical-space field or Chebyshev coefficients.
        alpha : float
            Damping strength (≥ 0).  Default 36.0.
        power : int
            Sharpening exponent (> 0).  Default 16.
        spectral : bool
            If ``True``, ``u`` is treated as spectral coefficients.

        Returns
        -------
        Num[Array, "Nypts Nxpts"]
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        a = u if spectral else self.grid.transform(u)
        Nx, Ny = self.grid.Nx, self.grid.Ny
        nx_modes = Nx + 1 if self.grid.node_type == "gauss-lobatto" else Nx
        ny_modes = Ny + 1 if self.grid.node_type == "gauss-lobatto" else Ny
        # Normalise by the highest mode index; derive dtype from coefficients.
        kx_max = max(1, nx_modes - 1)
        ky_max = max(1, ny_modes - 1)
        real_dtype = _coeff_dtype(a)
        kx = jnp.arange(nx_modes, dtype=real_dtype)
        ky = jnp.arange(ny_modes, dtype=real_dtype)
        Fx = jnp.exp(-alpha * (kx / kx_max) ** power)
        Fy = jnp.exp(-alpha * (ky / ky_max) ** power)
        # Separable filter: F = Fᵧ[:, None] · Fₓ[None, :]
        a_f = a * Fy[:, None] * Fx[None, :]
        return a_f if spectral else self.grid.transform(a_f, inverse=True)

    def hyperviscosity(
        self,
        u: Num[Array, "Nypts Nxpts"],
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Num[Array, "Nypts Nxpts"]:
        """Separable 2D hyperviscosity filter.

            F(kₓ, kᵧ) = exp(−ν_h kₓᵖ Δt) · exp(−ν_h kᵧᵖ Δt)

        Parameters
        ----------
        u : Num[Array, "Nypts Nxpts"]
            Physical-space field or Chebyshev coefficients.
        nu_hyper : float
            Hyperviscosity coefficient (≥ 0).
        dt : float
            Time step for the damping (≥ 0).
        power : int
            Diffusion order (> 0).  Default 4 (biharmonic).
        spectral : bool
            If ``True``, ``u`` is treated as spectral coefficients.

        Returns
        -------
        Num[Array, "Nypts Nxpts"]
        """
        if nu_hyper < 0:
            raise ValueError(f"nu_hyper must be >= 0, got {nu_hyper}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        a = u if spectral else self.grid.transform(u)
        Nx, Ny = self.grid.Nx, self.grid.Ny
        nx_modes = Nx + 1 if self.grid.node_type == "gauss-lobatto" else Nx
        ny_modes = Ny + 1 if self.grid.node_type == "gauss-lobatto" else Ny
        real_dtype = _coeff_dtype(a)
        kx = jnp.arange(nx_modes, dtype=real_dtype)
        ky = jnp.arange(ny_modes, dtype=real_dtype)
        # Separable: exp(−ν kₓᵖ Δt) · exp(−ν kᵧᵖ Δt)
        Fx = jnp.exp(-nu_hyper * kx**power * dt)
        Fy = jnp.exp(-nu_hyper * ky**power * dt)
        a_f = a * Fy[:, None] * Fx[None, :]
        return a_f if spectral else self.grid.transform(a_f, inverse=True)
