"""
Spherical Spectral Solvers
============================

Spectral elliptic solvers on the sphere using eigenvalue inversion in
spherical-harmonic space.  The Laplace–Beltrami eigenvalues
λₗ = −l(l+1)/R² diagonalise the operator, so each SHT mode can be
inverted independently.

Poisson on the sphere
    ∇²φ = f   ⇒   φ̂(l, m) = −f̂(l, m) / [l(l+1)/R²]   (l ≥ 1)

Helmholtz on the sphere
    (∇² − α) φ = f   ⇒   φ̂(l, m) = −f̂(l, m) / [l(l+1)/R² + α]

Gauge and compatibility
-----------------------
For pure Poisson (α = 0) the mean mode (l = 0) is indeterminate; the
solver sets it to zero by default, which corresponds to the zero-mean
gauge: ∫_S² φ dΩ = 0.  This also enforces the solvability condition
∫ f dΩ = 0 implicitly (any non-zero mean of f is discarded).  Use
``zero_mean=False`` only when α > 0.

Vorticity / streamfunction solver
    The vorticity ζ = (∇×V)·r̂ on the sphere satisfies ∇²ψ = ζ with
    streamfunction ψ.  :class:`SphericalVorticityInversionSolver` is a
    thin wrapper that also returns the horizontal velocity V = ẑ × ∇ψ.

Divergence / velocity-potential solver
    For horizontal divergence δ = ∇·V, ∇²χ = δ with velocity potential χ.
    :class:`SphericalDivergenceInversionSolver` returns both χ and the
    irrotational velocity V = ∇χ.

References
----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[4] Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from .grid import SphericalGrid1D, SphericalGrid2D
from .operators import SphericalDerivative2D

# Array-shape aliases:
#   "N"          — 1D GL latitude grid (scalar fields)
#   "Nlat Nlon"  — 2D lat-lon grid / SHT coefficients


def _sphere_radius(grid: SphericalGrid1D | SphericalGrid2D) -> Float[Array, ""]:
    """Infer sphere radius R from ``grid.L`` (1D) or ``grid.Ly`` (2D)."""
    L = grid.L if isinstance(grid, SphericalGrid1D) else grid.Ly
    return L / jnp.pi


class SphericalPoissonSolver(eqx.Module):
    """Spectral Poisson solver on the sphere:  ∇²φ = f.

    In SHT-coefficient space the mode-by-mode inversion is

        φ̂(l, m) = −f̂(l, m) · [l(l+1)/R²]⁻¹    (l ≥ 1)

    The l=0 mode is set to zero when ``zero_mean=True`` (default) since
    ∇² annihilates constants on the sphere.

    Attributes
    ----------
    grid : SphericalGrid1D or SphericalGrid2D
        Underlying spherical grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = SphericalGrid2D.from_N_L(Nx=32, Ny=16)
    >>> solver = SphericalPoissonSolver(grid=grid)
    >>> PHI, THETA = grid.X
    >>> # Laplacian of cos(θ) is −2 cos(θ)/R², so Poisson RHS is that:
    >>> R = grid.Ly / jnp.pi
    >>> f = -2.0 * jnp.cos(THETA) / R**2
    >>> phi = solver.solve(f)  # ≈ cos(θ) up to an additive constant
    """

    grid: SphericalGrid1D | SphericalGrid2D

    def solve(
        self,
        f: Num[Array, ...],
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Float[Array, ...]:
        """Solve ∇²φ = f on the sphere.

        Parameters
        ----------
        f : Num[Array, ...]
            Source field.  Shape ``(N,)`` for 1D or ``(Nlat, Nlon)`` for 2D.
        zero_mean : bool
            If ``True``, pin the l=0 mode of φ to zero (gauge fix).  Required
            for well-posedness of Poisson on the sphere.
        spectral : bool
            If ``True``, ``f`` is already a DLT/SHT coefficient array.

        Returns
        -------
        Float[Array, ...]
            Solution in physical space (same shape as ``f``).
        """
        R = _sphere_radius(self.grid)
        f_hat = f if spectral else self.grid.transform(f)
        l = self.grid.l
        eigenval = l * (l + 1) / (R**2)

        if isinstance(self.grid, SphericalGrid1D):
            # Guard l=0 division; the mode is overwritten immediately below.
            denom = jnp.where(eigenval == 0.0, 1.0, eigenval)
            phi_hat = -f_hat / denom
            if zero_mean:
                phi_hat = jnp.where(l == 0.0, 0.0, phi_hat)
        else:
            denom = jnp.where(eigenval[:, None] == 0.0, 1.0, eigenval[:, None])
            phi_hat = -f_hat / denom
            if zero_mean:
                phi_hat = jnp.where(l[:, None] == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True)


class SphericalHelmholtzSolver(eqx.Module):
    """Spectral Helmholtz solver on the sphere:  (∇² − α) φ = f.

    In SHT-coefficient space:

        φ̂(l, m) = −f̂(l, m) / [l(l+1)/R² + α]

    Non-singular for α > 0; for α = 0 this reduces to Poisson and the
    l=0 gauge (zero-mean) is enforced by default.

    Attributes
    ----------
    grid : SphericalGrid1D or SphericalGrid2D
        Underlying spherical grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = SphericalGrid2D.from_N_L(Nx=32, Ny=16)
    >>> solver = SphericalHelmholtzSolver(grid=grid)
    >>> PHI, THETA = grid.X
    >>> R = grid.Ly / jnp.pi
    >>> alpha = 4.0
    >>> # For φ = cos θ: (∇² − α) φ = (−2/R² − α) cos θ
    >>> f = (-2.0 / R**2 - alpha) * jnp.cos(THETA)
    >>> phi = solver.solve(f, alpha=alpha, zero_mean=False)  # ≈ cos(θ)
    """

    grid: SphericalGrid1D | SphericalGrid2D

    def solve(
        self,
        f: Num[Array, ...],
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Float[Array, ...]:
        """Solve (∇² − α) φ = f on the sphere.

        Parameters
        ----------
        f : Num[Array, ...]
            Source field (1D ``(N,)`` or 2D ``(Nlat, Nlon)``).
        alpha : float
            Helmholtz parameter (≥ 0).  α=0 falls back to Poisson.
        zero_mean : bool
            If ``True`` and α = 0, enforce the l=0 gauge by zeroing the
            mean of φ.  Ignored effectively when α > 0 (non-singular).
        spectral : bool
            If ``True``, ``f`` is already a DLT/SHT coefficient array.

        Returns
        -------
        Float[Array, ...]
            Solution in physical space.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        R = _sphere_radius(self.grid)
        f_hat = f if spectral else self.grid.transform(f)
        l = self.grid.l
        eigenval = l * (l + 1) / (R**2)

        if isinstance(self.grid, SphericalGrid1D):
            denom = eigenval + alpha
            denom_safe = jnp.where(denom == 0.0, 1.0, denom)
            phi_hat = -f_hat / denom_safe
            if zero_mean:
                phi_hat = jnp.where(l == 0.0, 0.0, phi_hat)
        else:
            denom = eigenval[:, None] + alpha
            denom_safe = jnp.where(denom == 0.0, 1.0, denom)
            phi_hat = -f_hat / denom_safe
            if zero_mean:
                phi_hat = jnp.where(l[:, None] == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True)


# ============================================================================
# Vorticity / divergence inversion — staples of spherical GFD dynamics
# ============================================================================


class SphericalVorticityInversionSolver(eqx.Module):
    """Vorticity-inversion solver on the sphere.

    Given scalar vorticity ζ = (∇×V)·r̂, solves for the streamfunction ψ
    via Poisson

        ∇²ψ = ζ

    and returns the rotational (non-divergent) velocity field

        V = ẑ × ∇ψ       ⇔       V_θ = −(1/sin θ) ∂ψ/∂φ ·(1/R),
                                   V_φ = (1/R) ∂ψ/∂θ.

    This is the canonical vorticity–streamfunction inversion used in
    barotropic and quasigeostrophic spherical models.

    Attributes
    ----------
    grid : SphericalGrid2D
        Underlying 2D lat-lon grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = SphericalGrid2D.from_N_L(Nx=32, Ny=16)
    >>> solver = SphericalVorticityInversionSolver(grid=grid)
    >>> # zonal vorticity ζ = −2 cos θ / R² corresponds to ψ = cos θ
    >>> PHI, THETA = grid.X
    >>> R = grid.Ly / jnp.pi
    >>> zeta = -2.0 * jnp.cos(THETA) / R**2
    >>> psi, (v_theta, v_phi) = solver.solve(zeta)
    """

    grid: SphericalGrid2D

    def solve(
        self,
        zeta: Num[Array, "Nlat Nlon"],
        spectral: bool = False,
    ) -> tuple[
        Float[Array, "Nlat Nlon"],
        tuple[Float[Array, "Nlat Nlon"], Float[Array, "Nlat Nlon"]],
    ]:
        """Solve ∇²ψ = ζ and recover the rotational velocity V = ẑ × ∇ψ.

        Parameters
        ----------
        zeta : Num[Array, "Nlat Nlon"]
            Vorticity in physical space (or spectral if ``spectral=True``).
        spectral : bool
            If ``True``, treat ``zeta`` as SHT coefficients.

        Returns
        -------
        (psi, (v_theta, v_phi))
            Streamfunction ψ in physical space, and the tangent velocity
            field decomposed into (colatitude, longitude) components.
        """
        poisson = SphericalPoissonSolver(grid=self.grid)
        psi = poisson.solve(zeta, zero_mean=True, spectral=spectral)
        # V = ẑ × ∇ψ   ⇒   V_θ = −∇_φ ψ,  V_φ = +∇_θ ψ.
        deriv = SphericalDerivative2D(grid=self.grid)
        grad_theta_psi, grad_phi_psi = deriv.gradient(psi)
        v_theta = -grad_phi_psi
        v_phi = grad_theta_psi
        return psi, (v_theta, v_phi)


class SphericalDivergenceInversionSolver(eqx.Module):
    """Divergence-inversion solver on the sphere.

    Given horizontal divergence δ = ∇·V, solves for the velocity
    potential χ via Poisson

        ∇²χ = δ

    and returns the irrotational (curl-free) velocity field

        V = ∇χ       ⇔       V_θ = (1/R) ∂χ/∂θ,  V_φ = (1/(R sin θ)) ∂χ/∂φ.

    Attributes
    ----------
    grid : SphericalGrid2D
        Underlying 2D lat-lon grid.

    Examples
    --------
    >>> grid = SphericalGrid2D.from_N_L(Nx=32, Ny=16)
    >>> solver = SphericalDivergenceInversionSolver(grid=grid)
    >>> delta = ...  # horizontal divergence field  # doctest: +SKIP
    >>> chi, (v_theta, v_phi) = solver.solve(delta)  # doctest: +SKIP
    """

    grid: SphericalGrid2D

    def solve(
        self,
        delta: Num[Array, "Nlat Nlon"],
        spectral: bool = False,
    ) -> tuple[
        Float[Array, "Nlat Nlon"],
        tuple[Float[Array, "Nlat Nlon"], Float[Array, "Nlat Nlon"]],
    ]:
        """Solve ∇²χ = δ and recover the irrotational velocity V = ∇χ."""
        poisson = SphericalPoissonSolver(grid=self.grid)
        chi = poisson.solve(delta, zero_mean=True, spectral=spectral)
        deriv = SphericalDerivative2D(grid=self.grid)
        grad_theta_chi, grad_phi_chi = deriv.gradient(chi)
        return chi, (grad_theta_chi, grad_phi_chi)


# ============================================================================
# Helmholtz decomposition of a horizontal vector field
# ============================================================================


class SphericalHelmholtzDecomposition(eqx.Module):
    """Helmholtz decomposition of a horizontal vector field on the sphere.

    Given a tangent field V = (V_θ, V_φ) on the sphere, decomposes

        V = ∇χ + ẑ × ∇ψ

    into a curl-free part (velocity potential χ) and a divergence-free
    part (streamfunction ψ).  The scalar potentials are obtained by
    inverting the horizontal Laplacian applied to the divergence and
    vorticity of V:

        ∇²χ = ∇·V = δ
        ∇²ψ = (∇×V)·r̂ = ζ

    This is the spherical analogue of the classical Helmholtz
    decomposition for 2D incompressible/irrotational flow, and is the
    foundation of vorticity–divergence spectral GFD models.

    Accuracy note
    -------------
    The intermediate divergence and curl rely on
    :class:`SphericalDerivative2D`, whose colatitude derivative uses a
    1D Legendre transform column-by-column.  That is exact only for
    zonal (m = 0) modes; for m ≠ 0 modes there is a small truncation
    error proportional to the resolved smoothness of V.  The
    Laplace–Beltrami inversion itself is spectrally exact.  Pipelines
    that require machine-precision round-trip reconstruction of
    arbitrary fields should first project V onto its vorticity /
    divergence in SHT space (future work).

    Attributes
    ----------
    grid : SphericalGrid2D
        Underlying 2D lat-lon grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = SphericalGrid2D.from_N_L(Nx=32, Ny=16)
    >>> decomp = SphericalHelmholtzDecomposition(grid=grid)
    >>> v_theta = jnp.zeros((grid.Ny, grid.Nx))
    >>> v_phi = jnp.sin(grid.y)[:, None] * jnp.ones((grid.Ny, grid.Nx))
    >>> psi, chi, v_rot, v_div = decomp.decompose(v_theta, v_phi)
    """

    grid: SphericalGrid2D

    def decompose(
        self,
        v_theta: Num[Array, "Nlat Nlon"],
        v_phi: Num[Array, "Nlat Nlon"],
    ) -> tuple[
        Float[Array, "Nlat Nlon"],
        Float[Array, "Nlat Nlon"],
        tuple[Float[Array, "Nlat Nlon"], Float[Array, "Nlat Nlon"]],
        tuple[Float[Array, "Nlat Nlon"], Float[Array, "Nlat Nlon"]],
    ]:
        """Compute (ψ, χ, V_rot, V_div) for a tangent vector field.

        Parameters
        ----------
        v_theta : Num[Array, "Nlat Nlon"]
            Colatitude component of V (physical space).
        v_phi : Num[Array, "Nlat Nlon"]
            Longitude component of V (physical space).

        Returns
        -------
        psi : Float[Array, "Nlat Nlon"]
            Streamfunction (divergence-free potential).
        chi : Float[Array, "Nlat Nlon"]
            Velocity potential (curl-free potential).
        v_rot : (Float[Array, "Nlat Nlon"], Float[Array, "Nlat Nlon"])
            Divergence-free velocity components (V_θ, V_φ) = ẑ × ∇ψ.
        v_div : (Float[Array, "Nlat Nlon"], Float[Array, "Nlat Nlon"])
            Curl-free velocity components (V_θ, V_φ) = ∇χ.
        """
        deriv = SphericalDerivative2D(grid=self.grid)
        zeta = deriv.curl(v_theta, v_phi)
        delta = deriv.divergence(v_theta, v_phi)

        rot = SphericalVorticityInversionSolver(grid=self.grid)
        div = SphericalDivergenceInversionSolver(grid=self.grid)
        psi, v_rot = rot.solve(zeta)
        chi, v_div = div.solve(delta)
        return psi, chi, v_rot, v_div
