"""
Spherical Harmonic Grid Module
================================

Grid classes for pseudo-spectral methods on the sphere.  The latitude
direction uses Gauss–Legendre quadrature (non-uniform, exact for
polynomials) while the longitude direction uses a uniform Fourier grid.

Conventions
-----------
    • Colatitude θ ∈ (0, π): θ = 0 at the North Pole, π at the South Pole.
    • μ = cos θ ∈ [−1, 1]: natural coordinate for Legendre polynomials.
    • Gauss–Legendre nodes never include the poles, so sin θ > 0 everywhere.
    • Longitude φ ∈ [0, 2π): uniform, FFT-ready.

ALP normalisation (Schmidt semi-normalised)

    P̃ₗᵐ(μ) = Nₗ,ₘ · Pₗᵐ(μ),    Nₗ,ₘ = √((2l+1)/2 · (l−m)! / (l+m)!)

which makes the columns orthonormal against the GL weights:

    Σⱼ wⱼ · P̃ₗᵐ(μⱼ) · P̃_{l'}^m(μⱼ) = δ_{l, l'}

Stability envelope
    :func:`_alp_tables` builds P̃ₗᵐ with the normalised three-term recurrence
    (Holmes & Featherstone 2002), never forming the unnormalised Pₗᵐ, which
    overflows float64 near m ≈ 85 (gh-96).  It stays in float64 range up to
    about l ≈ 1500; beyond that the sectoral seed P̃ₘᵐ ∝ sinᵐθ underflows
    near the poles.

References
----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[2] Trefethen, L. N. (2000). Spectral Methods in MATLAB.
[3] Canuto et al. (2006). Spectral Methods: Fundamentals.
[4] Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float
import numpy as np


def _gauss_legendre_nodes_weights(N: int):
    """
    Compute Gauss-Legendre nodes and weights via scipy.

    Parameters
    ----------
    N : int
        Number of quadrature points.

    Returns
    -------
    nodes : ndarray [N]
        GL nodes (cos(theta)) ordered North to South (1 to -1).
    weights : ndarray [N]
        GL weights summing to 2.  Ordered to match nodes.
    """
    from scipy.special import roots_legendre

    nodes, weights = roots_legendre(N)
    # roots_legendre returns ascending order (-1 to 1); reverse to North-South.
    return nodes[::-1].copy(), weights[::-1].copy()


def _legendre_matrix(l_values: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Compute the normalised Legendre polynomial matrix.

    P_norm[l_idx, j] = sqrt((2*l+1)/2) * P_l(mu_j)

    This normalisation makes the columns orthonormal with respect to the
    Gauss-Legendre quadrature weights:
        sum_j w_j * P_norm[l, j] * P_norm[l', j] = delta_{l, l'}

    Parameters
    ----------
    l_values : ndarray [Nl]
        Legendre degree indices (integer values 0, 1, ..., N-1).
    mu : ndarray [Ny]
        Gauss-Legendre nodes (cos(theta) values).

    Returns
    -------
    P : ndarray [Nl, Ny]
        Normalised Legendre polynomial matrix.
    """
    from scipy.special import eval_legendre

    Nl = len(l_values)
    Ny = len(mu)
    P = np.zeros((Nl, Ny), dtype=np.float64)
    for i, l in enumerate(l_values):
        norm = np.sqrt((2 * l + 1) / 2.0)
        P[i, :] = norm * eval_legendre(l, mu)
    return P


def _alp_tables(m_values: np.ndarray, l_max: int, mu: np.ndarray) -> np.ndarray:
    """
    Compute normalised associated Legendre functions by the stable recurrence.

    Returns P[i, l, j] = P̃ₗᵐ(μⱼ) for m = m_values[i] and l = 0, …, l_max,
    with P̃ₗᵐ = Nₗ,ₘ · Pₗᵐ (Condon–Shortley phase included, as in
    ``scipy.special.lpmv``) and Nₗ,ₘ = √((2l+1)/2 · (l−m)!/(l+m)!), so that
    ∫₋₁¹ P̃ₗᵐ P̃ₗ′ᵐ dμ = δₗₗ′.  Rows with l < m are zero.

    The normalised functions are built directly, so no unnormalised Pₗᵐ (which
    grows like (2m−1)!! and overflows float64 near m ≈ 85, gh-96) and no
    factorial ratio is ever formed:

        P̃₀⁰ = 1/√2
        P̃ₘᵐ = −√((2m+1)/(2m)) · s · P̃ₘ₋₁ᵐ⁻¹,          s = √(1 − μ²)
        P̃ₘ₊₁ᵐ = √(2m+3) · μ · P̃ₘᵐ
        P̃ₗᵐ = aₗₘ (μ P̃ₗ₋₁ᵐ − bₗₘ P̃ₗ₋₂ᵐ),
            aₗₘ = √((4l²−1)/(l²−m²)),  bₗₘ = √(((l−1)²−m²)/(4(l−1)²−1))

    (Holmes & Featherstone 2002, "standard forward column" recurrence).  In
    float64 the sectoral seed P̃ₘᵐ ∝ sᵐ stays in range up to about l ≈ 1500;
    beyond that it underflows near the poles.

    Parameters
    ----------
    m_values : ndarray [Nm]
        Orders m >= 0 to tabulate (any order, repeats allowed).
    l_max : int
        Maximum degree, inclusive.
    mu : ndarray [Ny]
        Evaluation points μ = cos θ in [−1, 1].

    Returns
    -------
    P : ndarray [Nm, l_max + 1, Ny]
        Normalised ALP tables, float64.

    References
    ----------
    Holmes, S. A. and Featherstone, W. E. (2002). A unified approach to the
    Clenshaw summation and the recursive computation of very high degree and
    order normalised associated Legendre functions. J. Geodesy 76, 279–299.

    Examples
    --------
    >>> import numpy as np
    >>> P = _alp_tables(np.array([0, 2]), 3, np.array([0.0, 0.5]))
    >>> P.shape
    (2, 4, 2)
    >>> round(float(P[0, 0, 0]), 6)  # P̃₀⁰ = 1/√2
    0.707107
    """
    m_values = np.asarray(m_values, dtype=int)
    mu = np.asarray(mu, dtype=np.float64)
    s = np.sqrt(np.clip(1.0 - mu**2, 0.0, None))  # (Ny,)
    n = l_max + 1
    m_top = int(m_values.max(initial=0))

    # Sectoral values P̃ₘᵐ for m = 0, …, m_top by a running product over m.
    sectoral = np.empty((m_top + 1, mu.size))  # (m_top + 1, Ny)
    sectoral[0] = 1.0 / np.sqrt(2.0)
    for m in range(1, m_top + 1):
        sectoral[m] = -np.sqrt((2 * m + 1) / (2 * m)) * s * sectoral[m - 1]

    P = np.zeros((m_values.size, n, mu.size), dtype=np.float64)  # (Nm, L, Ny)
    for i, m in enumerate(m_values):
        if m > l_max:
            continue  # P̃ₗᵐ = 0 for every l <= l_max < m
        P[i, m] = sectoral[m]
        if m + 1 < n:
            P[i, m + 1] = np.sqrt(2 * m + 3) * mu * sectoral[m]
        for l in range(m + 2, n):
            a = np.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = np.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[i, l] = a * (mu * P[i, l - 1] - b * P[i, l - 2])
    return P


def _alp_matrix(m_abs: int, l_values: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Compute the normalised Associated Legendre Polynomial (ALP) matrix.

    P_norm[l_idx, j] = N_{l,m} * P_l^m(mu_j)

    where N_{l,m} = sqrt((2*l+1)/2 * (l-m)! / (l+m)!) is the normalisation
    constant such that:
        sum_j w_j * P_norm[l, j] * P_norm[l', j] = delta_{l, l'}

    For l < m, P_l^m = 0 (returned as zero rows).  Evaluated with the stable
    normalised recurrence of :func:`_alp_tables` (gh-96).

    Parameters
    ----------
    m_abs : int
        Absolute value of the zonal wavenumber m (>= 0).
    l_values : ndarray [Nl]
        Legendre degree indices (non-negative integers).
    mu : ndarray [Ny]
        Gauss-Legendre nodes (cos(theta)).

    Returns
    -------
    P : ndarray [Nl, Ny]
        Normalised ALP matrix.  Rows with l < m_abs are zero.

    Examples
    --------
    >>> import numpy as np
    >>> _alp_matrix(1, np.arange(3), np.array([0.0])).shape
    (3, 1)
    """
    l_values = np.asarray(l_values, dtype=int)
    l_max = int(l_values.max(initial=0))
    return _alp_tables(np.array([m_abs]), l_max, mu)[0][l_values]


# ============================================================================
# SphericalGrid1D — Gauss-Legendre latitude grid
# ============================================================================


class SphericalGrid1D(eqx.Module):
    """
    Gauss-Legendre grid for the latitude (colatitude) direction.

    The grid spans the colatitude theta in [0, pi] using N Gauss-Legendre
    quadrature nodes.  These nodes never include the poles, so sin(theta) > 0
    at all grid points.

    Mathematical Framework:
    -----------------------
    Gauss-Legendre quadrature approximates integrals as:
        integral_{-1}^{1} f(mu) d_mu ≈ sum_{j=0}^{N-1} w_j * f(mu_j)

    where mu = cos(theta), (mu_j, w_j) are the GL nodes and weights.  For N
    nodes the quadrature is exact for all polynomials of degree <= 2*N-1.

    Discrete Legendre Transform (DLT):
        Forward: c_l = sum_j w_j * P_l_norm(mu_j) * u(theta_j)
                     = P_norm_matrix @ (w * u)
        Inverse: u(theta_j) ≈ sum_l c_l * P_l_norm(mu_j)
                            = P_norm_matrix.T @ c

    where P_l_norm(mu) = sqrt((2*l+1)/2) * P_l(mu) is the normalised Legendre
    polynomial satisfying sum_j w_j * P_l_norm(mu_j) * P_l'_norm(mu_j) = delta_{l,l'}.

    Attributes
    ----------
    N : int
        Number of Gauss-Legendre quadrature points.
    L : float
        Physical domain length [rad].  Standard sphere: L = pi.
    dx : float
        Average grid spacing L/N [rad] (metric for resolution; spacing is non-uniform).
    dealias : str or None
        Dealiasing method ('2/3' rule truncates l > 2*N//3, None keeps all).
    """

    N: int
    L: float
    dx: float
    dealias: Literal["2/3", None] | None
    # Precomputed at init (numpy -> jax arrays)
    _nodes: Float[Array, "N"]
    _weights: Float[Array, "N"]
    _P_matrix: Float[Array, "N N"]  # P_l_norm[l, j]

    def __init__(
        self,
        N: int,
        L: float,
        dx: float,
        dealias: Literal["2/3", None] | None = "2/3",
    ):
        self.N = N
        self.L = L
        self.dx = dx
        self.dealias = dealias

        nodes_np, weights_np = _gauss_legendre_nodes_weights(N)
        l_values = np.arange(N)
        P_np = _legendre_matrix(l_values, nodes_np)

        self._nodes = jnp.asarray(nodes_np)
        self._weights = jnp.asarray(weights_np)
        self._P_matrix = jnp.asarray(P_np)

    # ------------------------------------------------------------------
    # Consistency / factory
    # ------------------------------------------------------------------

    def check_consistency(self, rtol: float = 1e-5) -> bool:
        """
        Verify that N, L, and dx are consistent: L ≈ N * dx.

        Parameters
        ----------
        rtol : float
            Relative tolerance.

        Returns
        -------
        bool
            True if consistent, raises ValueError otherwise.
        """
        expected_L = self.N * self.dx
        if not jnp.isclose(self.L, expected_L, rtol=rtol):
            raise ValueError(f"Grid inconsistency: L={self.L} vs N*dx={expected_L}")
        return True

    @classmethod
    def from_N_L(
        cls,
        N: int,
        L: float,
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> SphericalGrid1D:
        """Construct from number of points N and domain length L. dx = L / N."""
        return cls(N=N, L=L, dx=L / N, dealias=dealias)

    @classmethod
    def from_N_dx(
        cls,
        N: int,
        dx: float,
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> SphericalGrid1D:
        """Construct from N and spacing dx. L = N * dx."""
        return cls(N=N, L=N * dx, dx=dx, dealias=dealias)

    @classmethod
    def from_L_dx(
        cls,
        L: float,
        dx: float,
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> SphericalGrid1D:
        """Construct from domain length L and spacing dx. N = L / dx (must be integer)."""
        N_float = L / dx
        if not jnp.isclose(N_float % 1, 0) and not jnp.isclose(N_float % 1, 1):
            raise ValueError(
                f"L={L} is not divisible by dx={dx}. N={N_float} is not an integer."
            )
        return cls(N=int(round(N_float)), L=L, dx=dx, dealias=dealias)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cos_theta(self) -> Float[Array, "N"]:
        """
        Gauss-Legendre nodes mu = cos(theta) in [-1, 1], ordered North to South.

        Returns
        -------
        mu : Float[Array, "N"]
            cos(theta) values, mu[0] = near 1 (North Pole), mu[-1] = near -1 (South Pole).
        """
        return self._nodes

    @property
    def weights(self) -> Float[Array, "N"]:
        """
        Gauss-Legendre quadrature weights (sum to 2).

        These weights absorb the sin(theta) * d_theta Jacobian, so:
            integral_0^pi u(theta) sin(theta) d_theta ≈ sum_j w_j * u(theta_j)

        Returns
        -------
        w : Float[Array, "N"]
            Quadrature weights.
        """
        return self._weights

    @property
    def x(self) -> Float[Array, "N"]:
        """
        Colatitude grid points theta = arccos(mu) in (0, pi).

        Points are clustered near the poles.  The poles themselves are excluded.

        Returns
        -------
        theta : Float[Array, "N"]
            Colatitude [rad].
        """
        return jnp.arccos(self._nodes)

    @property
    def nodes_weights(self) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
        """
        Gauss-Legendre nodes and weights.

        Returns
        -------
        (mu, w) : tuple of Float[Array, "N"]
            mu = cos(theta) nodes, w = quadrature weights.
        """
        return self._nodes, self._weights

    @property
    def l(self) -> Float[Array, "N"]:
        """Spherical harmonic degree indices [0, 1, ..., N-1]."""
        return jnp.arange(self.N, dtype=float)

    @property
    def l_dealias(self) -> Float[Array, "N"]:
        """
        Dealiased degree array: keeps l <= 2*N//3, zeros out higher degrees.

        Returns
        -------
        l_d : Float[Array, "N"]
        """
        l = self.l
        if self.dealias == "2/3":
            cutoff = 2 * self.N // 3
            return jnp.where(l <= cutoff, l, 0.0)
        return l

    def dealias_filter(self) -> Float[Array, "N"]:
        """
        Dealiasing filter mask: 1 for kept modes, 0 for truncated modes.

        Returns
        -------
        mask : Float[Array, "N"]
            Binary mask in degree space.
        """
        l = self.l
        if self.dealias == "2/3":
            cutoff = 2 * self.N // 3
            return jnp.where(l <= cutoff, 1.0, 0.0)
        return jnp.ones(self.N)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self, u: Float[Array, "N"], inverse: bool = False
    ) -> Float[Array, "N"]:
        """
        Discrete Legendre Transform (DLT).

        Forward (physical -> spectral):
            c_l = sum_j w_j * P_l_norm(cos(theta_j)) * u(theta_j)
                = P_matrix @ (w * u)

        Inverse (spectral -> physical):
            u(theta_j) ≈ sum_l c_l * P_l_norm(cos(theta_j))
                       = P_matrix.T @ c

        Parameters
        ----------
        u : Float[Array, "N"]
            Physical field (if inverse=False) or spectral coefficients (if inverse=True).
        inverse : bool
            Direction of transform. Default False (physical -> spectral).

        Returns
        -------
        Array [N]
            Spectral coefficients c_l (forward) or physical values u(theta_j) (inverse).
        """
        if inverse:
            return self._P_matrix.T @ u
        return self._P_matrix @ (self._weights * u)


# ============================================================================
# SphericalGrid2D — Full sphere lat-lon grid
# ============================================================================


class SphericalGrid2D(eqx.Module):
    """
    Full sphere latitude-longitude pseudo-spectral grid.

    The longitude (phi) direction uses a uniform Fourier grid; the latitude
    (colatitude theta) direction uses Gauss-Legendre quadrature.

    Mathematical Framework:
    -----------------------
    For the unit sphere:
        phi  in [0, 2*pi): longitude, uniform, periodic — FFT in phi.
        theta in (0, pi): colatitude, Gauss-Legendre nodes.

    Full Spherical Harmonic Transform (SHT):
        Forward:
            u_hat(l, m) = (1/2) sum_j w_j sum_k P_l^m_norm(cos(theta_j))
                              * u(theta_j, phi_k) * exp(-i*m*phi_k)
        Inverse:
            u(theta_j, phi_k) = sum_{l,m} u_hat(l, m)
                                  * P_l^m_norm(cos(theta_j)) * exp(i*m*phi_k)

    In practice:
        1. FFT in phi-direction -> u_m(theta_j) for each m.
        2. For each m: Legendre transform in theta -> u_hat(l, m).

    Attributes
    ----------
    Nx : int
        Number of longitude points (uniform, Fourier).
    Ny : int
        Number of latitude / colatitude points (Gauss-Legendre).
    Lx : float
        Longitude domain [rad].  Standard: 2*pi.
    Ly : float
        Latitude / colatitude domain [rad].  Standard: pi.
    dx : float
        Longitude grid spacing = Lx / Nx.
    dy : float
        Average latitude spacing = Ly / Ny (non-uniform grid).
    dealias : str or None
        Dealiasing method.
    """

    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dx: float
    dy: float
    dealias: Literal["2/3", None] | None
    # Precomputed GL data
    _nodes_y: Float[Array, "Ny"]
    _weights_y: Float[Array, "Ny"]
    # ALP matrix per zonal wavenumber: shape (Nx, Ny, Ny)
    # _P_lm[m_fft_idx, l_idx, lat_idx] = N_{l,|m|} * P_l^{|m|}(cos(theta_j))
    _P_lm: Float[Array, "Nx Ny Ny"]

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        dx: float,
        dy: float,
        dealias: Literal["2/3", None] | None = "2/3",
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.dealias = dealias

        nodes_np, weights_np = _gauss_legendre_nodes_weights(Ny)
        self._nodes_y = jnp.asarray(nodes_np)
        self._weights_y = jnp.asarray(weights_np)

        # Zonal wavenumbers in FFT order
        m_fft = np.fft.fftfreq(Nx, 1.0 / Nx).astype(int)  # integer wavenumbers
        l_values = np.arange(Ny)

        # Build ALP matrix using |m| for each FFT wavenumber.
        # Note: both positive-m and negative-m columns share the same
        # P_l^{|m|} matrix.  This is self-consistent: the inverse SHT uses the
        # same matrix so forward(inverse(u)) = u exactly.  The stored coefficients
        # are therefore NOT standard complex-SHT coefficients for m < 0, but they
        # give correct eigenvalue computations (Laplacian, solvers) because l
        # alone determines the eigenvalue -l*(l+1)/R^2.
        m_abs = np.abs(m_fft)
        tables = _alp_tables(np.arange(m_abs.max() + 1), l_values[-1], nodes_np)
        P_lm_np = tables[m_abs]  # (Nx, Ny, Ny): one table per FFT wavenumber

        self._P_lm = jnp.asarray(P_lm_np)

    # ------------------------------------------------------------------
    # Consistency / factory
    # ------------------------------------------------------------------

    def check_consistency(self, rtol: float = 1e-5) -> bool:
        """Verify Lx ≈ Nx * dx and Ly ≈ Ny * dy."""
        errors = []
        if not jnp.isclose(self.Lx, self.Nx * self.dx, rtol=rtol):
            errors.append(f"X: Lx={self.Lx} vs Nx*dx={self.Nx * self.dx}")
        if not jnp.isclose(self.Ly, self.Ny * self.dy, rtol=rtol):
            errors.append(f"Y: Ly={self.Ly} vs Ny*dy={self.Ny * self.dy}")
        if errors:
            raise ValueError("Grid inconsistency:\n" + "\n".join(errors))
        return True

    @classmethod
    def from_N_L(
        cls,
        Nx: int,
        Ny: int,
        Lx: float = 2 * np.pi,
        Ly: float = np.pi,
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> SphericalGrid2D:
        """Construct from grid sizes and domain lengths. dx=Lx/Nx, dy=Ly/Ny."""
        return cls(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=Lx / Nx, dy=Ly / Ny, dealias=dealias)

    @classmethod
    def from_N_dx(
        cls,
        Nx: int,
        Ny: int,
        dx: float,
        dy: float,
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> SphericalGrid2D:
        """Construct from grid sizes and spacings. Lx=Nx*dx, Ly=Ny*dy."""
        return cls(Nx=Nx, Ny=Ny, Lx=Nx * dx, Ly=Ny * dy, dx=dx, dy=dy, dealias=dealias)

    @classmethod
    def from_L_dx(
        cls,
        Lx: float,
        Ly: float,
        dx: float,
        dy: float,
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> SphericalGrid2D:
        """Construct from domain lengths and spacings. Nx=Lx/dx, Ny=Ly/dy."""
        Nx_f, Ny_f = Lx / dx, Ly / dy
        errors = []
        if not jnp.isclose(Nx_f % 1, 0) and not jnp.isclose(Nx_f % 1, 1):
            errors.append(f"Lx={Lx} not divisible by dx={dx}")
        if not jnp.isclose(Ny_f % 1, 0) and not jnp.isclose(Ny_f % 1, 1):
            errors.append(f"Ly={Ly} not divisible by dy={dy}")
        if errors:
            raise ValueError("\n".join(errors))
        return cls(
            Nx=int(round(Nx_f)),
            Ny=int(round(Ny_f)),
            Lx=Lx,
            Ly=Ly,
            dx=dx,
            dy=dy,
            dealias=dealias,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> Float[Array, "Nx"]:
        """Longitude grid points phi in [0, Lx), uniform."""
        return jnp.linspace(0, self.Lx, self.Nx, endpoint=False)

    @property
    def y(self) -> Float[Array, "Ny"]:
        """Colatitude grid points theta = arccos(mu_j) in (0, pi)."""
        return jnp.arccos(self._nodes_y)

    @property
    def nodes_weights_y(
        self,
    ) -> tuple[Float[Array, "Ny"], Float[Array, "Ny"]]:
        """Gauss-Legendre nodes (cos(theta)) and weights for the latitude direction."""
        return self._nodes_y, self._weights_y

    @property
    def cos_theta(self) -> Float[Array, "Ny"]:
        """GL nodes mu = cos(theta) in [-1, 1], ordered North to South."""
        return self._nodes_y

    @property
    def weights_y(self) -> Float[Array, "Ny"]:
        """Latitudinal GL quadrature weights (sum to 2)."""
        return self._weights_y

    @property
    def legendre_table(self) -> Float[Array, "Nm Ny Ny"]:
        """
        Normalised associated Legendre functions at the Gauss nodes.

        legendre_table[m, l, j] = P̃ₗᵐ(μⱼ) for m = 0, …, Nx // 2 and
        l = 0, …, Ny − 1, with μⱼ = :attr:`cos_theta` (North to South) and
        P̃ₗᵐ = √((2l+1)/2 · (l−m)!/(l+m)!) · Pₗᵐ, Condon–Shortley phase
        included, so that Σⱼ wⱼ P̃ₗᵐ(μⱼ) P̃ₗ′ᵐ(μⱼ) = δₗₗ′.  Rows with l < m
        are zero.  These are the tables the transform uses, indexed by the
        non-negative order m rather than by FFT wavenumber.

        Returns
        -------
        table : Float[Array, "Nm Ny Ny"]
            Nm = Nx // 2 + 1 orders, Ny degrees, Ny latitudes.

        Examples
        --------
        >>> grid = SphericalGrid2D.from_N_L(Nx=16, Ny=8)
        >>> grid.legendre_table.shape
        (9, 8, 8)
        """
        # FFT index k holds |m| = k for k <= Nx // 2.
        return self._P_lm[: self.Nx // 2 + 1]

    @property
    def weights(self) -> Float[Array, "Ny Nx"]:
        """
        Full 2D integration weights: w[j, k] = w_lat[j] * dx_lon.

        Suitable for computing global integrals:
            integral u d_Omega ≈ sum_{j,k} weights[j,k] * u[j,k]
        """
        return jnp.outer(self._weights_y, jnp.full(self.Nx, self.dx))

    @property
    def X(self) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """2D meshgrid (PHI, THETA) — shapes (Ny, Nx)."""
        result = jnp.meshgrid(self.x, self.y, indexing="xy")
        return (result[0], result[1])

    @property
    def m(self) -> Float[Array, "Nx"]:
        """Zonal wavenumbers in FFT order (integer wavenumbers * 2*pi/Lx)."""
        return 2 * jnp.pi * jnp.fft.fftfreq(self.Nx, self.dx)

    @property
    def l(self) -> Float[Array, "Ny"]:
        """Spherical harmonic degree indices [0, 1, ..., Ny-1]."""
        return jnp.arange(self.Ny, dtype=float)

    @property
    def laplacian_eigenvalues(self) -> Float[Array, "Ny Nx"]:
        """
        Laplace-Beltrami eigenvalues: -l*(l+1)/R^2 broadcast to (Ny, Nx).

        nabla^2 Y_l^m = -l*(l+1)/R^2 * Y_l^m  where R = Ly / pi.
        """
        R = self.Ly / jnp.pi
        l = self.l  # (Ny,)
        eig = -l * (l + 1) / (R**2)  # (Ny,)
        return jnp.broadcast_to(eig[:, None], (self.Ny, self.Nx))

    def dealias_filter(self) -> Float[Array, "Ny Nx"]:
        """
        2D dealiasing filter: outer product of l-filter and m-filter.

        For the '2/3' rule:
            - Keeps l <= 2*Ny//3 in latitude
            - Keeps |m| <= 2*Nx//3 / 2 in longitude (Fourier 2/3 rule)

        Returns
        -------
        mask : Float[Array, "Ny Nx"]
            Binary mask (1 = kept, 0 = truncated).
        """
        if self.dealias == "2/3":
            l = self.l  # (Ny,)
            l_cutoff = 2 * self.Ny // 3
            l_mask = jnp.where(l <= l_cutoff, 1.0, 0.0)  # (Ny,)

            m_phys = 2 * jnp.pi * jnp.fft.fftfreq(self.Nx, self.dx)
            m_max = jnp.abs(m_phys).max()
            m_cutoff = m_max * 2.0 / 3.0
            m_mask = jnp.where(jnp.abs(m_phys) <= m_cutoff, 1.0, 0.0)  # (Nx,)

            return jnp.outer(l_mask, m_mask)
        return jnp.ones((self.Ny, self.Nx))

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self, u: Float[Array, "Ny Nx"], inverse: bool = False
    ) -> Complex[Array, "Ny Nx"]:
        """
        Full Spherical Harmonic Transform (SHT).

        Forward (physical -> spectral):
            Step 1: FFT in longitude phi -> u_m(theta_j) for each m.
            Step 2: Legendre transform in latitude for each m:
                u_hat(l, m) = sum_j w_j * P_l^m_norm(cos(theta_j)) * u_m(theta_j)

        Inverse (spectral -> physical):
            Step 1: Inverse Legendre transform for each m:
                u_m(theta_j) = sum_l u_hat(l, m) * P_l^m_norm(cos(theta_j))
            Step 2: IFFT in longitude.

        Parameters
        ----------
        u : Float[Array, "Ny Nx"] or Complex[Array, "Ny Nx"]
            Physical field (inverse=False) or spectral coefficients (inverse=True).
        inverse : bool
            Direction of transform.

        Returns
        -------
        Complex[Array, "Ny Nx"]
            Spectral coefficients u_hat(l, m) (forward) or
            Float[Array, "Ny Nx"] reconstructed physical field (inverse).
        """
        if inverse:
            # Step 1: Inverse Legendre for each m
            # u_hat shape: (Ny, Nx) = (Nl, Nm)
            # _P_lm shape: (Nx, Ny, Ny) = (Nm, Nl, Ntheta)
            # U_m[j, m_idx] = sum_l P_lm[m_idx, l, j] * u_hat[l, m_idx]
            #                = _P_lm[m_idx, :, j] @ u_hat[:, m_idx]
            # = einsum('mlj,lm->jm', _P_lm, u_hat)
            U_m = jnp.einsum("mlj,lm->jm", self._P_lm, u)  # (Ny, Nx)
            return jnp.fft.ifft(U_m, axis=-1).real
        else:
            # Step 1: FFT in longitude
            U_m = jnp.fft.fft(u, axis=-1)  # (Ny, Nx)
            # Step 2: Legendre transform for each m
            # u_hat[l, m_idx] = sum_j w_j * P_lm[m_idx, l, j] * U_m[j, m_idx]
            # = einsum('mlj,jm->lm', _P_lm, w*U_m)
            wU = self._weights_y[:, None] * U_m  # (Ny, Nx)
            return jnp.einsum("mlj,jm->lm", self._P_lm, wU)  # (Ny, Nx)
