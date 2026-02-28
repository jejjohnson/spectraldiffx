"""
Chebyshev Pseudo-Spectral Discretization Module
================================================

Spectral methods using Chebyshev polynomial basis for PDEs on non-periodic domains.
Provides spectral accuracy on [-L, L] with Gauss-Lobatto or Gauss quadrature nodes.

Key Concepts:
-------------
    • Chebyshev expansion: u(x) = Σ aₖ Tₖ(x)  where Tₖ(x) = cos(k arccos(x))
    • Differentiation matrix D: (Du)ᵢ = du/dx evaluated at node xᵢ
    • Gauss-Lobatto nodes include endpoints ±L (suitable for Dirichlet BCs)
    • Gauss nodes exclude endpoints (suitable for spectral integration)

Advantages:
-----------
    • Exponential (spectral) accuracy for smooth functions
    • Handles non-periodic boundary conditions (Dirichlet, Neumann)
    • Node clustering near boundaries reduces Runge phenomenon

References:
-----------
[1] Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.
[2] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods. Dover.
[3] Canuto et al. (2006). Spectral Methods: Fundamentals in Single Domains.
"""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

# ============================================================================
# Internal numpy helpers (called once at __init__ time, not inside JIT)
# ============================================================================


def _cheb_diff_matrix_gl(N: int) -> np.ndarray:
    """
    Chebyshev differentiation matrix for Gauss-Lobatto nodes on [-1, 1].

    Uses the standard formula from Trefethen (2000), Ch. 6:

        D_{ij} = (cᵢ/cⱼ) * (-1)^{i+j} / (xᵢ - xⱼ)   i ≠ j
        D_{jj} = -Σ_{k≠j} D_{jk}                         (row-sum = 0)

    where cᵢ = 2 for i = 0 or N, else cᵢ = 1.

    Parameters:
    -----------
    N : int
        Polynomial degree. Matrix size is (N+1) × (N+1).

    Returns:
    --------
    D : ndarray [N+1, N+1]
        Differentiation matrix on [-1, 1].
    """
    if N == 0:
        return np.zeros((1, 1))

    x = np.cos(np.pi * np.arange(N + 1) / N)  # GL nodes: x[0]=1, x[N]=-1

    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0

    ii = np.arange(N + 1)
    X = np.tile(x, (N + 1, 1))  # X[i,j] = x[j]
    dX = X.T - X  # dX[i,j] = x[i] - x[j]

    sign = (-1.0) ** (ii[:, None] + ii[None, :])  # [N+1, N+1]

    with np.errstate(divide="ignore", invalid="ignore"):
        D = (c[:, None] / c[None, :]) * sign / dX

    np.fill_diagonal(D, 0.0)
    # Diagonal: negative row sum (ensures D * constant = 0)
    D -= np.diag(D.sum(axis=1))
    return D


def _cheb_diff_matrix_gauss(N: int) -> np.ndarray:
    """
    Chebyshev differentiation matrix for Gauss nodes on [-1, 1].

    Uses the barycentric interpolation formula (Berrut & Trefethen 2004):

        D_{ij} = (wⱼ/wᵢ) / (xᵢ - xⱼ)   i ≠ j
        D_{ii} = -Σ_{j≠i} D_{ij}          (row-sum = 0)

    where wⱼ = (-1)ʲ sin(π(2j+1)/(2N)) are the barycentric weights.

    Parameters:
    -----------
    N : int
        Number of Gauss nodes. Matrix size is N × N.

    Returns:
    --------
    D : ndarray [N, N]
        Differentiation matrix on [-1, 1].
    """
    if N == 0:
        return np.zeros((1, 1))

    j = np.arange(N)
    theta = np.pi * (2 * j + 1) / (2 * N)
    x = np.cos(theta)  # Gauss nodes: x[0] near 1, x[N-1] near -1
    w = (-1.0) ** j * np.sin(theta)  # barycentric weights

    X = np.tile(x, (N, 1))  # X[i,j] = x[j]
    W = np.tile(w, (N, 1))  # W[i,j] = w[j]

    with np.errstate(divide="ignore", invalid="ignore"):
        D = (W / W.T) / (X.T - X)  # D[i,j] = (w[j]/w[i]) / (x[i]-x[j])

    np.fill_diagonal(D, 0.0)
    D -= np.diag(D.sum(axis=1))
    return D


# ============================================================================
# ChebyshevGrid1D
# ============================================================================


class ChebyshevGrid1D(eqx.Module):
    """
    1D Chebyshev grid on the domain [-L, L].

    Mathematical Framework:
    -----------------------
    Maps the reference domain [-1, 1] to the physical domain [-L, L] via x = L·ξ.

    Gauss-Lobatto nodes (includes endpoints):
        ξⱼ = cos(πj/N),  j = 0, ..., N     (N+1 points; ξ₀=1, ξₙ=-1)
        xⱼ = L·ξⱼ

    Gauss nodes (excludes endpoints):
        ξⱼ = cos(π(2j+1)/(2N)),  j = 0, ..., N-1    (N points)
        xⱼ = L·ξⱼ

    Differentiation (chain rule):
        d/dx = (1/L) · d/dξ

    So the physical differentiation matrix is:
        D_phys = D_ref / L

    Attributes:
    -----------
        N : int
            Polynomial degree (Gauss-Lobatto: N+1 points; Gauss: N points).
        L : float
            Physical domain half-length. Domain is [-L, L].
        node_type : str
            Node distribution: 'gauss-lobatto' (default) or 'gauss'.
        dealias : str or None
            Dealiasing strategy: '2/3' truncates top 1/3 of modes; None keeps all.
    """

    N: int
    L: float
    node_type: str
    dealias: Literal["2/3", None] | None
    _D: Array  # differentiation matrix on [-L, L], shape (n_pts, n_pts)

    def __init__(
        self,
        N: int,
        L: float = 1.0,
        node_type: Literal["gauss-lobatto", "gauss"] = "gauss-lobatto",
        dealias: Literal["2/3", None] | None = "2/3",
    ):
        """
        Parameters:
        -----------
        N : int
            Polynomial degree (≥ 1).
        L : float
            Domain half-length (domain is [-L, L]). Default 1.0.
        node_type : str
            'gauss-lobatto' (default) or 'gauss'.
        dealias : str or None
            '2/3' to zero the top third of modes; None for no dealiasing.
        """
        self.N = N
        self.L = L
        self.node_type = node_type
        self.dealias = dealias

        if node_type == "gauss-lobatto":
            D_np = _cheb_diff_matrix_gl(N) / L
        else:
            D_np = _cheb_diff_matrix_gauss(N) / L
        self._D = jnp.array(D_np)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_N_L(
        cls,
        N: int,
        L: float,
        node_type: Literal["gauss-lobatto", "gauss"] = "gauss-lobatto",
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> "ChebyshevGrid1D":
        """
        Initialize from polynomial degree N and domain half-length L.

        Parameters:
        -----------
        N : int
            Polynomial degree.
        L : float
            Domain half-length (domain is [-L, L]).

        Example:
        --------
        >>> grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0)
        """
        return cls(N=N, L=L, node_type=node_type, dealias=dealias)

    @classmethod
    def from_N_dx(
        cls,
        N: int,
        dx: float,
        node_type: Literal["gauss-lobatto", "gauss"] = "gauss-lobatto",
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> "ChebyshevGrid1D":
        """
        Initialize from polynomial degree N and average grid spacing dx.

        Computes: L = N * dx / 2  (since dx ≈ 2L/N for the full domain [-L, L])

        Parameters:
        -----------
        N : int
            Polynomial degree.
        dx : float
            Average grid spacing (dx = 2L/N).

        Example:
        --------
        >>> grid = ChebyshevGrid1D.from_N_dx(N=16, dx=0.125)  # L = 1.0
        """
        L = N * dx / 2.0
        return cls(N=N, L=L, node_type=node_type, dealias=dealias)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> Float[Array, "N1"]:
        """
        Physical grid nodes on [-L, L].

        Gauss-Lobatto: xⱼ = L·cos(πj/N), j = 0,...,N  (N+1 points, decreasing)
        Gauss:         xⱼ = L·cos(π(2j+1)/(2N)), j=0,...,N-1 (N points, decreasing)

        Returns:
        --------
        x : Array [N+1] for GL, [N] for Gauss
        """
        if self.node_type == "gauss-lobatto":
            return self.L * jnp.cos(jnp.pi * jnp.arange(self.N + 1) / self.N)
        else:
            j = jnp.arange(self.N)
            return self.L * jnp.cos(jnp.pi * (2 * j + 1) / (2 * self.N))

    @property
    def D(self) -> Float[Array, "N1 N1"]:
        """
        Chebyshev differentiation matrix on [-L, L].

        Satisfies: (D @ u)ᵢ ≈ du/dx at xᵢ (exact for polynomials of degree ≤ N).

        Rows sum to zero: D @ ones = 0 (derivative of constant = 0).

        Returns:
        --------
        D : Array [N+1, N+1] for GL, [N, N] for Gauss
        """
        return self._D

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, u: Array, inverse: bool = False) -> Array:
        """
        Chebyshev spectral transform (forward or inverse).

        Forward (physical → spectral):
            Computes Chebyshev expansion coefficients aₖ from nodal values uⱼ.
            Uses DCT-I (FFT-based) for Gauss-Lobatto nodes.
            Uses DCT-II (FFT-based) for Gauss nodes.

        Inverse (spectral → physical):
            Recovers nodal values uⱼ from coefficients aₖ.

        Convention:
            a[k] = FFT-extended(u)[k] / N   (Gauss-Lobatto)
            a[0] = (1/N) Σⱼ uⱼ,  a[k] = (2/N) Σⱼ uⱼ Tₖ(xⱼ)  k > 0  (Gauss)

        Parameters:
        -----------
        u : Array
            Physical-space values (forward) or spectral coefficients (inverse).
        inverse : bool
            If True, compute physical values from spectral coefficients.

        Returns:
        --------
        Array
            Spectral coefficients (forward) or physical values (inverse).
        """
        N = self.N
        if self.node_type == "gauss-lobatto":
            return self._transform_gl(u, N, inverse)
        else:
            return self._transform_gauss(u, N, inverse)

    @staticmethod
    def _transform_gl(u: Array, N: int, inverse: bool) -> Array:
        """FFT-based DCT-I Chebyshev transform for Gauss-Lobatto nodes."""
        if not inverse:
            # Forward: extend symmetrically to length 2N, then rfft
            # y = [u_0, u_1, ..., u_N, u_{N-1}, ..., u_1]  (length 2N)
            y = jnp.concatenate([u, u[-2:0:-1]])
            c = jnp.fft.rfft(y)
            # Normalize: a[k] = Re(C[k]) / N
            return c.real / N
        else:
            # Inverse: irfft(N * a)[:N+1]
            # a has length N+1; irfft produces length 2N
            y = jnp.fft.irfft(N * u + 0j, n=2 * N)
            return y[: N + 1]

    @staticmethod
    def _transform_gauss(u: Array, N: int, inverse: bool) -> Array:
        """FFT-based DCT-II / synthesis transform for Gauss nodes.

        Forward convention:
            a[0]  = (1/N) Σⱼ uⱼ              (k=0: halved)
            a[k]  = (2/N) Σⱼ uⱼ cos(πk(2j+1)/(2N))  (k > 0)

        Inverse:
            u[n] = Σₖ a[k] Tₖ(xₙ)   (direct synthesis, no adjustment to a[0])

        Dtypes are derived from the input to preserve precision in x64 mode.
        """
        # Derive real and complex dtypes from the input to preserve precision.
        real_dtype = jnp.result_type(u, jnp.float32)
        complex_dtype = jnp.result_type(real_dtype, jnp.complex64)

        if not inverse:
            # Forward DCT-II: a[k] = (2/N) Σⱼ uⱼ cos(πk(2j+1)/(2N))
            # Via FFT: zero-pad, apply half-sample phase shift
            u_pad = jnp.concatenate([u, jnp.zeros(N, dtype=real_dtype)])  # length 2N
            Y = jnp.fft.rfft(u_pad)  # complex, length N+1
            k_idx = jnp.arange(N + 1, dtype=real_dtype)
            phase = jnp.exp((-1j * jnp.pi * k_idx / (2 * N)).astype(complex_dtype))
            Z = Y * phase  # Z[k] = Σⱼ uⱼ exp(-iπ(2j+1)k/(2N))
            a = Z[:N].real * (2.0 / N)
            # Halve k=0 so that a[0] = (1/N) Σ uⱼ
            return a.at[0].multiply(0.5)
        else:
            # Inverse (synthesis): u[n] = Σₖ a[k] Tₖ(xₙ)
            # = Re[Σₖ a[k] exp(iπk(2n+1)/(2N))]
            # = Re[2N · IFFT_{2N}([h₀,...,h_{N-1},0,...,0])[n]]
            # where h[k] = a[k] · exp(iπk/(2N))
            # NOTE: a[0] is NOT doubled here; the halving in the forward
            # is already accounted for since the synthesis uses a[k] directly.
            k_idx = jnp.arange(N, dtype=real_dtype)
            phase = jnp.exp((1j * jnp.pi * k_idx / (2 * N)).astype(complex_dtype))
            h = u.astype(complex_dtype) * phase  # complex, length N
            H_full = jnp.concatenate([h, jnp.zeros(N, dtype=complex_dtype)])
            y = jnp.fft.ifft(H_full)
            return (2 * N * y.real)[:N]

    # ------------------------------------------------------------------
    # Dealiasing
    # ------------------------------------------------------------------

    def dealias_filter(self) -> Float[Array, "N1"]:
        """
        Dealiasing mask in Chebyshev mode space.

        2/3 rule: keep modes 0,...,floor(2N/3), zero out higher modes.
        This prevents aliasing in quadratic nonlinearities.

        Returns:
        --------
        mask : Array [N+1] for GL, [N] for Gauss
            1.0 for kept modes, 0.0 for removed modes.
        """
        n_modes = self.N + 1 if self.node_type == "gauss-lobatto" else self.N
        if self.dealias == "2/3":
            cutoff = int(2 * self.N / 3)
            mask = jnp.arange(n_modes) <= cutoff
            return mask.astype(jnp.float32)
        else:
            return jnp.ones(n_modes)

    # ------------------------------------------------------------------
    # Consistency check
    # ------------------------------------------------------------------

    def check_consistency(self, rtol: float = 1e-5) -> bool:
        """
        Verify that N ≥ 1 and L > 0.

        Parameters:
        -----------
        rtol : float
            Unused (kept for API consistency with FourierGrid).

        Returns:
        --------
        bool
            True if consistent, raises ValueError otherwise.
        """
        if self.N < 1:
            raise ValueError(f"N must be ≥ 1, got N={self.N}")
        if self.L <= 0:
            raise ValueError(f"L must be > 0, got L={self.L}")
        return True


# ============================================================================
# ChebyshevGrid2D
# ============================================================================


class ChebyshevGrid2D(eqx.Module):
    """
    2D Chebyshev grid on [-Lx, Lx] × [-Ly, Ly].

    Tensor product of two 1D Chebyshev grids. Physical arrays have shape
    (Ny_pts, Nx_pts) where Ny_pts = Ny+1 (GL) or Ny (Gauss), similarly for x.

    Mathematical Framework:
    -----------------------
    For u(x, y) on [-Lx, Lx] × [-Ly, Ly]:

        Partial derivatives via differentiation matrices:
            ∂u/∂x [j,i] = (u @ Dxᵀ)[j,i]     (Dx applied along axis 1)
            ∂u/∂y [j,i] = (Dy @ u)[j,i]        (Dy applied along axis 0)

    Attributes:
    -----------
        Nx, Ny : int
            Polynomial degrees in x and y directions.
        Lx, Ly : float
            Physical domain half-lengths.
        node_type : str
            Node type for both directions.
        dealias : str or None
            Dealiasing strategy.
    """

    Nx: int
    Ny: int
    Lx: float
    Ly: float
    node_type: str
    dealias: Literal["2/3", None] | None
    _Dx: Array  # x-direction differentiation matrix
    _Dy: Array  # y-direction differentiation matrix
    _Dx2: Array  # x-direction second-derivative matrix (Dx @ Dx)
    _Dy2: Array  # y-direction second-derivative matrix (Dy @ Dy)

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float = 1.0,
        Ly: float = 1.0,
        node_type: Literal["gauss-lobatto", "gauss"] = "gauss-lobatto",
        dealias: Literal["2/3", None] | None = "2/3",
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.node_type = node_type
        self.dealias = dealias

        if node_type == "gauss-lobatto":
            Dx_np = _cheb_diff_matrix_gl(Nx) / Lx
            Dy_np = _cheb_diff_matrix_gl(Ny) / Ly
        else:
            Dx_np = _cheb_diff_matrix_gauss(Nx) / Lx
            Dy_np = _cheb_diff_matrix_gauss(Ny) / Ly

        self._Dx = jnp.array(Dx_np)
        self._Dy = jnp.array(Dy_np)
        self._Dx2 = jnp.array(Dx_np @ Dx_np)
        self._Dy2 = jnp.array(Dy_np @ Dy_np)

    @classmethod
    def from_N_L(
        cls,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        node_type: Literal["gauss-lobatto", "gauss"] = "gauss-lobatto",
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> "ChebyshevGrid2D":
        """Initialize from polynomial degrees and domain half-lengths."""
        return cls(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, node_type=node_type, dealias=dealias)

    @classmethod
    def from_N_dx(
        cls,
        Nx: int,
        Ny: int,
        dx: float,
        dy: float,
        node_type: Literal["gauss-lobatto", "gauss"] = "gauss-lobatto",
        dealias: Literal["2/3", None] | None = "2/3",
    ) -> "ChebyshevGrid2D":
        """Initialize from polynomial degrees and average grid spacings."""
        Lx = Nx * dx / 2.0
        Ly = Ny * dy / 2.0
        return cls(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, node_type=node_type, dealias=dealias)

    @property
    def x(self) -> Float[Array, "Nx1"]:
        """Physical x-nodes on [-Lx, Lx]."""
        if self.node_type == "gauss-lobatto":
            return self.Lx * jnp.cos(jnp.pi * jnp.arange(self.Nx + 1) / self.Nx)
        else:
            j = jnp.arange(self.Nx)
            return self.Lx * jnp.cos(jnp.pi * (2 * j + 1) / (2 * self.Nx))

    @property
    def y(self) -> Float[Array, "Ny1"]:
        """Physical y-nodes on [-Ly, Ly]."""
        if self.node_type == "gauss-lobatto":
            return self.Ly * jnp.cos(jnp.pi * jnp.arange(self.Ny + 1) / self.Ny)
        else:
            j = jnp.arange(self.Ny)
            return self.Ly * jnp.cos(jnp.pi * (2 * j + 1) / (2 * self.Ny))

    @property
    def X(self) -> tuple[Float[Array, "Ny1 Nx1"], Float[Array, "Ny1 Nx1"]]:
        """
        2D meshgrid (X, Y) with shapes (Ny_pts, Nx_pts).

        Indexing: X[j, i] = x[i], Y[j, i] = y[j].
        """
        result = jnp.meshgrid(self.x, self.y, indexing="xy")
        return result[0], result[1]

    @property
    def Dx(self) -> Float[Array, "Nx1 Nx1"]:
        """x-direction differentiation matrix on [-Lx, Lx]."""
        return self._Dx

    @property
    def Dy(self) -> Float[Array, "Ny1 Ny1"]:
        """y-direction differentiation matrix on [-Ly, Ly]."""
        return self._Dy

    @property
    def Dx2(self) -> Float[Array, "Nx1 Nx1"]:
        """Precomputed x-direction second-derivative matrix: Dx @ Dx."""
        return self._Dx2

    @property
    def Dy2(self) -> Float[Array, "Ny1 Ny1"]:
        """Precomputed y-direction second-derivative matrix: Dy @ Dy."""
        return self._Dy2

    def dealias_filter(
        self,
    ) -> tuple[Float[Array, "Nx1"], Float[Array, "Ny1"]]:
        """
        1D dealiasing masks for x and y mode spaces.

        Returns:
        --------
        (mask_x, mask_y) : tuple of Arrays
            Masks of shape (Nx_pts,) and (Ny_pts,).
        """
        nx_modes = self.Nx + 1 if self.node_type == "gauss-lobatto" else self.Nx
        ny_modes = self.Ny + 1 if self.node_type == "gauss-lobatto" else self.Ny
        if self.dealias == "2/3":
            cutoff_x = int(2 * self.Nx / 3)
            cutoff_y = int(2 * self.Ny / 3)
            mask_x = (jnp.arange(nx_modes) <= cutoff_x).astype(jnp.float32)
            mask_y = (jnp.arange(ny_modes) <= cutoff_y).astype(jnp.float32)
        else:
            mask_x = jnp.ones(nx_modes)
            mask_y = jnp.ones(ny_modes)
        return mask_x, mask_y

    def transform(self, u: Array, inverse: bool = False) -> Array:
        """
        2D Chebyshev transform (tensor product of 1D transforms).

        Forward: applies 1D transform along x-axis (axis 1), then y-axis (axis 0).
        Inverse: applies 1D inverse along y-axis (axis 0), then x-axis (axis 1).

        Parameters:
        -----------
        u : Array [Ny_pts, Nx_pts]
            Physical-space field (forward) or spectral coefficients (inverse).
        inverse : bool
            If True, inverse transform.

        Returns:
        --------
        Array [Ny_pts, Nx_pts]
        """
        Nx, Ny = self.Nx, self.Ny
        is_gl = self.node_type == "gauss-lobatto"
        _gl = ChebyshevGrid1D._transform_gl
        _gauss = ChebyshevGrid1D._transform_gauss

        def tx(row):
            return _gl(row, Nx, inverse) if is_gl else _gauss(row, Nx, inverse)

        def ty(col):
            return _gl(col, Ny, inverse) if is_gl else _gauss(col, Ny, inverse)

        if not inverse:
            # Transform along x (rows), then y (cols)
            u_x = jax.vmap(tx, in_axes=0, out_axes=0)(u)
            return jax.vmap(ty, in_axes=1, out_axes=1)(u_x)
        else:
            # Untransform along y (cols), then x (rows)
            u_y = jax.vmap(ty, in_axes=1, out_axes=1)(u)
            return jax.vmap(tx, in_axes=0, out_axes=0)(u_y)

    def check_consistency(self, rtol: float = 1e-5) -> bool:
        """Verify that Nx, Ny ≥ 1 and Lx, Ly > 0."""
        errors = []
        if self.Nx < 1:
            errors.append(f"Nx must be ≥ 1, got Nx={self.Nx}")
        if self.Ny < 1:
            errors.append(f"Ny must be ≥ 1, got Ny={self.Ny}")
        if self.Lx <= 0:
            errors.append(f"Lx must be > 0, got Lx={self.Lx}")
        if self.Ly <= 0:
            errors.append(f"Ly must be > 0, got Ly={self.Ly}")
        if errors:
            raise ValueError("\n".join(errors))
        return True
