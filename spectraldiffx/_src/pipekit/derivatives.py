"""Fourier derivative operators on the physical / ``Spectral`` carrier.

Each operator is one ``SpectralDerivative2D`` call: inputs are turned into
coefficients (``Spectral`` as is, a physical field with one FFT), the
Tier-A method runs with ``spectral=True`` and ``spectral_out`` set from
``out``, and the result is wrapped back. No Fourier multiplier lives here.

Transforms per call, ``Spectral`` in and ``out="spectral"``:

    linear operators (Gradient … Dealias)        0
    Jacobian                                     4 inverse + 1 forward
    AdvectionScalar (u, v physical, q Spectral)  2 inverse + 1 forward (+ u, v dealias)

A physical input adds one forward transform; ``out="physical"`` adds one
inverse transform per output.
"""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from spectraldiffx._src.fourier.operators import SpectralDerivative2D
from spectraldiffx._src.pipekit.base import SpectralOperator
from spectraldiffx._src.pipekit.carrier import Spectral, as_hat, to_physical

Field = Array | Spectral


class _FourierOp(SpectralOperator):
    """A Tier-B op that calls one ``SpectralDerivative2D`` method.

    Subclasses name the method (`method`) and may add keyword arguments
    (`_kwargs`).
    """

    method: ClassVar[str]

    def _kwargs(self) -> dict[str, Any]:
        return {}

    def _apply(self, *fields: Field) -> Any:
        grid = self.fourier_grid
        hats = [as_hat(x, grid) for x in fields]
        deriv = SpectralDerivative2D(grid)
        result = getattr(deriv, self.method)(
            *hats, spectral=True, spectral_out=self.spectral_out, **self._kwargs()
        )
        if isinstance(result, tuple):
            return tuple(self.emit(r) for r in result)
        return self.emit(result)


class Gradient(_FourierOp):
    """(∂ₓu, ∂ᵧu): (i kₓ û, i k_y û).  1 → 2."""

    method: ClassVar[str] = "gradient"


class Divergence(_FourierOp):
    """∂ₓvₓ + ∂ᵧv_y: i kₓ v̂ₓ + i k_y v̂_y.  2 → 1."""

    method: ClassVar[str] = "divergence"


class Curl(_FourierOp):
    """ζ = ∂ₓv_y − ∂ᵧvₓ: i kₓ v̂_y − i k_y v̂ₓ.  2 → 1."""

    method: ClassVar[str] = "curl"


class Laplacian(_FourierOp):
    """∇²u: −|k|² û.  1 → 1."""

    method: ClassVar[str] = "laplacian"


class InverseLaplacian(_FourierOp):
    """∇⁻²u: −û / |k|², zero mean.  1 → 1."""

    method: ClassVar[str] = "inverse_laplacian"


class Biharmonic(_FourierOp):
    """∇⁴u: |k|⁴ û.  1 → 1."""

    method: ClassVar[str] = "biharmonic"


class VelocityFromStreamfunction(_FourierOp):
    """(u, v) = (−∂ᵧψ, ∂ₓψ): (−i k_y ψ̂, i kₓ ψ̂).  1 → 2."""

    method: ClassVar[str] = "velocity_from_streamfunction"


class Jacobian(_FourierOp):
    """J(f, g) = ∂ₓf ∂ᵧg − ∂ᵧf ∂ₓg, 2/3-dealiased.  2 → 1.

    Spectral in and out: 4 inverse + 1 forward transforms.
    """

    method: ClassVar[str] = "jacobian"


class Dealias(_FourierOp):
    """2/3-rule truncation m ⊙ û.  1 → 1."""

    method: ClassVar[str] = "apply_dealias"


class Hyperviscosity(_FourierOp):
    """−ν |k|²ⁿ û (always dissipative).  1 → 1.

    Attributes
    ----------
    grid : GridSpec
        ``(Nx, Ny, Lx, Ly, dealias)``.
    nu : Array
        Coefficient ν ≥ 0 (a learnable leaf; a negative traced value raises
        at run time).
    order : int
        n (1: diffusion, 2: biharmonic).
    out : {"physical", "spectral"}
        Output space.
    """

    method: ClassVar[str] = "hyperviscosity"

    nu: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    order: int = eqx.field(static=True, default=2, kw_only=True)

    def _kwargs(self) -> dict[str, Any]:
        return {"nu": self.nu, "order": self.order}


class AdvectionScalar(SpectralOperator):
    """(u·∇)q, 2/3-dealiased.  (u, v, q) → 1.

    The velocities enter the products in physical space (a ``Spectral``
    velocity costs one inverse transform); ``q`` may be ``Spectral``.
    """

    def _apply(self, vx: Field, vy: Field, q: Field) -> Field:
        grid = self.fourier_grid
        vx_p, vy_p = (
            to_physical(v.hat, grid) if isinstance(v, Spectral) else v for v in (vx, vy)
        )
        result = SpectralDerivative2D(grid).advection_scalar(
            vx_p,
            vy_p,
            as_hat(q, grid),
            spectral=True,
            spectral_out=self.spectral_out,
        )
        return self.emit(result)
