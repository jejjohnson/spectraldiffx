"""The ``Spectral`` carrier: Fourier coefficients on a pipeline edge.

A pipekit edge carries either a real physical field or ``Spectral(hat)``,
the 2-D FFT coefficients of one. Operators skip the forward transform for
``Spectral`` input and emit coefficients when ``out="spectral"``, so the
transforms in a pipeline sit only where a physical-space product needs
them::

    q : real [..., Ny, Nx] ──ToSpectral──► Spectral(q̂) ──Laplacian(out="spectral")──► Spectral(−|k|² q̂)
                                1 fft                       0 transforms

XLA merges identical transforms of the same array but does not cancel an
inverse → forward pair, which is why the carrier exists.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex

from spectraldiffx._src.fourier.grid import FourierGrid2D


class Spectral(eqx.Module):
    """2-D FFT coefficients ``û = fft2(u)`` of a real field ``u``.

    A pytree (one array leaf), so ``jax.tree.map``, ``vmap`` and
    ``lax.scan`` carries work unchanged. Linear algebra stays on the
    carrier, so sums of terms and Runge–Kutta stages need no transform:
    ``a + b``, ``a - b``, ``c * a`` (``c`` a scalar), ``-a``.

    Attributes
    ----------
    hat : Complex[Array, "... Ny Nx"]
        Coefficients in FFT order on the last two axes (unnormalised
        forward transform, as ``FourierGrid2D.transform``).
    """

    hat: Complex[Array, "... Ny Nx"]

    def __add__(self, other: Spectral) -> Spectral:
        return Spectral(self.hat + _hat_of(other))

    def __radd__(self, other: Any) -> Spectral:
        if isinstance(other, int | float) and other == 0:  # sum() starts at 0
            return self
        return Spectral(_hat_of(other) + self.hat)

    def __sub__(self, other: Spectral) -> Spectral:
        return Spectral(self.hat - _hat_of(other))

    def __mul__(self, coeff: Any) -> Spectral:
        return Spectral(coeff * self.hat)

    __rmul__ = __mul__

    def __truediv__(self, coeff: Any) -> Spectral:
        return Spectral(self.hat / coeff)

    def __neg__(self) -> Spectral:
        return Spectral(-self.hat)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the coefficient array (``[..., Ny, Nx]``)."""
        return self.hat.shape


def _hat_of(x: Any) -> Array:
    if not isinstance(x, Spectral):
        raise TypeError(
            "Spectral only combines with Spectral (got "
            f"{type(x).__name__}); transform it first with ToSpectral"
        )
    return x.hat


def as_hat(x: Array | Spectral, grid: FourierGrid2D) -> Array:
    """Coefficients of ``x``: transformed if physical, as is if ``Spectral``."""
    if isinstance(x, Spectral):
        return x.hat
    if jnp.iscomplexobj(x):
        raise TypeError(
            "physical fields must be real; wrap Fourier coefficients in Spectral"
        )
    return grid.transform(x)


def to_physical(hat: Array, grid: FourierGrid2D) -> Array:
    """The real field with coefficients ``hat`` (one inverse transform)."""
    return grid.transform(hat, inverse=True).real
