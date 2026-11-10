"""Explicit transform nodes: ``ToSpectral`` and ``ToPhysical``.

Making the transforms graph nodes shows where they happen in
``Graph.describe()``; every other operator only transforms what its carrier
requires.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from spectraldiffx._src.pipekit.base import SpectralOperator
from spectraldiffx._src.pipekit.carrier import Spectral, as_hat, to_physical


class ToSpectral(SpectralOperator):
    """Real field → `Spectral` (1 forward FFT; 0 if already `Spectral`)."""

    def _apply(self, x: Float[Array, "... Ny Nx"] | Spectral) -> Spectral:
        return Spectral(as_hat(x, self.fourier_grid))


class ToPhysical(SpectralOperator):
    """`Spectral` → real field (1 inverse FFT; 0 if already physical)."""

    def _apply(self, x: Spectral | Float[Array, "... Ny Nx"]) -> Array:
        if isinstance(x, Spectral):
            return to_physical(x.hat, self.fourier_grid)
        return x
