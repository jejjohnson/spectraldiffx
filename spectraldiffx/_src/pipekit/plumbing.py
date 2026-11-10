"""Graph plumbing: linear combinations and tuple selection.

Both work on either carrier: ``Spectral`` implements the linear algebra, so
a sum of spectral terms costs no transform.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_jax import JaxOperator


class LinComb(JaxOperator):
    """y = ∑ᵢ cᵢ xᵢ on physical fields or `Spectral` (coefficients are leaves).

    Attributes
    ----------
    coeffs : Array
        ``[n]`` coefficients, one per input.
    """

    coeffs: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, *xs: Any) -> Any:
        if len(xs) != self.coeffs.shape[0]:
            raise ValueError(
                f"LinComb has {self.coeffs.shape[0]} coefficients, got {len(xs)} inputs"
            )
        terms = [c * x for c, x in zip(self.coeffs, xs, strict=True)]
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total


class Pick(JaxOperator):
    """Select one output of a tuple-valued node: ``(a, b, …)[index]``.

    Attributes
    ----------
    index : int
        Position to select.
    """

    index: int = eqx.field(static=True)

    def _apply(self, xs: tuple[Any, ...]) -> Any:
        return xs[self.index]
