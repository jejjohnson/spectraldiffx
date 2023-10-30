from typing import NamedTuple, Iterable, Optional
from jaxtyping import Array
import jax.numpy as jnp
from functools import reduce
from spectraldiffx._src.difference import difference, spectral_difference
from spectraldiffx._src.utils import (
    calculate_fft_freq,
    calculate_aliasing,
    fft_transform,
)
import math


def calculate_operators(k_vec: Iterable[Array], order: int=1):
    """Calculates operators for each of the 1D fourier frequencies
    Operator(n) = |k|^2n

    Args:
        k_vec (Iterable[Array]):"""

    # expand each of the dimensions
    fn = lambda x: x ** (2 * float(order))
    return [jnp.expand_dims(fn(array), axis=i) for i, array in enumerate(k_vec[::-1])]


def elliptical_operator(
    k_vec: Iterable[Array],
    order: int = 1,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Array:
    """Generates the elliptical operator
    Equation:
        Operator(n) = |k|^2n

    Args:
        k_vec (Iterable[Array]): the vector of frequencies
        order (int, optional): the order of the operator.
            Defaults to 1.
        alpha (float, optional): any scaler in front of the laplacian.
            Defaults to 1.0.
        beta (float, optional): any scalar/array after the Laplacian.
            This corresponds to the helmholtz operator. Defaults to 0.0.

    Returns:
        Array: _description_
    """

    ks = calculate_operators(k_vec, order=order)

    # sum each of dimensions
    ksq = reduce(lambda x, y: x + y, ks)

    # reshape and add beta
    ksq = alpha * ksq + beta

    return ksq


# def elliptical_inversion(u: Array, k_vec: Iterable[Array]) -> Array:
#     msg = "Error: the spectral field should be 2D"
#     assert len(u.k_vec) == u.domain.ndim == 2, msg

#     # calculate scalar quantity
#     ksq = elliptical_operator(k_vec)

#     uh_values = jnp.fft.fftn(u, axes=(-2, -1)) / sum(u.domain.Nx)

#     # do inversion
#     invksq = 1.0 / ksq
#     invksq = invksq.at[0, 0].set(1.0)

#     uh_values = -invksq * uh_values

#     u_values = jnp.real(sum(u.domain.Nx) * jnp.fft.ifftn(uh_values, axes=(-2, -1)))

#     return SpectralField(values=u_values, domain=u.domain)
