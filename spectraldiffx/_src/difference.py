import typing as tp
import functools as ft
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
from spectraldiffx._src.utils import calculate_fft_freq, fft_transform
import math


def spectral_difference(
    fu: Array, k_vec: Array, axis: int = 0, derivative: int = 1
) -> Array:
    """the difference method in spectral space

    Args:
        fu (Array): the array in spectral space to take the difference
        k_vec (Array): the parameters to take the finite difference
        axis (int, optional): the axis of the multidim array, u, to
            take the finite difference. Defaults to 0.
        derivative (int, optional): the number of derivatives to take.
            Defaults to 1.

    Returns:
        dfu (Array): the finite difference method
    """

    # reshape axis
    fu = jnp.moveaxis(fu, axis, -1)
    
    # calculate factor
    factor = (1j * k_vec) ** float(derivative)

    # do spectral difference
    dfu = jnp.einsum("...j, j -> ...j", fu, factor)

    # re-reshape axis
    dfu = jnp.moveaxis(dfu, -1, axis)

    return dfu


def difference(
    u: Array, k_vec: Array, axis: int = 0, derivative: int = 1,
    real: bool=True
) -> Array:
    """spectral difference from the real space

    Args:
        u (Array): the array in real space to do the difference
        k_vec (Array): the vector of frequencies
        axis (int, optional): the axis to do the difference.
            Defaults to 0.
        derivative (int, optional): the number of the derivatives to do.
            Defaults to 1.
        real (bool, optional): to return the array in real or complex.
            Defaults to True.

    Returns:
        du (Array): the resulting array with the derivatives
    """
    # forward transformation
    fu = fft_transform(u, axis=axis, inverse=False)

    # difference operator
    dfu = spectral_difference(fu, k_vec=k_vec, axis=axis, derivative=derivative)

    # inverse transformation
    du = fft_transform(dfu, axis=axis, inverse=True)

    # return real components
    return jnp.real(du)


def elliptical_operator(
    k_vec: tp.Iterable[Array],
    order: int = 1,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Array:

    # multiply by order
    fn = lambda x: x ** (2 * order)
    k_vec = [fn(ik) for ik in k_vec]

    # expand each of the dimensions
    ks = [jnp.expand_dims(array, axis=i) for i, array in enumerate(k_vec[::-1])]

    # sum each of dimensions
    ksq = ft.reduce(lambda x, y: x + y, ks)

    # add alpha and beta
    ksq = alpha * ksq + beta

    return ksq
