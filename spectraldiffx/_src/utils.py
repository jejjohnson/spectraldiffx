from typing import Iterable
import math
import jax.numpy as jnp
from jaxtyping import Array


def fft_transform(u: Array, axis: int = -1, inverse: bool = False) -> Array:
    """the FFT transformation (forward and inverse)

    Args:
        u (Array): the input array to be transformed
        axis (int, optional): the axis to do the FFT transformation. Defaults to -1.
        scale (float, optional): the scaler value to rescale the inputs.
            Defaults to 1.0.
        inverse (bool, optional): whether to do the forward or inverse transformation.
            Defaults to False.

    Returns:
        u (Array): the transformation that maybe forward or backwards
    """
    # check if complex or real (save time!!!)
    if inverse:
        return jnp.fft.fft(a=u, axis=axis)
    else:
        return jnp.fft.ifft(a=u, axis=axis)
    


def calculate_fft_freq(N: int, L: float = 2.0 * math.pi) -> Array:
    """a helper function to generate 1D FFT frequencies

    Args:
        Nx (int): the number of points for the grid
        Lx (float): the distance for the points along the grid

    Returns:
        freq (Array): the 1D fourier frequencies
    """
    # return jnp.fft.fftfreq(n=Nx, d=Lx / (2.0 * math.pi * Nx))
    return (2 * math.pi / L) * jnp.fft.fftfreq(n=N, d=L / (N * 2.0 * math.pi))


def calculate_aliasing(kvec: Array, ratio: float = 1.0 / 3.0) -> Array:
    """Generates a conditional vector the anti-aliasing.
    Uses a ratio of wavenumbers that we say we remove.

    Args:
        kvec (Array): a list of 1D fourier frequencies
        ratio (float, optional): the ratio of frequenies to remove
            Defaults to 1.0/3.0.

    Returns:
        cond (Array): a boolean vector for which wavenumber frequencies
            are kept and which are cut
    """
    return jnp.abs(kvec) > (1 - ratio) * jnp.abs(kvec).max()
