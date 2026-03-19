"""JAX-native spectral transforms: DCT and DST types I–IV.

All transforms follow the scipy convention and support two normalization modes:

* ``norm=None`` (default) — unnormalized forward, normalized inverse.
* ``norm="ortho"`` — orthonormal: the transform matrix is orthogonal, so
  Parseval's theorem holds without extra factors.

Unnormalized definitions (``norm=None``):

* DCT-I:   Y[k] = x[0] + (-1)^k x[N-1] + 2 Σ_{n=1}^{N-2} x[n] cos(πnk/(N-1))
* DCT-II:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] cos(πk(2n+1)/(2N))
* DCT-III: Y[k] = x[0] + 2 Σ_{n=1}^{N-1} x[n] cos(πn(2k+1)/(2N))
* DCT-IV:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] cos(π(2n+1)(2k+1)/(4N))

* DST-I:   Y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(n+1)(k+1)/(N+1))
* DST-II:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(2n+1)(k+1)/(2N))
* DST-III: Y[k] = (-1)^k x[N-1] + 2 Σ_{n=0}^{N-2} x[n] sin(π(n+1)(2k+1)/(2N))
* DST-IV:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(2n+1)(2k+1)/(4N))

Orthonormal scaling (``norm="ortho"``):

The unnormalized output is multiplied by a uniform factor and, for certain
types, edge elements receive an additional correction to make the transform
matrix orthogonal (C @ C^T = I):

* DCT-I:   uniform sqrt(1/(2(N-1))), then y[0] and y[-1] *= 1/sqrt(2)
* DCT-II:  uniform sqrt(1/(2N)),     then y[0] *= 1/sqrt(2)
* DCT-III: pre-scale x[0] *= sqrt(2), then uniform sqrt(1/(2N))
* DCT-IV:  uniform sqrt(1/(2N))

* DST-I:   uniform sqrt(1/(2(N+1)))
* DST-II:  uniform sqrt(1/(2N)),     then y[-1] *= 1/sqrt(2)
* DST-III: pre-scale x[-1] *= sqrt(2), then uniform sqrt(1/(2N))
* DST-IV:  uniform sqrt(1/(2N))

Base transforms (``dct``, ``dst``, ``idct``, ``idst``) operate on 1-D vectors
only. Multi-axis helpers (``dctn``, ``dstn``, ``idctn``, ``idstn``) apply the
corresponding transform along each requested axis in sequence.

Inverse transforms satisfy:
  idct(dct(x, t, norm=m), t, norm=m) == x   for all t ∈ {1,2,3,4}, m ∈ {None, "ortho"}
  idst(dst(x, t, norm=m), t, norm=m) == x   for all t ∈ {1,2,3,4}, m ∈ {None, "ortho"}

All functions accept JAX arrays and are compatible with ``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT_HALF = 1.0 / math.sqrt(2.0)


def _sl(x: Array, start: int, stop: int, axis: int) -> Array:
    """Take a contiguous slice [start:stop] along *axis*."""
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(start, stop)
    return x[tuple(idx)]


def _phase_shape(ndim: int, axis: int, size: int) -> tuple[int, ...]:
    """Return a shape broadcastable with an array of *ndim* dims along *axis*."""
    s = [1] * ndim
    s[axis] = size
    return tuple(s)


def _norm_axis(axis: int, ndim: int) -> int:
    """Normalise a (possibly negative) axis index."""
    if axis < 0:
        axis = axis + ndim
    return axis


def _make_idx(ndim: int, axis: int, pos: int | slice) -> tuple:
    """Create index tuple selecting position *pos* along *axis*."""
    idx = [slice(None)] * ndim
    idx[axis] = pos
    return tuple(idx)


# ---------------------------------------------------------------------------
# Ortho normalization helpers
# ---------------------------------------------------------------------------


def _ortho_uniform_factor(N: int, type: int, transform: str) -> float:
    """Uniform ortho scale factor applied to all output elements.

    DST-I:  sqrt(1 / (2(N+1)))
    All others: sqrt(1 / (2N))

    Note: DCT-I ortho is handled separately by :func:`_dct1_ortho` because
    the unnormalized DCT-I has asymmetric weights (1x for endpoints, 2x for
    interior) that cannot be corrected by simple output post-processing.
    """
    if transform == "dst" and type == 1:
        return math.sqrt(0.5 / (N + 1))
    return math.sqrt(0.5 / N)


def _dct1_ortho(x: Array, axis: int) -> Array:
    """Orthonormal DCT Type I.

    Computes the ortho DCT-I directly because the unnormalized DCT-I has
    asymmetric weights (coefficient 1 for x[0] and x[N-1], coefficient 2
    for interior points), so the ortho form cannot be obtained by simple
    post-scaling of the unnormalized output.

    The orthogonal DCT-I matrix has entries:

        C[k,n] = c(k) c(n) sqrt(2/(N-1)) cos(pi n k / (N-1))

    where c(0) = c(N-1) = 1/sqrt(2), c(j) = 1 for 0 < j < N-1.

    The ortho DCT-I is symmetric (C = C^T) and orthogonal (C C^T = I),
    therefore it is its own inverse: idct_ortho(y, 1) == dct_ortho(y, 1).
    """
    N = x.shape[axis]
    # Build c vector: [1/sqrt(2), 1, ..., 1, 1/sqrt(2)]
    c = jnp.ones(N)
    c = c.at[0].set(_SQRT_HALF)
    c = c.at[N - 1].set(_SQRT_HALF)
    c = c.reshape(_phase_shape(x.ndim, axis, N))
    # Scale input by c
    x_s = x * c
    # Unnormalized DCT-I of scaled input:
    #   Y'[k] = x_s[0] + (-1)^k x_s[-1] + 2 sum_{n=1}^{N-2} x_s[n] cos(...)
    y_unnorm = _dct1(x_s, axis)
    # Extract the pure cosine sum (without the 2x interior factor):
    #   S[k] = x_s[0] + sum_mid + (-1)^k x_s[-1]
    #        = (Y'[k] + x_s[0] + (-1)^k x_s[-1]) / 2
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    idx0 = _make_idx(x.ndim, axis, slice(0, 1))
    idxN = _make_idx(x.ndim, axis, slice(N - 1, N))
    endpoint_term = x_s[idx0] + (-1.0) ** k * x_s[idxN]
    pure_sum = (y_unnorm + endpoint_term) / 2.0
    # Scale output: c(k) * sqrt(2/(N-1))
    return pure_sum * c * math.sqrt(2.0 / (N - 1))


def _apply_ortho_forward(y: Array, type: int, axis: int, transform: str) -> Array:
    """Apply ortho post-scaling to the output of an unnormalized forward transform.

    For type 3 the input must be pre-scaled *before* the unnormalized forward
    via :func:`_prescale_type3` — this function only handles the output scaling.

    Note: DCT type 1 ortho is handled by :func:`_dct1_ortho` instead.
    """
    N = y.shape[axis]
    factor = _ortho_uniform_factor(N, type, transform)
    y = y * factor
    # Edge corrections
    if transform == "dct":
        if type == 2:
            y = y.at[_make_idx(y.ndim, axis, 0)].multiply(_SQRT_HALF)
    elif transform == "dst":
        if type == 2:
            y = y.at[_make_idx(y.ndim, axis, -1)].multiply(_SQRT_HALF)
    return y


def _remove_ortho_forward(y: Array, type: int, axis: int, transform: str) -> Array:
    """Undo ortho post-scaling to recover the unnormalized transform output.

    Reverses :func:`_apply_ortho_forward`: undoes edge corrections, then
    divides by the uniform scale factor.

    Note: DCT type 1 ortho is handled by :func:`_dct1_ortho` instead
    (it is self-inverse).
    """
    N = y.shape[axis]
    # Undo edge corrections (multiply by reciprocal)
    if transform == "dct":
        if type == 2:
            y = y.at[_make_idx(y.ndim, axis, 0)].multiply(_SQRT2)
    elif transform == "dst":
        if type == 2:
            y = y.at[_make_idx(y.ndim, axis, -1)].multiply(_SQRT2)
    # Undo uniform factor
    factor = _ortho_uniform_factor(N, type, transform)
    return y / factor


def _prescale_type3(x: Array, axis: int, transform: str) -> Array:
    """Pre-scale input for type 3 ortho forward.

    DCT-III has asymmetric input weight on x[0] (coefficient 1 vs 2 for
    the rest).  Pre-multiplying x[0] by sqrt(2) compensates, so that the
    uniform output scale alone produces an orthogonal matrix.

    DST-III has the same asymmetry on x[-1].
    """
    if transform == "dct":
        return x.at[_make_idx(x.ndim, axis, 0)].multiply(_SQRT2)
    else:
        return x.at[_make_idx(x.ndim, axis, -1)].multiply(_SQRT2)


def _undo_prescale_type3(x: Array, axis: int, transform: str) -> Array:
    """Undo the type 3 pre-scaling after the unnormalized inverse."""
    if transform == "dct":
        return x.at[_make_idx(x.ndim, axis, 0)].multiply(_SQRT_HALF)
    else:
        return x.at[_make_idx(x.ndim, axis, -1)].multiply(_SQRT_HALF)


def _validate_norm(norm: str | None) -> None:
    """Raise ValueError if *norm* is not None or "ortho"."""
    if norm is not None and norm != "ortho":
        raise ValueError(f"norm must be None or 'ortho'; got {norm!r}")


# ---------------------------------------------------------------------------
# DCT types I-IV  (unnormalized, scipy-compatible)
# ---------------------------------------------------------------------------


def _dct1(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type I via rfft (even-extension trick).

    Even-extension of length 2*(N-1):
        v = [x[0], x[1], ..., x[N-1], x[N-2], ..., x[1]]
    DCT-I[k] = Re( rfft(v)[k] )
    """
    N = x.shape[axis]
    interior = _sl(x, 1, N - 1, axis)
    v = jnp.concatenate([x, jnp.flip(interior, axis=axis)], axis=axis)
    V = jnp.fft.rfft(v, axis=axis)
    return jnp.real(_sl(V, 0, N, axis))


def _dct2(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type II via Makhoul algorithm (N-point FFT).

    Reorder: v = [x[0::2], x[1::2][::-1]]
    DCT-II[k] = Re( 2·exp(−iπk/(2N)) · FFT(v)[k] )
    """
    N = x.shape[axis]
    idx_even = [slice(None)] * x.ndim
    idx_even[axis] = slice(0, None, 2)
    idx_odd = [slice(None)] * x.ndim
    idx_odd[axis] = slice(1, None, 2)
    even = x[tuple(idx_even)]
    odd_rev = jnp.flip(x[tuple(idx_odd)], axis=axis)
    v = jnp.concatenate([even, odd_rev], axis=axis)
    V = jnp.fft.fft(v, axis=axis)
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    phase = 2.0 * jnp.exp(-1j * jnp.pi * k / (2 * N))
    return jnp.real(V * phase)


def _dct3(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type III via irfft.

    Form B[k] = x[k]·exp(iπk/(2N)), B[N]=0 (Hermitian one-sided spectrum).
    DCT-III[n] = 2N · irfft([B[0], ..., B[N-1], 0], n=2N)[n]

    Note: DCT-III is the unnormalized inverse of DCT-II.
    """
    N = x.shape[axis]
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    phase = jnp.exp(1j * jnp.pi * k / (2 * N))
    b = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * phase
    zero_shape = list(b.shape)
    zero_shape[axis] = 1
    zeros = jnp.zeros(zero_shape, dtype=b.dtype)
    B = jnp.concatenate([b, zeros], axis=axis)
    out_full = jnp.fft.irfft(B, n=2 * N, axis=axis)
    return 2.0 * N * _sl(out_full, 0, N, axis)


def _dct4(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type IV via zero-padded IFFT.

    w[n] = x[n]·exp(iπ(2n+1)/(4N))
    W_pad = [w[0], ..., w[N-1], 0, ..., 0]  (length 2N)
    A[k]  = 2N · IFFT(W_pad)[k] · exp(iπk/(2N))
    DCT-IV[k] = 2·Re(A[k])
    """
    N = x.shape[axis]
    n = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    w = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * jnp.exp(
        1j * jnp.pi * (2 * n + 1) / (4 * N)
    )
    zero_shape = list(w.shape)
    zero_shape[axis] = N
    zeros = jnp.zeros(zero_shape, dtype=w.dtype)
    W_pad = jnp.concatenate([w, zeros], axis=axis)
    A = 2.0 * N * jnp.fft.ifft(W_pad, axis=axis)
    A = _sl(A, 0, N, axis) * jnp.exp(1j * jnp.pi * k / (2 * N))
    return 2.0 * jnp.real(A)


# ---------------------------------------------------------------------------
# DST types I-IV  (unnormalized, scipy-compatible)
# ---------------------------------------------------------------------------


def _dst1(x: Array, axis: int) -> Array:
    """Unnormalized DST Type I via rfft (odd-extension trick).

    Odd-antisymmetric extension of length 2*(N+1):
        v = [0, x[0], ..., x[N-1], 0, −x[N-1], ..., −x[0]]
    DST-I[k] = −Im( rfft(v)[k+1] )
    """
    N = x.shape[axis]
    zero_shape = list(x.shape)
    zero_shape[axis] = 1
    zeros = jnp.zeros(zero_shape, dtype=x.dtype)
    v = jnp.concatenate([zeros, x, zeros, -jnp.flip(x, axis=axis)], axis=axis)
    V = jnp.fft.rfft(v, axis=axis)
    return -jnp.imag(_sl(V, 1, N + 1, axis))


def _dst2(x: Array, axis: int) -> Array:
    """Unnormalized DST Type II via rfft.

    F = rfft(x, n=2N)
    DST-II[k] = 2·Im( exp(iπ(k+1)/(2N)) · conj(F[k+1]) )
    """
    N = x.shape[axis]
    F = jnp.fft.rfft(x, n=2 * N, axis=axis)
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    phase = jnp.exp(1j * jnp.pi * (k + 1) / (2 * N))
    F_slice = _sl(F, 1, N + 1, axis)
    return 2.0 * jnp.imag(phase * jnp.conj(F_slice))


def _dst3(x: Array, axis: int) -> Array:
    """Unnormalized DST Type III via the DCT-III / reversal identity.

    Let z[n] = (−1)^n · x[N−1−n], then
        DST-III(x)[k] = (−1)^k · DCT-III(z)[N−1−k]

    Note: DST-III is the unnormalized inverse of DST-II.
    """
    N = x.shape[axis]
    n = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    z = (-1.0) ** n * jnp.flip(x, axis=axis)
    dct3_z = _dct3(z, axis)
    return (-1.0) ** k * jnp.flip(dct3_z, axis=axis)


def _dst4(x: Array, axis: int) -> Array:
    """Unnormalized DST Type IV via zero-padded IFFT.

    Uses the same intermediate array A as DCT-IV:
        DST-IV[k] = 2·Im(A[k])   (A defined in ``_dct4``)
    """
    N = x.shape[axis]
    n = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    w = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * jnp.exp(
        1j * jnp.pi * (2 * n + 1) / (4 * N)
    )
    zero_shape = list(w.shape)
    zero_shape[axis] = N
    zeros = jnp.zeros(zero_shape, dtype=w.dtype)
    W_pad = jnp.concatenate([w, zeros], axis=axis)
    A = 2.0 * N * jnp.fft.ifft(W_pad, axis=axis)
    A = _sl(A, 0, N, axis) * jnp.exp(1j * jnp.pi * k / (2 * N))
    return 2.0 * jnp.imag(A)


# ---------------------------------------------------------------------------
# Internal dispatch tables
# ---------------------------------------------------------------------------

_DCT_IMPLS = {1: _dct1, 2: _dct2, 3: _dct3, 4: _dct4}
_DST_IMPLS = {1: _dst1, 2: _dst2, 3: _dst3, 4: _dst4}

# ---------------------------------------------------------------------------
# Inverse helpers (used by both 1D and n-D public functions)
# ---------------------------------------------------------------------------


def _idct_along_axis(x: Array, type: int, axis: int) -> Array:
    """Unnormalized inverse DCT along a single axis (arbitrary ndim).

    Inverse scaling:
    * IDCT-I   = DCT-I(x)   / (2(N-1))  — DCT-I is self-inverse up to scale
    * IDCT-II  = DCT-III(x) / (2N)      — uses irfft shortcut for efficiency
    * IDCT-III = DCT-II(x)  / (2N)
    * IDCT-IV  = DCT-IV(x)  / (2N)      — DCT-IV is self-inverse up to scale
    """
    N = x.shape[axis]
    if type == 1:
        # IDCT-I = DCT-I(x) / (2(N-1))
        return _dct1(x, axis) / (2 * (N - 1))
    if type == 2:
        # IDCT-II = DCT-III(x) / (2N) — uses irfft shortcut for efficiency
        k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
        phase = jnp.exp(1j * jnp.pi * k / (2 * N))
        b = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * phase
        zero_shape = list(b.shape)
        zero_shape[axis] = 1
        zeros = jnp.zeros(zero_shape, dtype=b.dtype)
        B = jnp.concatenate([b, zeros], axis=axis)
        out_full = jnp.fft.irfft(B, n=2 * N, axis=axis)
        return _sl(out_full, 0, N, axis)
    if type == 3:
        # IDCT-III = DCT-II(x) / (2N)
        return _dct2(x, axis) / (2 * N)
    if type == 4:
        # IDCT-IV = DCT-IV(x) / (2N) — DCT-IV is self-inverse up to scale
        return _dct4(x, axis) / (2 * N)
    raise ValueError(f"DCT type must be 1, 2, 3, or 4; got {type}")


def _idst_along_axis(x: Array, type: int, axis: int) -> Array:
    """Unnormalized inverse DST along a single axis (arbitrary ndim).

    Inverse scaling:
    * IDST-I   = DST-I(x)   / (2(N+1))  — DST-I is self-inverse up to scale
    * IDST-II  = DST-III(x) / (2N)
    * IDST-III = DST-II(x)  / (2N)
    * IDST-IV  = DST-IV(x)  / (2N)      — DST-IV is self-inverse up to scale
    """
    N = x.shape[axis]
    if type == 1:
        # IDST-I = DST-I(x) / (2(N+1))
        return _dst1(x, axis) / (2 * (N + 1))
    if type == 2:
        # IDST-II = DST-III(x) / (2N)
        return _dst3(x, axis) / (2 * N)
    if type == 3:
        # IDST-III = DST-II(x) / (2N)
        return _dst2(x, axis) / (2 * N)
    if type == 4:
        # IDST-IV = DST-IV(x) / (2N) — DST-IV is self-inverse up to scale
        return _dst4(x, axis) / (2 * N)
    raise ValueError(f"DST type must be 1, 2, 3, or 4; got {type}")


# ---------------------------------------------------------------------------
# Public 1-D transforms (vector-only)
# ---------------------------------------------------------------------------


def dct(
    x: Float[Array, " N"],
    type: Literal[1, 2, 3, 4] = 2,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, " N"]:
    """Discrete Cosine Transform of a 1-D vector.

    Computes the DCT of *x* using an FFT-based O(N log N) algorithm.
    The default type is DCT-II:

        Y[k] = 2 Sigma_{n=0}^{N-1} x[n] cos(pi k(2n+1) / (2N)),  k = 0, ..., N-1

    All four types (I-IV) follow the scipy convention.

    Parameters
    ----------
    x : Float[Array, " N"]
        Input 1-D array of length N.
    type : {1, 2, 3, 4}
        DCT variant.  Default: 2.
    norm : {None, "ortho"}
        Normalization mode.  ``None`` (default) gives the unnormalized
        transform (``scipy.fft.dct(x, type, norm=None)``).  ``"ortho"``
        scales the output so the transform matrix is orthogonal.

    Returns
    -------
    Float[Array, " N"]
        DCT of *x*, same shape as input.

    Raises
    ------
    ValueError
        If *x* is not 1-D, *type* is invalid, or *norm* is unrecognised.
    """
    if x.ndim != 1:
        raise ValueError(
            f"dct expects a 1-D array, got ndim={x.ndim}. Use dctn for multi-dimensional input."
        )
    if type not in _DCT_IMPLS:
        raise ValueError(f"DCT type must be 1, 2, 3, or 4; got {type}")
    _validate_norm(norm)
    # DCT-I ortho requires a custom implementation (asymmetric weights)
    if norm == "ortho" and type == 1:
        return _dct1_ortho(x, axis=0)
    # Type 3 ortho requires pre-scaling the input
    if norm == "ortho" and type == 3:
        x = _prescale_type3(x, axis=0, transform="dct")
    y = _DCT_IMPLS[type](x, 0)
    if norm == "ortho":
        y = _apply_ortho_forward(y, type, axis=0, transform="dct")
    return y


def idct(
    x: Float[Array, " N"],
    type: Literal[1, 2, 3, 4] = 2,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, " N"]:
    """Inverse Discrete Cosine Transform of a 1-D vector.

    Satisfies ``idct(dct(x, t, norm=m), t, norm=m) == x`` for all types
    t in {1, 2, 3, 4} and normalization modes m in {None, "ortho"}.

    Parameters
    ----------
    x : Float[Array, " N"]
        DCT-transformed 1-D array of length N.
    type : {1, 2, 3, 4}
        DCT variant of the *forward* transform to invert.
    norm : {None, "ortho"}
        Normalization mode — must match the mode used in the forward transform.

    Returns
    -------
    Float[Array, " N"]
        Reconstructed signal, same shape as *x*.

    Raises
    ------
    ValueError
        If *x* is not 1-D, *type* is invalid, or *norm* is unrecognised.
    """
    if x.ndim != 1:
        raise ValueError(
            f"idct expects a 1-D array, got ndim={x.ndim}. Use idctn for multi-dimensional input."
        )
    _validate_norm(norm)
    # Ortho DCT-I is self-inverse (symmetric orthogonal matrix)
    if norm == "ortho" and type == 1:
        return _dct1_ortho(x, axis=0)
    if norm == "ortho":
        x = _remove_ortho_forward(x, type, axis=0, transform="dct")
    y = _idct_along_axis(x, type, axis=0)
    # Type 3 ortho: undo the pre-scaling applied in the forward pass
    if norm == "ortho" and type == 3:
        y = _undo_prescale_type3(y, axis=0, transform="dct")
    return y


def dst(
    x: Float[Array, " N"],
    type: Literal[1, 2, 3, 4] = 1,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, " N"]:
    """Discrete Sine Transform of a 1-D vector.

    Computes the DST of *x* using an FFT-based O(N log N) algorithm.
    The default type is DST-I:

        Y[k] = 2 Sigma_{n=0}^{N-1} x[n] sin(pi(n+1)(k+1) / (N+1)),  k = 0, ..., N-1

    All four types (I-IV) follow the scipy convention.

    Parameters
    ----------
    x : Float[Array, " N"]
        Input 1-D array of length N.
    type : {1, 2, 3, 4}
        DST variant.  Default: 1.
    norm : {None, "ortho"}
        Normalization mode.  ``None`` (default) gives the unnormalized
        transform.  ``"ortho"`` scales the output so the transform matrix
        is orthogonal.

    Returns
    -------
    Float[Array, " N"]
        DST of *x*, same shape as input.

    Raises
    ------
    ValueError
        If *x* is not 1-D, *type* is invalid, or *norm* is unrecognised.
    """
    if x.ndim != 1:
        raise ValueError(
            f"dst expects a 1-D array, got ndim={x.ndim}. Use dstn for multi-dimensional input."
        )
    if type not in _DST_IMPLS:
        raise ValueError(f"DST type must be 1, 2, 3, or 4; got {type}")
    _validate_norm(norm)
    if norm == "ortho" and type == 3:
        x = _prescale_type3(x, axis=0, transform="dst")
    y = _DST_IMPLS[type](x, 0)
    if norm == "ortho":
        y = _apply_ortho_forward(y, type, axis=0, transform="dst")
    return y


def idst(
    x: Float[Array, " N"],
    type: Literal[1, 2, 3, 4] = 1,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, " N"]:
    """Inverse Discrete Sine Transform of a 1-D vector.

    Satisfies ``idst(dst(x, t, norm=m), t, norm=m) == x`` for all types
    t in {1, 2, 3, 4} and normalization modes m in {None, "ortho"}.

    Parameters
    ----------
    x : Float[Array, " N"]
        DST-transformed 1-D array of length N.
    type : {1, 2, 3, 4}
        DST variant of the *forward* transform to invert.
    norm : {None, "ortho"}
        Normalization mode — must match the mode used in the forward transform.

    Returns
    -------
    Float[Array, " N"]
        Reconstructed signal, same shape as *x*.

    Raises
    ------
    ValueError
        If *x* is not 1-D, *type* is invalid, or *norm* is unrecognised.
    """
    if x.ndim != 1:
        raise ValueError(
            f"idst expects a 1-D array, got ndim={x.ndim}. Use idstn for multi-dimensional input."
        )
    _validate_norm(norm)
    if norm == "ortho":
        x = _remove_ortho_forward(x, type, axis=0, transform="dst")
    y = _idst_along_axis(x, type, axis=0)
    if norm == "ortho" and type == 3:
        y = _undo_prescale_type3(y, axis=0, transform="dst")
    return y


# ---------------------------------------------------------------------------
# Multi-axis transforms
# ---------------------------------------------------------------------------


def dctn(
    x: Float[Array, ...],
    type: Literal[1, 2, 3, 4] = 2,
    axes: Sequence[int] | None = None,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, ...]:
    """N-dimensional DCT: apply DCT sequentially along each axis.

    For a 2-D array with ``axes=[0, 1]``, this computes the separable
    transform ``DCT_y(DCT_x(x))``.  The result is identical to applying the
    1-D DCT independently along each axis in sequence.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array of any dimensionality.
    type : {1, 2, 3, 4}
        DCT variant.  Default: 2.
    axes : sequence of int or None
        Axes to transform.  ``None`` transforms all axes.
    norm : {None, "ortho"}
        Normalization mode.

    Returns
    -------
    Float[Array, "..."]
        N-D DCT of *x*, same shape as input.
    """
    if type not in _DCT_IMPLS:
        raise ValueError(f"DCT type must be 1, 2, 3, or 4; got {type}")
    _validate_norm(norm)
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        a = _norm_axis(ax, y.ndim)
        if norm == "ortho" and type == 1:
            y = _dct1_ortho(y, a)
        else:
            if norm == "ortho" and type == 3:
                y = _prescale_type3(y, a, "dct")
            y = _DCT_IMPLS[type](y, a)
            if norm == "ortho":
                y = _apply_ortho_forward(y, type, a, "dct")
    return y


def idctn(
    x: Float[Array, ...],
    type: Literal[1, 2, 3, 4] = 2,
    axes: Sequence[int] | None = None,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, ...]:
    """N-dimensional inverse DCT: apply IDCT sequentially along each axis.

    Satisfies ``idctn(dctn(x, t, axes, norm=m), t, axes, norm=m) == x``.
    The inverse is separable — each axis is inverted independently.

    Parameters
    ----------
    x : Float[Array, "..."]
        DCT-transformed array.
    type : {1, 2, 3, 4}
        DCT variant of the *forward* transform to invert.
    axes : sequence of int or None
        Axes to inverse-transform.  ``None`` transforms all axes.
    norm : {None, "ortho"}
        Normalization mode — must match the forward transform.

    Returns
    -------
    Float[Array, "..."]
        Reconstructed N-D array, same shape as input.
    """
    _validate_norm(norm)
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        a = _norm_axis(ax, y.ndim)
        if norm == "ortho" and type == 1:
            y = _dct1_ortho(y, a)
        else:
            if norm == "ortho":
                y = _remove_ortho_forward(y, type, a, "dct")
            y = _idct_along_axis(y, type, a)
            if norm == "ortho" and type == 3:
                y = _undo_prescale_type3(y, a, "dct")
    return y


def dstn(
    x: Float[Array, ...],
    type: Literal[1, 2, 3, 4] = 1,
    axes: Sequence[int] | None = None,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, ...]:
    """N-dimensional DST: apply DST sequentially along each axis.

    For a 2-D array with ``axes=[0, 1]``, this computes the separable
    transform ``DST_y(DST_x(x))``.  The result is identical to applying the
    1-D DST independently along each axis in sequence.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array of any dimensionality.
    type : {1, 2, 3, 4}
        DST variant.  Default: 1.
    axes : sequence of int or None
        Axes to transform.  ``None`` transforms all axes.
    norm : {None, "ortho"}
        Normalization mode.

    Returns
    -------
    Float[Array, "..."]
        N-D DST of *x*, same shape as input.
    """
    if type not in _DST_IMPLS:
        raise ValueError(f"DST type must be 1, 2, 3, or 4; got {type}")
    _validate_norm(norm)
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        a = _norm_axis(ax, y.ndim)
        if norm == "ortho" and type == 3:
            y = _prescale_type3(y, a, "dst")
        y = _DST_IMPLS[type](y, a)
        if norm == "ortho":
            y = _apply_ortho_forward(y, type, a, "dst")
    return y


def idstn(
    x: Float[Array, ...],
    type: Literal[1, 2, 3, 4] = 1,
    axes: Sequence[int] | None = None,
    norm: Literal["ortho"] | None = None,
) -> Float[Array, ...]:
    """N-dimensional inverse DST: apply IDST sequentially along each axis.

    Satisfies ``idstn(dstn(x, t, axes, norm=m), t, axes, norm=m) == x``.
    The inverse is separable — each axis is inverted independently.

    Parameters
    ----------
    x : Float[Array, "..."]
        DST-transformed array.
    type : {1, 2, 3, 4}
        DST variant of the *forward* transform to invert.
    axes : sequence of int or None
        Axes to inverse-transform.  ``None`` transforms all axes.
    norm : {None, "ortho"}
        Normalization mode — must match the forward transform.

    Returns
    -------
    Float[Array, "..."]
        Reconstructed N-D array, same shape as input.
    """
    _validate_norm(norm)
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        a = _norm_axis(ax, y.ndim)
        if norm == "ortho":
            y = _remove_ortho_forward(y, type, a, "dst")
        y = _idst_along_axis(y, type, a)
        if norm == "ortho" and type == 3:
            y = _undo_prescale_type3(y, a, "dst")
    return y
