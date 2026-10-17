"""Contract check and transform counting for :mod:`spectraldiffx.pipekit`."""

from __future__ import annotations

import re
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit_jax.testing import check_jax_operator

from spectraldiffx._src.pipekit.carrier import Spectral

_FFT_TYPE = re.compile(r"fft_type=(\w+)")


def compiled_hlo(fn: Any, *args: Any) -> str:
    """Optimised HLO text of ``fn(*args)`` (``fn`` may be an eqx module)."""
    dynamic, static = eqx.partition((fn, args), eqx.is_array)

    def run(d: Any) -> Any:
        f, xs = eqx.combine(d, static)
        return f(*xs)

    text = jax.jit(run).lower(dynamic).compile().as_text()
    return text or ""


def count_transforms(fn: Any, *args: Any) -> dict[str, int]:
    """FFT ops in the compiled HLO of ``fn(*args)``, by ``fft_type``.

    Returns
    -------
    dict[str, int]
        e.g. ``{"FFT": 1, "IFFT": 4}``; an empty dict means no transform.
    """
    kinds = _FFT_TYPE.findall(compiled_hlo(fn, *args))
    return {k: kinds.count(k) for k in sorted(set(kinds))}


def _members(inputs: tuple[Any, ...], n: int) -> list[tuple[Any, ...]]:
    """``n`` distinct members: member ``m`` scales every input by ``1 + m/4``."""
    return [
        tuple(
            jax.tree.map(lambda a, m=m: a * (1.0 + m / 4), x)
            if eqx.is_array(x) or isinstance(x, Spectral)
            else x
            for x in inputs
        )
        for m in range(n)
    ]


def _real_sum(y: Any) -> Any:
    return sum(jnp.sum(jnp.real(leaf)) for leaf in jax.tree.leaves(y))


def check_operator(op: Any, *inputs: Any, budget: dict[str, int] | None = None) -> None:
    """Assert that ``op`` is a well-behaved spectraldiffx pipekit operator.

    Runs `pipekit_jax.testing.check_jax_operator` (operator + module, eager
    ≡ ``filter_jit``, ``vmap``, visible leaves, JSON config round trip),
    then checks:

    1. **Leading axes**: inputs stacked to ``[2, ...]`` and ``[2, 3, ...]``
       equal a loop over the members.
    2. **Gradient**: ∂/∂x₀ of the summed real output is finite.
    3. **Transform budget** (when ``budget`` is given): the compiled HLO
       has exactly these FFT counts (`count_transforms`).

    Parameters
    ----------
    op : SpectralOperator
        Operator under test.
    *inputs : Array or Spectral
        Example inputs ``[Ny, Nx]``.
    budget : dict[str, int], optional
        Expected `count_transforms` result, e.g. ``{"FFT": 1}``.

    Raises
    ------
    AssertionError
        On the first violated property.
    """
    check_jax_operator(op, *inputs)

    for lead in ((2,), (2, 3)):
        members = _members(inputs, int(np.prod(lead)))
        per = [op(*m) for m in members]
        want = jax.tree.map(
            lambda *ys, lead=lead: jnp.stack(ys).reshape(*lead, *ys[0].shape), *per
        )
        batched = jax.tree.map(
            lambda *xs, lead=lead: jnp.stack(xs).reshape(*lead, *xs[0].shape),
            *members,
        )
        got = op(*batched)
        assert jax.tree.structure(got) == jax.tree.structure(want), (
            f"leading axes {lead}: output structure differs"
        )
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            tol = 1e3 * float(jnp.finfo(a.dtype).eps)  # FFT rounding, O(N log N)
            np.testing.assert_allclose(
                a, b, rtol=tol, atol=tol, err_msg=f"leading axes {lead}"
            )

    first = next(
        i
        for i, x in enumerate(inputs)
        if eqx.is_inexact_array(x) or isinstance(x, Spectral)
    )

    def loss(x: Any) -> Any:
        xs = list(inputs)
        xs[first] = x
        return _real_sum(op(*xs))

    grad = jax.grad(loss)(inputs[first])
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grad)), (
        "gradient w.r.t. the first input"
    )

    if budget is not None:
        counts = count_transforms(op, *inputs)
        assert counts == budget, f"transform budget {budget}, compiled {counts}"
