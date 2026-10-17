"""spectraldiffx operators as pipekit Operators (Tier B).

Requires the ``pipekit`` extra (``pip install "spectraldiffx[pipekit]"``);
``import spectraldiffx`` never imports pipekit.

Every operator is a `pipekit_jax.JaxOperator`: a pipekit ``Operator``
(JSON config, ``Sequential`` / ``Graph`` nodes) that is also an
``eqx.Module`` (``jit`` / ``grad`` / ``vmap``). Edges carry a real
physical field or `Spectral` coefficients; each operator transforms only
what its carrier requires, and ``out`` picks the output space.
"""

from __future__ import annotations

try:
    import pipekit_jax as _pipekit_jax  # noqa: F401
except ImportError as err:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "spectraldiffx.pipekit needs the `pipekit` extra: "
        'pip install "spectraldiffx[pipekit]"'
    ) from err

from spectraldiffx._src.pipekit.base import (
    GridSpec,
    SpectralOperator,
    as_grid_spec,
    build_grid,
)
from spectraldiffx._src.pipekit.carrier import Spectral, as_hat, to_physical
from spectraldiffx._src.pipekit.transforms import ToPhysical, ToSpectral

__all__ = [
    "GridSpec",
    "Spectral",
    "SpectralOperator",
    "ToPhysical",
    "ToSpectral",
    "as_grid_spec",
    "as_hat",
    "build_grid",
    "to_physical",
]
