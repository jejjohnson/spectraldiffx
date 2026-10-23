"""Base classes and the grid spec for :mod:`spectraldiffx.pipekit`.

Two tiers::

    Tier A  spectraldiffx primitives     SpectralDerivative2D(grid).laplacian(u)
            arrays in, arrays out         eqx.Module, no config
                    │
                    ▼
    Tier B  spectraldiffx.pipekit         Laplacian(grid=G, out="spectral")
            pipekit Operators             JSON config · eqx leaves · Graph nodes,
                                          physical or Spectral carrier

A Tier-B operator stores its grid as a static JSON tuple (`GridSpec`) and
rebuilds the Tier-A grid at trace time, so the config round-trips through
``op.state`` / ``Operator.from_state`` and no grid crosses ``jit`` as a
pytree of Python scalars. Fields may carry leading (time, ensemble) axes:
the FFTs act on the last two.
"""

from __future__ import annotations

from collections.abc import Sequence
import functools
from typing import Any, Literal, cast

import equinox as eqx
from jaxtyping import Array
from pipekit_jax import JaxOperator

from spectraldiffx._src.fourier.grid import FourierGrid2D
from spectraldiffx._src.pipekit.carrier import Spectral, to_physical

#: ``(Nx, Ny, Lx, Ly, dealias)``: the arguments of `FourierGrid2D.from_N_L`.
GridSpec = tuple[int, int, float, float, Literal["2/3"] | None]

Out = Literal["physical", "spectral"]


def as_grid_spec(spec: Sequence[Any] | FourierGrid2D) -> GridSpec:
    """Normalise a grid description to a `GridSpec` tuple.

    Parameters
    ----------
    spec : sequence or FourierGrid2D
        ``(Nx, Ny, Lx, Ly)`` or ``(Nx, Ny, Lx, Ly, dealias)`` (a list after
        a JSON round trip), or a grid to describe. ``dealias`` defaults to
        ``"2/3"``.

    Returns
    -------
    GridSpec
        ``(int, int, float, float, "2/3" | None)``, hashable.

    Raises
    ------
    ValueError
        On a wrong length, a non-positive size or an unknown ``dealias``.
    """
    values: tuple[Any, ...]
    if isinstance(spec, FourierGrid2D):
        values = cast(
            tuple[Any, ...], (spec.Nx, spec.Ny, spec.Lx, spec.Ly, spec.dealias)
        )
    else:
        values = tuple(spec)
    if len(values) == 4:
        values = (*values, "2/3")
    if len(values) != 5:
        raise ValueError(
            f"grid spec must be (Nx, Ny, Lx, Ly[, dealias]), got {values!r}"
        )
    nx, ny, lx, ly, dealias = values
    if int(nx) < 1 or int(ny) < 1:
        raise ValueError(f"grid spec needs positive sizes, got {values!r}")
    if dealias not in ("2/3", None):
        raise ValueError(f"grid spec dealias must be '2/3' or None, got {dealias!r}")
    return (int(nx), int(ny), float(lx), float(ly), dealias)


@functools.lru_cache(maxsize=64)
def build_grid(spec: GridSpec) -> FourierGrid2D:
    """The `FourierGrid2D` a `GridSpec` describes (cached per spec)."""
    nx, ny, lx, ly, dealias = as_grid_spec(spec)
    return FourierGrid2D.from_N_L(nx, ny, lx, ly, dealias=dealias)


def _check_out(out: str) -> None:
    if out not in ("physical", "spectral"):
        raise ValueError(f"out must be 'physical' or 'spectral', got {out!r}")


class SpectralOperator(JaxOperator):
    """Base for Tier-B operators on a 2-D Fourier grid.

    Inputs may be real physical fields or `Spectral`; the output is chosen
    by ``out``.

    Attributes
    ----------
    grid : GridSpec
        ``(Nx, Ny, Lx, Ly, dealias)``, static (a list or a
        `FourierGrid2D` is converted on construction).
    out : {"physical", "spectral"}
        Emit a real field or `Spectral` coefficients.
    """

    grid: GridSpec = eqx.field(static=True, converter=as_grid_spec)
    out: Out = eqx.field(static=True, default="physical", kw_only=True)

    def __check_init__(self) -> None:
        _check_out(self.out)

    @property
    def fourier_grid(self) -> FourierGrid2D:
        """The Tier-A grid, rebuilt from `grid` (cached per spec)."""
        return build_grid(self.grid)

    @property
    def spectral_out(self) -> bool:
        """Whether the operator emits `Spectral` coefficients."""
        return self.out == "spectral"

    def emit(self, y: Array) -> Array | Spectral:
        """Wrap a Tier-A result: coefficients → `Spectral`, physical → as is."""
        return Spectral(y) if self.spectral_out else y

    def emit_hat(self, hat: Array) -> Array | Spectral:
        """Coefficients → `Spectral`, or their real inverse transform."""
        if self.spectral_out:
            return Spectral(hat)
        return to_physical(hat, self.fourier_grid)
