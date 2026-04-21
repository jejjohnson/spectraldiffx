"""Chebyshev pseudo-spectral methods for non-periodic domains.

Classes exported
----------------
Grids:
    ChebyshevGrid1D, ChebyshevGrid2D

Transforms:
    ChebyshevTransform1D, ChebyshevTransform2D

Derivative operators:
    ChebyshevDerivative1D, ChebyshevDerivative2D

Filters:
    ChebyshevFilter1D, ChebyshevFilter2D

Solvers:
    ChebyshevHelmholtzSolver1D, ChebyshevHelmholtzSolver2D
    ChebyshevPoissonSolver1D, ChebyshevPoissonSolver2D

Quadrature:
    clenshaw_curtis_weights, clenshaw_curtis_integrate_1d,
    clenshaw_curtis_integrate_2d
"""

from __future__ import annotations

from .filters import ChebyshevFilter1D, ChebyshevFilter2D
from .grid import ChebyshevGrid1D, ChebyshevGrid2D
from .operators import ChebyshevDerivative1D, ChebyshevDerivative2D
from .quadrature import (
    clenshaw_curtis_integrate_1d,
    clenshaw_curtis_integrate_2d,
    clenshaw_curtis_weights,
)
from .solvers import (
    ChebyshevHelmholtzSolver1D,
    ChebyshevHelmholtzSolver2D,
    ChebyshevPoissonSolver1D,
    ChebyshevPoissonSolver2D,
)
from .transforms import ChebyshevTransform1D, ChebyshevTransform2D

__all__ = [
    "ChebyshevDerivative1D",
    "ChebyshevDerivative2D",
    "ChebyshevFilter1D",
    "ChebyshevFilter2D",
    "ChebyshevGrid1D",
    "ChebyshevGrid2D",
    "ChebyshevHelmholtzSolver1D",
    "ChebyshevHelmholtzSolver2D",
    "ChebyshevPoissonSolver1D",
    "ChebyshevPoissonSolver2D",
    "ChebyshevTransform1D",
    "ChebyshevTransform2D",
    "clenshaw_curtis_integrate_1d",
    "clenshaw_curtis_integrate_2d",
    "clenshaw_curtis_weights",
]
