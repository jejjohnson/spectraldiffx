"""Chebyshev pseudo-spectral methods for non-periodic domains.

Classes exported
----------------
Grids:
    ChebyshevGrid1D, ChebyshevGrid2D, ChebyshevGrid3D

Transforms:
    ChebyshevTransform1D, ChebyshevTransform2D

Derivative operators:
    ChebyshevDerivative1D, ChebyshevDerivative2D, ChebyshevDerivative3D

Filters:
    ChebyshevFilter1D, ChebyshevFilter2D

Solvers:
    ChebyshevHelmholtzSolver1D, ChebyshevHelmholtzSolver2D
    ChebyshevPoissonSolver1D, ChebyshevPoissonSolver2D

Quadrature:
    clenshaw_curtis_weights, clenshaw_curtis_integrate_1d,
    clenshaw_curtis_integrate_2d

Coefficient-space calculus:
    chebyshev_derivative_coeffs, chebyshev_antiderivative_coeffs,
    chebyshev_integral_coeffs
"""

from __future__ import annotations

from .filters import ChebyshevFilter1D, ChebyshevFilter2D
from .grid import ChebyshevGrid1D, ChebyshevGrid2D, ChebyshevGrid3D
from .operators import (
    ChebyshevDerivative1D,
    ChebyshevDerivative2D,
    ChebyshevDerivative3D,
)
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
from .transforms import (
    ChebyshevTransform1D,
    ChebyshevTransform2D,
    chebyshev_antiderivative_coeffs,
    chebyshev_derivative_coeffs,
    chebyshev_integral_coeffs,
)

__all__ = [
    "ChebyshevDerivative1D",
    "ChebyshevDerivative2D",
    "ChebyshevDerivative3D",
    "ChebyshevFilter1D",
    "ChebyshevFilter2D",
    "ChebyshevGrid1D",
    "ChebyshevGrid2D",
    "ChebyshevGrid3D",
    "ChebyshevHelmholtzSolver1D",
    "ChebyshevHelmholtzSolver2D",
    "ChebyshevPoissonSolver1D",
    "ChebyshevPoissonSolver2D",
    "ChebyshevTransform1D",
    "ChebyshevTransform2D",
    "chebyshev_antiderivative_coeffs",
    "chebyshev_derivative_coeffs",
    "chebyshev_integral_coeffs",
    "clenshaw_curtis_integrate_1d",
    "clenshaw_curtis_integrate_2d",
    "clenshaw_curtis_weights",
]
