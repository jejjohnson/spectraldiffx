"""
Spherical Harmonic Pseudo-Spectral Methods
===========================================

Spectral methods using Gauss-Legendre quadrature in latitude and Fourier in
longitude for PDEs on the sphere.  Suited to global geophysical fluid-dynamics
and atmospheric modeling.

Public API
----------
Grid classes:
    SphericalGrid1D, SphericalGrid2D

Transform:
    SphericalHarmonicTransform

Derivative operators:
    SphericalDerivative1D, SphericalDerivative2D

Filters:
    SphericalFilter1D, SphericalFilter2D

Solvers:
    SphericalPoissonSolver, SphericalHelmholtzSolver
"""

from .filters import SphericalFilter1D, SphericalFilter2D
from .grid import SphericalGrid1D, SphericalGrid2D
from .harmonics import SphericalHarmonicTransform
from .operators import SphericalDerivative1D, SphericalDerivative2D
from .solvers import SphericalHelmholtzSolver, SphericalPoissonSolver

__all__ = [
    "SphericalGrid1D",
    "SphericalGrid2D",
    "SphericalHarmonicTransform",
    "SphericalDerivative1D",
    "SphericalDerivative2D",
    "SphericalFilter1D",
    "SphericalFilter2D",
    "SphericalPoissonSolver",
    "SphericalHelmholtzSolver",
]
