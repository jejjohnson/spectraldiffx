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
    (both include ``laplacian``, ``iterated_laplacian``, and ``biharmonic``)

Filters:
    SphericalFilter1D, SphericalFilter2D

Solvers:
    SphericalPoissonSolver, SphericalHelmholtzSolver
    SphericalVorticityInversionSolver     (∇²ψ = ζ, returns V = ẑ × ∇ψ)
    SphericalDivergenceInversionSolver    (∇²χ = δ, returns V = ∇χ)
    SphericalHelmholtzDecomposition       (V = ∇χ + ẑ × ∇ψ)
"""

from __future__ import annotations

from .filters import SphericalFilter1D, SphericalFilter2D
from .grid import SphericalGrid1D, SphericalGrid2D
from .harmonics import SphericalHarmonicTransform
from .operators import SphericalDerivative1D, SphericalDerivative2D
from .solvers import (
    SphericalDivergenceInversionSolver,
    SphericalHelmholtzDecomposition,
    SphericalHelmholtzSolver,
    SphericalPoissonSolver,
    SphericalVorticityInversionSolver,
)

__all__ = [
    "SphericalDerivative1D",
    "SphericalDerivative2D",
    "SphericalDivergenceInversionSolver",
    "SphericalFilter1D",
    "SphericalFilter2D",
    "SphericalGrid1D",
    "SphericalGrid2D",
    "SphericalHarmonicTransform",
    "SphericalHelmholtzDecomposition",
    "SphericalHelmholtzSolver",
    "SphericalPoissonSolver",
    "SphericalVorticityInversionSolver",
]
