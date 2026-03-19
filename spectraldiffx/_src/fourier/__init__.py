"""Fourier spectral methods: grids, operators, filters, transforms, and solvers."""

from __future__ import annotations

from spectraldiffx._src.fourier.capacitance import (
    CapacitanceSolver,
    build_capacitance_solver,
)
from spectraldiffx._src.fourier.eigenvalues import (
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
)
from spectraldiffx._src.fourier.filters import (
    SpectralFilter1D,
    SpectralFilter2D,
    SpectralFilter3D,
)
from spectraldiffx._src.fourier.grid import (
    FourierGrid1D,
    FourierGrid2D,
    FourierGrid3D,
)
from spectraldiffx._src.fourier.operators import (
    SpectralDerivative1D,
    SpectralDerivative2D,
    SpectralDerivative3D,
)
from spectraldiffx._src.fourier.solvers import (
    DirichletHelmholtzSolver2D,
    NeumannHelmholtzSolver2D,
    SpectralHelmholtzSolver1D,
    SpectralHelmholtzSolver2D,
    SpectralHelmholtzSolver3D,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
    solve_helmholtz_fft_1d,
    solve_poisson_dct,
    solve_poisson_dst,
    solve_poisson_fft,
    solve_poisson_fft_1d,
)
from spectraldiffx._src.fourier.transforms import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)

__all__ = [
    "CapacitanceSolver",
    "DirichletHelmholtzSolver2D",
    "FourierGrid1D",
    "FourierGrid2D",
    "FourierGrid3D",
    "NeumannHelmholtzSolver2D",
    "SpectralDerivative1D",
    "SpectralDerivative2D",
    "SpectralDerivative3D",
    "SpectralFilter1D",
    "SpectralFilter2D",
    "SpectralFilter3D",
    "SpectralHelmholtzSolver1D",
    "SpectralHelmholtzSolver2D",
    "SpectralHelmholtzSolver3D",
    "build_capacitance_solver",
    "dct",
    "dct2_eigenvalues",
    "dctn",
    "dst",
    "dst1_eigenvalues",
    "dstn",
    "fft_eigenvalues",
    "idct",
    "idctn",
    "idst",
    "idstn",
    "solve_helmholtz_dct",
    "solve_helmholtz_dst",
    "solve_helmholtz_fft",
    "solve_helmholtz_fft_1d",
    "solve_poisson_dct",
    "solve_poisson_dst",
    "solve_poisson_fft",
    "solve_poisson_fft_1d",
]
