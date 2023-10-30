# Spectral Difference Tools in JAX (In Progress)

[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/spectraldiffx/badge)](https://www.codefactor.io/repository/github/jejjohnson/spectraldiffx)
[![codecov](https://codecov.io/gh/jejjohnson/spectraldiffx/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/spectraldiffx)

> This package has some tools for pseudospectral methods in JAX.



---
## Installation

We can install it directly through pip

```bash
pip install git+https://github.com/jejjohnson/spectraldiffx
```

We also use poetry for the development environment.

```bash
git clone https://github.com/jejjohnson/spectraldiffx.git
cd spectraldiffx
conda create -n spectraldiffx python=3.11 poetry
poetry install
```



---
## References

**Software**

* [kernex](https://github.com/ASEM000/kernex) - differentiable stencils
* [FiniteDiffX](https://github.com/ASEM000/finitediffX) - finite difference tools in JAX
* [SpectralDiffX](https://github.com/jejjohnson/finitevolX) - Finite Volume Tool in Python


**Algorithms**

* [Thiry et al, 2023](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1715/) | [MQGeometry 1.0](https://github.com/louity/MQGeometry) - the WENO reconstructions applied to the multilayer Quasi-Geostrophic equations, Arakawa Grid masks
* [Roullet & Gaillard, 20