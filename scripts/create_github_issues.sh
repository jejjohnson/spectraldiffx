#!/usr/bin/env bash
# =============================================================================
# Create GitHub Issues for the Documentation Revamp
# =============================================================================
#
# Run this script locally with the GitHub CLI (`gh`) authenticated.
# Usage:
#   bash scripts/create_github_issues.sh
#
# Prerequisites:
#   gh auth login   # authenticate once
#   gh auth status  # verify authentication
#
# This script creates the following issues with dependency links:
#   1. Convert scripts/ to jupytext and consolidate into notebooks/
#   2. Revamp README: remove useless links, add docs badge
#   3. Set up MkDocs documentation site
#   4. Theory page: Fourier pseudospectral methods
#   5. Theory page: Chebyshev pseudospectral methods
#   6. Theory page: Spherical harmonic methods
#   7. Complete documentation revamp (meta-issue, depends on all above)
#
set -euo pipefail

REPO="jejjohnson/spectraldiffx"

echo "Creating GitHub issues for documentation revamp..."
echo "Repository: $REPO"
echo ""

# -----------------------------------------------------------------------
# Issue 1: Consolidate notebooks and scripts into jupytext format
# -----------------------------------------------------------------------
ISSUE1=$(gh issue create \
  --repo "$REPO" \
  --title "chore: consolidate notebooks and scripts into jupytext format" \
  --label "documentation,enhancement" \
  --body "## Summary

Consolidate all example notebooks and scripts into a single \`notebooks/\` directory using [jupytext](https://jupytext.readthedocs.io/) percent format (\`.py\` files with \`# %%\` cell markers).

## Background

Currently:
- \`notebooks/\` contains 4 demo notebooks already in jupytext percent format
- \`scripts/\` contains 3 standalone simulation scripts as plain Python files

Both collections serve as examples/tutorials but live in separate places with different formats.

## Tasks

- [ ] Convert \`scripts/kdv.py\` → \`notebooks/kdv.py\` (jupytext percent format)
- [ ] Convert \`scripts/navier_stokes_2d.py\` → \`notebooks/navier_stokes_2d.py\` (jupytext percent format)
- [ ] Convert \`scripts/qg_model.py\` → \`notebooks/qg_model.py\` (jupytext percent format)
- [ ] Ensure all converted files have proper jupytext YAML frontmatter headers
- [ ] Ensure all converted files use consistent \`eqx.field(static=True)\` API (not deprecated \`eqx.static_field()\`)
- [ ] Keep original \`scripts/\` versions for CLI usage (they remain functional Python scripts)
- [ ] Verify jupytext can parse all notebooks: \`jupytext --check notebooks/*.py\`

## Acceptance Criteria

- All 7 files in \`notebooks/\` are valid jupytext percent-format notebooks
- \`jupytext --to ipynb notebooks/*.py\` converts them successfully
- Notebooks can be included in MkDocs via mkdocs-jupyter

## Notes

The jupytext format allows files to be:
1. Executed as standalone Python scripts (\`python notebooks/kdv.py\`)
2. Opened as Jupyter notebooks via jupytext
3. Rendered as documentation pages via mkdocs-jupyter
" \
  --json number --jq '.number')

echo "Created Issue #$ISSUE1: consolidate notebooks and scripts"

# -----------------------------------------------------------------------
# Issue 2: Revamp README
# -----------------------------------------------------------------------
ISSUE2=$(gh issue create \
  --repo "$REPO" \
  --title "docs: revamp README - remove useless links, add docs badge" \
  --label "documentation" \
  --body "## Summary

Clean up the README to remove outdated/useless external links and add a proper link to the documentation site.

## Tasks

- [ ] Remove the CodeFactor badge (third-party code quality service, not actively maintained)
- [ ] Fix the broken MIT license link (\`<a>MIT</a>\` has no href — fix to \`[MIT](LICENSE)\`)
- [ ] Add a documentation badge pointing to GitHub Pages: \`[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://jejjohnson.github.io/spectraldiffx)\`
- [ ] Add a **Documentation** section linking to the full docs site
- [ ] Ensure remaining badges (codecov, license, python version, status) are all functional

## Current State

The README header contains:
\`\`\`html
<a href=\"https://www.codefactor.io/repository/github/jejjohnson/spectraldiffx\">...</a>
\`\`\`
This service is not critical and the badge may be stale.

The footer contains:
\`\`\`html
<a>MIT</a> © J Emmanuel Johnson
\`\`\`
The \`<a>\` tag has no \`href\` — it renders as plain text without a link, which is confusing.

## Acceptance Criteria

- README has 4 meaningful badges: codecov, docs, license, python version, status
- All badge links resolve correctly
- MIT license links to the \`LICENSE\` file
- Documentation section points to GitHub Pages
" \
  --json number --jq '.number')

echo "Created Issue #$ISSUE2: revamp README"

# -----------------------------------------------------------------------
# Issue 3: Set up MkDocs documentation site
# -----------------------------------------------------------------------
ISSUE3=$(gh issue create \
  --repo "$REPO" \
  --title "docs: set up MkDocs documentation site with examples and API reference" \
  --label "documentation,enhancement" \
  --body "## Summary

Set up a full MkDocs-based documentation site for SpectralDiffX using the Material theme, with:
1. API reference generated from docstrings via mkdocstrings
2. Example notebooks rendered via mkdocs-jupyter
3. Theory pages with mathematical formulations

## Depends On

- #${ISSUE1} (consolidated jupytext notebooks are needed for mkdocs-jupyter)
- #${ISSUE2} (README should link to the docs site)

## Tasks

### Configuration
- [ ] Create \`mkdocs.yml\` with Material theme configuration
- [ ] Configure mkdocstrings plugin for Python API reference
- [ ] Configure mkdocs-jupyter plugin to render jupytext notebooks
- [ ] Configure pymdownx.arithmatex for LaTeX math rendering
- [ ] Set up GitHub Actions deployment to GitHub Pages (via \`pages.yml\`)

### Documentation Structure
- [ ] \`docs/index.md\` — Homepage with feature overview and quick example
- [ ] \`docs/installation.md\` — Installation guide (pip, uv dev setup)
- [ ] \`docs/quickstart.md\` — Quick start with 1D, 2D, and solver examples

### API Reference Pages
- [ ] \`docs/api/fourier/\` — Grids, Operators, Filters, Solvers
- [ ] \`docs/api/chebyshev/\` — Grids, Operators, Filters, Solvers
- [ ] \`docs/api/spherical/\` — Grids, Operators, Filters, Harmonics, Solvers

### Examples Section
- [ ] All 7 jupytext notebooks from \`notebooks/\` rendered as doc pages

### Theory Section
- [ ] Theory overview page linking to detail pages
- [ ] Fourier methods theory page (see #$(( ISSUE3 + 1 )))
- [ ] Chebyshev methods theory page (see #$(( ISSUE3 + 2 )))
- [ ] Spherical harmonic methods theory page (see #$(( ISSUE3 + 3 )))

## Tech Stack

\`\`\`
mkdocs                # core
mkdocs-material       # Material theme
mkdocstrings[python]  # API reference from docstrings
mkdocs-jupyter        # render jupytext/ipynb notebooks
pymdownx-arithmatex   # LaTeX math via MathJax
\`\`\`

## Acceptance Criteria

- \`uv run mkdocs build\` completes without errors
- \`uv run mkdocs serve\` shows a navigable, correct site
- All API classes are documented
- All 7 notebooks render correctly
- LaTeX math renders in theory pages
- Deployed to GitHub Pages via CI
" \
  --json number --jq '.number')

echo "Created Issue #$ISSUE3: set up MkDocs"

# -----------------------------------------------------------------------
# Issue 4: Theory page - Fourier pseudospectral methods
# -----------------------------------------------------------------------
ISSUE4=$(gh issue create \
  --repo "$REPO" \
  --title "docs: add theory page for Fourier pseudospectral methods" \
  --label "documentation,theory" \
  --body "## Summary

Write a comprehensive, pedagogically rigorous theory page covering Fourier pseudospectral methods as used in SpectralDiffX.

## Depends On

- #${ISSUE3} (MkDocs site must be set up first)

## Content Outline

### 1. Mathematical Setup
- Periodic function spaces and \$L^2([0, L))\$ inner products
- Continuous Fourier series: \$u(x) = \\sum_k \\hat{u}_k e^{ikx}\$
- Fourier coefficients and orthogonality

### 2. The Discrete Fourier Transform (DFT)
- Uniform grid discretization: \$x_j = j \\Delta x\$
- DFT definition and inverse DFT
- Fast Fourier Transform (FFT) and \$O(N \\log N)\$ complexity
- Wavenumber ordering and the Nyquist frequency

### 3. Spectral Differentiation
- Key identity: \$\\widehat{u'}(k) = ik \\hat{u}(k)\$
- Higher-order derivatives: \$\\widehat{u^{(n)}}(k) = (ik)^n \\hat{u}(k)\$
- Practical implementation steps (FFT → multiply by \$(ik)^n\$ → IFFT)
- Comparison with finite-difference accuracy

### 4. Aliasing and Dealiasing
- What is aliasing? Modes above Nyquist fold back
- The aliasing error in nonlinear terms (\$u \\cdot \\nabla u\$)
- **Orszag's 2/3 rule**: zero out top 1/3 of modes before nonlinear multiplication
- The padding approach (zero-padding in spectral space)
- Mathematical derivation of why 2/3 rule works

### 5. Spectral Filters
- Exponential filter: \$\\sigma(k) = \\exp(-\\alpha (k/k_{\\max})^{2p})\$
- Raised-cosine (Cesaro) filter
- Filter design trade-offs: sharpness vs. smoothness
- Applications: hyperviscosity as a spectral filter

### 6. Elliptic Solvers in Spectral Space
- Poisson equation: \$\\nabla^2 \\phi = f \\Rightarrow \\hat{\\phi}_k = \\hat{f}_k / (-k^2)\$
- Helmholtz equation: \$(\\nabla^2 - \\alpha)\\phi = f \\Rightarrow \\hat{\\phi}_k = \\hat{f}_k / (-k^2 - \\alpha)\$
- Extension to 2D and 3D with \$k^2 = k_x^2 + k_y^2 + k_z^2\$

### 7. Spectral Accuracy and Convergence
- Spectral convergence for smooth periodic functions: \$O(N^{-\\infty})\$
- Gibbs phenomenon for non-smooth functions
- Comparison: FD (\$O(\\Delta x^p)\$) vs. spectral (\$O(e^{-\\alpha N})\$)

## Style Requirements

- Use LaTeX math notation throughout (\$...\$ and \$\$...\$\$)
- Include worked examples with Python/JAX code snippets
- Include figures/diagrams where appropriate (Gibbs phenomenon, aliasing diagram)
- Pedagogically paced — explain WHY, not just how
- Reference key papers: Orszag (1971), Gottlieb & Orszag (1977), Trefethen (2000)

## Acceptance Criteria

- Page renders correctly in MkDocs with LaTeX math
- Covers all mathematical foundations needed to understand the SpectralDiffX Fourier API
- Includes at least 3 runnable code examples
- Cross-referenced with API docs and examples
" \
  --json number --jq '.number')

echo "Created Issue #$ISSUE4: Fourier theory page"

# -----------------------------------------------------------------------
# Issue 5: Theory page - Chebyshev pseudospectral methods
# -----------------------------------------------------------------------
ISSUE5=$(gh issue create \
  --repo "$REPO" \
  --title "docs: add theory page for Chebyshev pseudospectral methods" \
  --label "documentation,theory" \
  --body "## Summary

Write a comprehensive, pedagogically rigorous theory page covering Chebyshev pseudospectral methods as used in SpectralDiffX.

## Depends On

- #${ISSUE3} (MkDocs site must be set up first)

## Motivation

While Fourier methods excel at periodic problems, **Chebyshev methods** handle non-periodic domains with the same spectral accuracy. They are essential for problems with boundary conditions (e.g., channel flow, wall-bounded turbulence, boundary layers).

## Content Outline

### 1. Chebyshev Polynomials
- Definition: \$T_n(x) = \\cos(n \\arccos x)\$ for \$x \\in [-1, 1]\$
- Recurrence relation: \$T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)\$
- Orthogonality: \$\\int_{-1}^{1} T_m(x) T_n(x) / \\sqrt{1-x^2} \\, dx = \\pi c_n \\delta_{mn}/2\$
- The change of variables \$x = \\cos(\\theta)\$ connecting to Fourier

### 2. Chebyshev-Gauss-Lobatto (CGL) Nodes
- Definition: \$x_j = \\cos(j\\pi/N)\$ for \$j = 0, \\ldots, N\$
- Why these nodes? They minimize the Runge phenomenon
- Clustering at boundaries and the \$O(1/N^2)\$ grid spacing issue
- Comparison with equidistant grids

### 3. The Chebyshev Transform
- DCT-II connection: the Chebyshev expansion on CGL nodes
- Forward transform: \$a_k = (2/N) \\cdot DCT_k[u_j]\$ (with endpoint corrections)
- Inverse transform: \$u_j = \\sum_k a_k T_k(x_j)\$
- \$O(N \\log N)\$ implementation via FFT

### 4. Spectral Differentiation
- The differentiation matrix \$D_{ij}\$ for Chebyshev methods
- Chebyshev derivative via coefficient recurrence:
  \$c_k a_k' = 2 \\sum_{p=k+1, p+k \\text{ odd}}^N p a_p\$
- Iterative application for higher derivatives
- Comparison with the matrix formulation

### 5. Boundary Conditions
- Dirichlet BCs: modify the first/last rows of the differentiation matrix
- Neumann BCs: implementation via boundary bordering
- Tau method vs. boundary bordering

### 6. The Chebyshev Helmholtz Solver
- Solving \$(D^2 - \\alpha)u = f\$ with BCs
- Spectral space formulation using the recurrence
- Banded matrix structure and efficient solvers

### 7. Extension to 2D
- Tensor product grids: \$(x_i, y_j)\$ with \$x_i = \\cos(i\\pi/N_x)\$, etc.
- Mixed Fourier-Chebyshev for channel flow
- Kronecker product structure of 2D differentiation matrices

### 8. Comparison with Fourier Methods
| Feature | Fourier | Chebyshev |
|---------|---------|-----------|
| Domain | Periodic | Non-periodic |
| BCs | Automatic (periodic) | Explicit |
| Grid clustering | Uniform | Clustered at boundaries |
| CFL restriction | Mild | Severe (\$\\Delta t \\sim O(N^{-2})\$) |

## Style Requirements

- Use LaTeX math notation throughout
- Include worked examples with Python/JAX code
- Visual illustrations of CGL node clustering
- Reference key papers: Canuto et al. (1988), Trefethen (2000), Boyd (2001)

## Acceptance Criteria

- Page renders correctly in MkDocs with LaTeX math
- Covers all mathematical foundations for the SpectralDiffX Chebyshev API
- Includes runnable code examples using \`ChebyshevGrid1D\` and \`ChebyshevDerivative1D\`
" \
  --json number --jq '.number')

echo "Created Issue #$ISSUE5: Chebyshev theory page"

# -----------------------------------------------------------------------
# Issue 6: Theory page - Spherical harmonic methods
# -----------------------------------------------------------------------
ISSUE6=$(gh issue create \
  --repo "$REPO" \
  --title "docs: add theory page for spherical harmonic methods" \
  --label "documentation,theory" \
  --body "## Summary

Write a comprehensive, pedagogically rigorous theory page covering spherical harmonic (SH) pseudospectral methods as used in SpectralDiffX for global geophysical applications.

## Depends On

- #${ISSUE3} (MkDocs site must be set up first)

## Motivation

Spherical harmonics are the natural basis functions for problems on the sphere — global weather models, ocean circulation, and geophysical fluid dynamics all use them. Unlike Fourier or Chebyshev, SH methods operate in two dimensions simultaneously (latitude and longitude) and couple them through the spherical geometry.

## Content Outline

### 1. Spherical Coordinates and the Sphere
- Coordinate system: colatitude \$\\theta \\in [0, \\pi]\$, longitude \$\\phi \\in [0, 2\\pi)\$
- Surface element: \$dS = \\sin\\theta \\, d\\theta \\, d\\phi\$
- The Laplace-Beltrami operator:
  \$\\Delta_S u = \\frac{1}{\\sin\\theta} \\partial_\\theta(\\sin\\theta \\, \\partial_\\theta u) + \\frac{1}{\\sin^2\\theta} \\partial_{\\phi\\phi} u\$

### 2. Spherical Harmonics
- Definition: \$Y_\\ell^m(\\theta, \\phi) = P_\\ell^m(\\cos\\theta) e^{im\\phi}\$
- Associated Legendre polynomials \$P_\\ell^m\$ — definition and recurrence
- Normalization convention: real/complex, Condon-Shortley phase
- Key property: \$\\Delta_S Y_\\ell^m = -\\ell(\\ell+1) Y_\\ell^m\$
- Orthogonality: \$\\int_S Y_\\ell^m Y_{\\ell'}^{m'*} \\, dS = \\delta_{\\ell\\ell'} \\delta_{mm'}\$

### 3. The Spherical Harmonic Transform (SHT)
- Decomposition: \$u(\\theta, \\phi) = \\sum_{\\ell=0}^L \\sum_{m=-\\ell}^{\\ell} \\hat{u}_\\ell^m Y_\\ell^m(\\theta, \\phi)\$
- Truncation at degree \$L\$ (triangular truncation)
- Forward SHT: FFT in longitude + Gauss-Legendre quadrature in latitude
- Gauss-Legendre nodes and weights: optimal for polynomial integration
- Implementation: \$O(L^2 N)\$ naive, \$O(L^2 \\log L)\$ with fast algorithms

### 4. Gauss-Legendre Quadrature
- Why not equidistant latitude grids? Aliasing issues near poles
- GL quadrature points \$\\theta_j\$ and weights \$w_j\$
- Exact for polynomials up to degree \$2N-1\$
- Connection to Legendre polynomials: \$P_N(\\cos\\theta_j) = 0\$

### 5. Spectral Differentiation on the Sphere
- Zonal derivative (longitudinal): \$\\partial_\\phi u \\leftrightarrow im \\hat{u}_\\ell^m\$
- Meridional derivative (latitudinal): requires recurrence involving \$P_\\ell^m\$ and \$P_\\ell^{m+1}\$
- Gradient, divergence, curl operators in spherical geometry
- The \$1/\\sin\\theta\$ singularity at the poles and how to handle it

### 6. Helmholtz and Poisson Equations on the Sphere
- Poisson: \$\\Delta_S \\phi = f \\Rightarrow \\hat{\\phi}_\\ell^m = \\hat{f}_\\ell^m / (-\\ell(\\ell+1))\$
- Helmholtz: \$(\\Delta_S - \\alpha)\\phi = f \\Rightarrow \\hat{\\phi}_\\ell^m = \\hat{f}_\\ell^m / (-\\ell(\\ell+1) - \\alpha)\$
- Zero mode issue: \$\\ell=0\$ mode must be handled separately

### 7. Applications in Geophysical Modeling
- Barotropic vorticity equation on the sphere
- Quasi-geostrophic models
- Rossby waves and their spherical harmonic structure
- Spectral general circulation models (GCMs)

### 8. Numerical Considerations
- Triangular vs. rhomboidal truncation
- Robert filter for time marching stability
- Transform resolution: \$N_{grid} \\geq 3L/2\$ for dealiasing
- Cost comparison: spectral (\$O(L^3)\$) vs. grid-point (\$O(N^2 \\log N)\$)

## Style Requirements

- Use LaTeX math notation throughout
- Include worked examples using \`SphericalGrid2D\` and \`SphericalDerivative2D\`
- Visual illustrations of spherical harmonic patterns (e.g., \$Y_2^1\$)
- Reference key papers: Orszag (1970), Eliasen et al. (1970), Temperton (1991), Wedi et al. (2013)

## Acceptance Criteria

- Page renders correctly in MkDocs with LaTeX math
- Covers all mathematical foundations for the SpectralDiffX Spherical API
- Includes runnable code examples
- Explains the Gauss-Legendre quadrature in detail
" \
  --json number --jq '.number')

echo "Created Issue #$ISSUE6: Spherical harmonics theory page"

# -----------------------------------------------------------------------
# Issue 7: Meta-issue - Complete documentation revamp
# -----------------------------------------------------------------------
ISSUE7=$(gh issue create \
  --repo "$REPO" \
  --title "docs: complete documentation revamp (meta-issue)" \
  --label "documentation,enhancement" \
  --body "## Summary

This is the meta-issue tracking the complete documentation revamp of SpectralDiffX. It tracks progress across all sub-tasks.

## Sub-Issues (must all be completed)

- [ ] #${ISSUE1} — Consolidate notebooks and scripts into jupytext format
- [ ] #${ISSUE2} — Revamp README (remove useless links, add docs badge)
- [ ] #${ISSUE3} — Set up MkDocs documentation site
- [ ] #${ISSUE4} — Theory page: Fourier pseudospectral methods
- [ ] #${ISSUE5} — Theory page: Chebyshev pseudospectral methods
- [ ] #${ISSUE6} — Theory page: Spherical harmonic methods

## Dependency Graph

\`\`\`
#${ISSUE1} (jupytext)  ─┐
                         ├─► #${ISSUE3} (MkDocs setup) ─► #${ISSUE7} (meta)
#${ISSUE2} (README)   ─┘                │
                                         ├─► #${ISSUE4} (Fourier theory)
                                         ├─► #${ISSUE5} (Chebyshev theory)
                                         └─► #${ISSUE6} (Spherical theory)
\`\`\`

## Definition of Done

The documentation revamp is complete when:

1. ✅ All 7 jupytext notebooks are in \`notebooks/\` and render correctly
2. ✅ README has meaningful badges and a link to docs
3. ✅ \`uv run mkdocs build\` succeeds without errors or warnings
4. ✅ \`uv run mkdocs serve\` shows a complete, navigable docs site with:
   - Home page with overview and quick start
   - Installation guide
   - 3 theory pages with LaTeX-rendered math
   - 7 rendered example notebooks
   - Complete API reference for all public classes
5. ✅ GitHub Pages is automatically deployed via CI
6. ✅ README links to the live docs site

## Related PRs

See: https://github.com/${REPO}/pulls
" \
  --json number --jq '.number')

echo "Created Issue #$ISSUE7: meta-issue"
echo ""
echo "All issues created successfully!"
echo ""
echo "Issue summary:"
echo "  #$ISSUE1 — Consolidate notebooks and scripts into jupytext format"
echo "  #$ISSUE2 — Revamp README"
echo "  #$ISSUE3 — Set up MkDocs documentation site (depends on #$ISSUE1, #$ISSUE2)"
echo "  #$ISSUE4 — Theory: Fourier methods (depends on #$ISSUE3)"
echo "  #$ISSUE5 — Theory: Chebyshev methods (depends on #$ISSUE3)"
echo "  #$ISSUE6 — Theory: Spherical harmonics (depends on #$ISSUE3)"
echo "  #$ISSUE7 — Meta-issue (depends on all above)"
