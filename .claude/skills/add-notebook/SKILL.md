---
name: add-notebook
description: Add or update an example notebook in spectraldiffx's docs — a jupytext percent-format .py file in notebooks/ (never .ipynb), figures saved under docs/images/<notebook>/ and linked from markdown cells because mkdocs-jupyter renders the notebooks unexecuted, an "Examples" nav entry in mkdocs.yml, and a strict MkDocs build. Use when asked to write, extend, re-run or fix a tutorial, demo or example notebook.
---

# Add or update a notebook

Notebooks here follow a different rule from the other GeoML repos: the
committed source is the jupytext **`.py`** file, and `.ipynb` files are
forbidden (`.gitignore`, the `forbid-ipynb` pre-commit hook,
`make nb-check`). mkdocs-jupyter renders the `.py` **without executing it**
(`execute: false` in `mkdocs.yml`), so figures cannot come from cell
outputs: each one is saved to a committed PNG and linked from a markdown
cell.

## 1. Plan it

- One question per notebook; read the existing ones in `notebooks/` and the
  "Examples" nav in `mkdocs.yml` so you extend rather than duplicate (1-D /
  2-D differentiation, eigenfunctions, solver comparison, mixed and
  inhomogeneous BCs, PS vs FD2 eigenvalues, capacitance, QG, KdV,
  Navier–Stokes).
- Use the public API the way a user would (`from spectraldiffx import …` or
  `import spectraldiffx as sdx`); no `_src` imports.

## 2. Write `notebooks/<name>.py`

- Jupytext percent format with the same header as the others (copy it from
  `notebooks/demo_1d.py`), `# %% [markdown]` and `# %%` cells; first
  markdown cell: a `#` title and what the notebook shows.
- Setup cell, as in the existing notebooks:

  ```python
  from pathlib import Path

  import jax
  import matplotlib
  import matplotlib.pyplot as plt

  matplotlib.use("Agg")
  jax.config.update("jax_enable_x64", True)

  IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "<name>"
  IMG_DIR.mkdir(parents=True, exist_ok=True)
  ```

- Each figure: `fig.savefig(IMG_DIR / "<figure>.png", dpi=150,
  bbox_inches="tight")`, `plt.show()`, then a markdown cell
  `# ![<alt text>](../../images/<name>/<figure>.png)` (the page is served
  at `notebooks/<name>/`, so the image is two levels up).
- Math in markdown cells as LaTeX (`$…$`, `$$…$$`; MathJax is configured in
  `mkdocs.yml`).
- Deterministic: no randomness, or a seeded `np.random.default_rng(seed)`.

## 3. Run it and commit the figures

```bash
uv sync --all-extras                      # matplotlib / seaborn live in the docs, exp and examples extras
uv run python notebooks/<name>.py         # writes docs/images/<name>/*.png
git add notebooks/<name>.py docs/images/<name>/
```

`.gitignore` ignores `*.png` except under `docs/images/`, so check
`git status` shows the new figures. Keep them small (`dpi=150`, a few
figures; the `check-added-large-files` pre-commit hook rejects big files).

## 4. Wire it in and verify

- Add `- <Title>: notebooks/<name>.py` under "Examples" in `mkdocs.yml`.
- `make nb-check` (no `.ipynb` left behind; delete any that jupytext or an
  editor created with `make nb-clean`).
- ruff excludes `notebooks/`, so lint does not cover it: keep code cells
  readable at 88 characters.
- `make docs` (`mkdocs build --strict`). Locally, mkdocs-jupyter asks
  jupytext to read every `.md` page, which shells out to pandoc when pandoc
  is installed and can take minutes on the API pages; if the build stalls
  in `pandoc`, run it with pandoc off the `PATH`.
- Open the built page (`make docs-serve`) and check every image link
  resolves.
