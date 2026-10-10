"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/spectraldiffx/skills/spectral-methods-with-spectraldiffx/SKILL.md``
is what agents in downstream projects read before writing pseudospectral
code on spectraldiffx, so a stale name or a broken example there teaches
them the wrong API. These tests run its worked example and check that every
``sdx.X`` / ``spectraldiffx.X`` / ``gaussx.X`` it and the plugin's reviewer
name is a current public name.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "spectraldiffx"
SKILL = PLUGIN / "skills" / "spectral-methods-with-spectraldiffx" / "SKILL.md"
AGENT = PLUGIN / "agents" / "spectraldiffx-reuse-reviewer.md"
if not SKILL.is_file():
    # A built distribution ships tests/ without the repository root.
    pytest.skip("the plugin is not present", allow_module_level=True)

_ALIASES = {"sdx": "spectraldiffx"}
# A name after a dot, not inside a URL or path (github.com/.../spectraldiffx.git).
_NAME = re.compile(r"(?<![\w./@])(sdx|spectraldiffx|gaussx)\.([A-Za-z]\w*)")
_BLOCKS = re.compile(r"```python\n(.*?)```", re.S)


@pytest.mark.slow
def test_worked_example_runs_and_its_claims_hold():
    (block,) = [
        b for b in _BLOCKS.findall(SKILL.read_text()) if "build_capacitance_solver" in b
    ]
    ns: dict = {}
    exec(block, ns)
    # Steps 2-4 and 8 act on single resolved modes, so they are exact up to
    # round-off; the prose says "around 1e-14". Measured under x64: 3.5e-14
    # (gradient), 5e-15 (self-advection), 8e-16 (Poisson), 3.5e-14
    # (Chebyshev, 33 nodes); 1e-12 leaves a wide margin and still fails for
    # any wrong wavenumber, sign or transform pairing (errors of order 1).
    for name in ("grad_err", "self_advection", "poisson_err", "cheb_err"):
        assert float(ns[name]) < 1e-12, name
    # The channel solve inverts the 5-point Laplacian exactly, so against the
    # continuous ψ its error is the ratio of the continuous to the FD2
    # eigenvalue of the one mode present, minus one (the prose: "about 8e-4").
    kx, ky, dx, dy = 2 * math.pi, math.pi, ns["dx"], ns["dy"]
    lam_fd2 = (
        4 / dx**2 * math.sin(kx * dx / 2) ** 2 + 4 / dy**2 * math.sin(ky * dy / 2) ** 2
    )
    predicted = (kx**2 + ky**2) / lam_fd2 - 1  # max |ψ| = 1
    assert float(ns["channel_err"]) == pytest.approx(predicted, rel=1e-6)
    assert 7e-4 < predicted < 9e-4
    # ψ̂ = f̂ / (Λ − λ) with Λ < 0: a larger λ shrinks every mode.
    assert float(ns["d_energy"]) < 0
    # The capacitance solve is exact for the masked 5-point system: the
    # residual is round-off relative to |ψ| ≈ 68 (measured 1.1e-13).
    assert float(ns["basin_residual"]) < 1e-10
    # ψ = 0 on land and on the coast (every non-interior cell).
    psi_b, interior = ns["psi_b"], ns["interior"]
    assert float(abs(psi_b[~interior]).max()) == 0.0


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path):
    stale = []
    for prefix, attr in set(_NAME.findall(path.read_text())):
        module_name = _ALIASES.get(prefix, prefix)
        module = importlib.import_module(module_name)
        if attr in getattr(module, "__all__", dir(module)):
            continue
        try:  # a submodule path
            importlib.import_module(f"{module_name}.{attr}")
        except ImportError:
            stale.append(f"{module_name}.{attr}")
    assert not stale, f"{path.name} names objects that do not exist: {sorted(stale)}"
