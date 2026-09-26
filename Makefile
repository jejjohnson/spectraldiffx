# =============================================================================
# spectraldiffx Makefile
# =============================================================================
#
# Every target runs through uv. The quality targets run exactly what CI runs:
#   make test       # uv run pytest tests -n auto
#   make lint       # uv run ruff check .          (no auto-fix: can fail)
#   make format     # uv run ruff format . && uv run ruff check --fix .
#   make typecheck  # uv run ty check spectraldiffx
#
# =============================================================================

.DEFAULT_GOAL := help

PKGROOT = spectraldiffx
TESTS = tests
NOTEBOOKS_DIR = notebooks

# ANSI Color Codes for pretty terminal output
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

.PHONY: help install sync lint format format-check typecheck precommit test test-fast test-slow test-cov \
        docs docs-serve

help:	## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(BLUE)<target>$(RESET)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-18s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
install: ## Install all extras and the pre-commit hooks
	@printf "$(YELLOW)>>> Installing all dependencies...$(RESET)\n"
	uv sync --all-extras
	uv run pre-commit install
	@printf "$(GREEN)>>> Environment is ready and pre-commit hooks are active.$(RESET)\n"

sync: ## Upgrade the lock file and re-sync all extras
	uv lock --upgrade
	uv sync --all-extras

##@ Quality (the same commands CI runs)
lint: ## Lint the whole repo with ruff (no auto-fix)
	uv run ruff check .

format: ## Format and auto-fix the whole repo with ruff
	uv run ruff format .
	uv run ruff check --fix .

format-check: ## Check formatting without rewriting files
	uv run ruff format --check .

typecheck: ## Type check the package with ty
	uv run ty check $(PKGROOT)

precommit: ## Run every pre-commit hook on all files
	uv run pre-commit run --all-files

##@ Testing
test: ## Run the whole test suite in parallel
	uv run pytest $(TESTS) -n auto

test-fast: ## Run the fast tests only (what PR CI runs)
	uv run pytest $(TESTS) -n auto -m "not slow and not integration"

test-slow: ## Run only the slow and integration tests
	uv run pytest $(TESTS) -n auto -m "slow or integration"

test-cov: ## Run the test suite with coverage (report in reports/)
	uv run pytest $(TESTS) -n auto --cov --cov-report=term --cov-report=xml

##@ Docs
docs: ## Build the docs site (strict)
	uv run --extra docs mkdocs build --strict

docs-serve: ## Serve the docs locally with live reload
	uv run --extra docs mkdocs serve

##@ Notebooks (Jupytext)
.PHONY: nb-to-py
nb-to-py: ## Convert all .ipynb notebooks to .py (percent format)
	@printf "$(YELLOW)>>> Converting notebooks to Python scripts...$(RESET)\n"
	@uv run jupytext --to py:percent $(NOTEBOOKS_DIR)/*.ipynb 2>/dev/null || printf "$(YELLOW)>>> No .ipynb files found.$(RESET)\n"
	@printf "$(GREEN)>>> Conversion complete.$(RESET)\n"

.PHONY: nb-to-ipynb
nb-to-ipynb: ## Convert all .py notebooks to .ipynb
	@printf "$(YELLOW)>>> Converting Python scripts to notebooks...$(RESET)\n"
	@uv run jupytext --to notebook $(NOTEBOOKS_DIR)/*.py
	@printf "$(GREEN)>>> Conversion complete.$(RESET)\n"

.PHONY: nb-sync
nb-sync: ## Sync .py and .ipynb notebooks (update whichever is older)
	@printf "$(YELLOW)>>> Syncing notebooks...$(RESET)\n"
	@uv run jupytext --sync $(NOTEBOOKS_DIR)/*.py
	@printf "$(GREEN)>>> Notebooks synced.$(RESET)\n"

.PHONY: nb-pair
nb-pair: ## Pair .py files with .ipynb (creates paired notebooks)
	@printf "$(YELLOW)>>> Pairing Python scripts with notebooks...$(RESET)\n"
	@uv run jupytext --set-formats py:percent,ipynb $(NOTEBOOKS_DIR)/*.py
	@printf "$(GREEN)>>> Pairing complete.$(RESET)\n"

.PHONY: nb-check
nb-check: ## Check that no .ipynb files exist (for CI)
	@printf "$(YELLOW)>>> Checking for .ipynb files...$(RESET)\n"
	@if ls $(NOTEBOOKS_DIR)/*.ipynb 1> /dev/null 2>&1; then \
		printf "$(RED)>>> ERROR: .ipynb files found in $(NOTEBOOKS_DIR)/. Please convert to .py format.$(RESET)\n"; \
		ls $(NOTEBOOKS_DIR)/*.ipynb; \
		exit 1; \
	else \
		printf "$(GREEN)>>> No .ipynb files found. All good!$(RESET)\n"; \
	fi

.PHONY: nb-clean
nb-clean: ## Remove all .ipynb files from notebooks directory
	@printf "$(YELLOW)>>> Removing .ipynb files...$(RESET)\n"
	@rm -f $(NOTEBOOKS_DIR)/*.ipynb
	@rm -rf $(NOTEBOOKS_DIR)/.ipynb_checkpoints
	@printf "$(GREEN)>>> Cleanup complete.$(RESET)\n"

.PHONY: nb-execute
nb-execute: ## Execute all notebooks (converts to ipynb, runs, then cleans)
	@printf "$(YELLOW)>>> Executing notebooks...$(RESET)\n"
	@uv run jupytext --to notebook --execute $(NOTEBOOKS_DIR)/*.py
	@printf "$(GREEN)>>> Notebooks executed.$(RESET)\n"


##@ Examples
.PHONY: install-examples
install-examples: ## Install optional dependencies for running examples
	@printf "$(YELLOW)>>> Installing dependencies for examples...$(RESET)\n"
	@uv sync --extra examples
	@printf "$(GREEN)>>> Example dependencies installed.$(RESET)\n"

.PHONY: run-burgers
run-burgers: ## Run the 1D Burgers' equation example
	@printf "$(YELLOW)>>> Running Burgers' equation example...$(RESET)\n"
	@uv run python scripts/burgers.py --nx 256 --viscosity 1e-3 --t-end 2.0
	@printf "$(GREEN)>>> Example finished.$(RESET)\n"

.PHONY: run-kdv
run-kdv: ## Run the KdV (Korteweg-de Vries) equation example
	@printf "$(YELLOW)>>> Running KdV equation example...$(RESET)\n"
	@uv run python scripts/kdv.py --nx 512 --length 100 --t-end 20.0
	@printf "$(GREEN)>>> Example finished.$(RESET)\n"

.PHONY: run-navier-stokes
run-navier-stokes: ## Run the 2D Navier-Stokes (vorticity) example
	@printf "$(YELLOW)>>> Running 2D Navier-Stokes example...$(RESET)\n"
	@uv run python scripts/navier_stokes_2d.py --nx 256 --ny 256 --viscosity 1e-6 --t-end 50.0
	@printf "$(GREEN)>>> Example finished.$(RESET)\n"

.PHONY: run-qg
run-qg: ## Run the Quasigeostrophic (QG) model example
	@printf "$(YELLOW)>>> Running QG model example...$(RESET)\n"
	@uv run python scripts/qg_model.py --nx 128 --ny 128 --beta 10.0 --rossby-radius 0.1 --t-end 50.0
	@printf "$(GREEN)>>> Example finished.$(RESET)\n"

.PHONY: run-all-examples
run-all-examples: run-burgers run-kdv run-navier-stokes run-qg ## Run all example scripts
