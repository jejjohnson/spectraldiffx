.PHONY: help install_mamba install_macos install_linux update_macos update_linux
.DEFAULT_GOAL = help

# ANSI Color Codes for pretty terminal output
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m



PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = ocn-tools
TESTS = ${PKGROOT}/tests
ENVS = ${PKGROOT}/environments

help:	## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)



##@ Installation
install_mamba: ## Install mamba in base environment
	conda install mamba -n base

install_macos: ## Install conda env in MACOS
	mamba env create -f ${ENVS}/macos.yaml

install_linux: ## Install conda env in Linux
	mamba env create -f ${ENVS}/linux.yaml

update_macos: ## Update conda env in MACOS
	mamba env update -f ${ENVS}/macos.yaml

update_linux: ## Update conda env in linux
	mamba env update -f ${ENVS}/linux.yaml

install_precommit: ## Install precommit tools
	mamba install pre-commit -c conda-forge
	pre-commit install --all-files

##@ Formatting
.PHONY: format
format: ## 🎨 Run ruff formatter
	@printf "$(YELLOW)>>> Formatting code with ruff...$(RESET)\n"
	@uv run ruff format spectraldiffx
	@uv run ruff check --fix spectraldiffx
	@printf "$(GREEN)>>> Codebase formatted successfully.$(RESET)\n"

.PHONY: lint
lint: ## 🔍 Run ruff check and mypy
	@printf "$(YELLOW)>>> Executing static analysis and type checking...$(RESET)\n"
	@uv run ruff check spectraldiffx
	@uv run mypy spectraldiffx
	@printf "$(GREEN)>>> Linting checks passed.$(RESET)\n"

.PHONY: pre-commit
pre-commit: ## 🛠️  Run all pre-commit hooks
	@printf "$(YELLOW)>>> Running full pre-commit validation suite...$(RESET)\n"
	@uv run pre-commit run --all-files || (printf "$(RED)>>> ⚠️ Hooks failed.")
	@printf "$(GREEN)>>> ✅ Pre-commit checks finalized.$(RESET)\n"

##@ Testing
test:  ## Test code using pytest.
	@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
	pytest -v ${PKGROOT}/ ${TESTS}
	@printf "\033[1;34mPyTest passes!\033[0m\n\n"

.PHONY: install
install: ## 📦 Install all project dependencies
	@printf "$(YELLOW)>>> Initiating environment synchronization and dependency installation...$(RESET)\n"
	@uv sync --all-extras
	@uv run pre-commit install
	@printf "$(GREEN)>>> ✅ Environment is ready and pre-commit hooks are active.$(RESET)\n"

.PHONY: uv-sync
uv-sync: ## 🔄 Update lock file and sync dependencies using uv
	@printf "$(YELLOW)>>> Updating and syncing dependencies with uv...$(RESET)\n"
	@uv lock --upgrade
	@uv sync --all-extras
	@printf "$(GREEN)>>> ✅ uv environment synchronized.$(RESET)\n"

.PHONY: uv-test
uv-test: ## 🧪 Run pytest with coverage using uv
	@printf "$(YELLOW)>>> Launching test suite with verbosity...$(RESET)\n"
	@uv run pytest tests -v
	@printf "$(GREEN)>>> ✅ All tests passed.$(RESET)\n"
