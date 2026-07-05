# Makefile for deform_conv2d_mps
# Run `make help` to list targets.

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

# Force the native MPS path instead of the torchvision fallback.
NATIVE_ENV = DCN_MPS_FORCE_NATIVE=1

# Conda envs often link two OpenMP runtimes (LLVM libomp from torch + Intel
# libiomp5 from numpy/MKL); importing both aborts with "OMP Error #15".
# Allow the duplicate so build/test/run work. The clean fix is to dedupe the
# OpenMP packages in the env (e.g. `conda install nomkl`, or ensure a single
# openmp/llvm-openmp), after which this can be removed.
export KMP_DUPLICATE_LIB_OK := TRUE

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Setup / install
# ---------------------------------------------------------------------------
# --no-build-isolation is REQUIRED: the extension bakes in the compile-time
# value of c10::DispatchKey::MPS, which must match the torch you actually run.
# Build isolation would pull a different torch into an overlay and the enum
# values can differ (kernel ends up registered under the wrong key, e.g. IPU).
.PHONY: install
install:  ## Editable install with test extras (builds against the installed torch)
	$(PIP) install -e ".[test]" --no-build-isolation

.PHONY: install-deps
install-deps:  ## Install torch + torchvision + build deps (setuptools, wheel, ninja)
	$(PIP) install torch torchvision setuptools wheel ninja

# ---------------------------------------------------------------------------
# Build / compile the native extension
# ---------------------------------------------------------------------------
.PHONY: build
build:  ## Compile the Objective-C++/Metal extension in place
	$(PYTHON) setup.py build_ext --inplace

.PHONY: wheel
wheel:  ## Build a distributable wheel
	$(PYTHON) -m build --wheel

.PHONY: rebuild
rebuild: clean build  ## Clean then build

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Run the full test suite (uses fallback where native is incomplete)
	$(PYTEST) -q

.PHONY: test-native
test-native:  ## Run tests forcing the native MPS kernel path
	$(NATIVE_ENV) $(PYTEST) -q

.PHONY: test-forward
test-forward:  ## Forward correctness vs torchvision reference
	$(PYTEST) -q tests/test_forward.py

.PHONY: test-forward-native
test-forward-native:  ## Forward tests forcing the native MPS kernel (Phase 2)
	$(NATIVE_ENV) $(PYTEST) -q tests/test_forward.py

.PHONY: test-backward
test-backward:  ## Backward + gradcheck
	$(PYTEST) -q tests/test_backward.py

.PHONY: test-module
test-module:  ## DeformConv2d module / state_dict parity
	$(PYTEST) -q tests/test_module.py

# ---------------------------------------------------------------------------
# Run / benchmark / examples
# ---------------------------------------------------------------------------
.PHONY: bench
bench:  ## MPS vs CPU-fallback timings
	$(PYTHON) benchmarks/bench.py

.PHONY: examples
examples:  ## Run the example scripts
	$(PYTHON) example/example01_deform_conv2d_cpu_mps.py
	$(PYTHON) example/example02_deform_conv2d_gradcheck.py

.PHONY: diag
diag:  ## Phase-1 diagnostics: im2col isolation ladder (tests/diag_im2col.py)
	$(PYTHON) tests/diag_im2col.py

.PHONY: diag-backward
diag-backward:  ## Phase-3 diagnostics: backward/col2im isolation ladder (tests/diag_col2im.py)
	$(PYTHON) tests/diag_col2im.py

.PHONY: smoke
smoke:  ## Quick import + version sanity check
	$(PYTHON) -c "import torch, deform_conv2d_mps as d; \
print('deform_conv2d_mps', d.__version__, '| torch', torch.__version__, \
'| mps', torch.backends.mps.is_available())"

# ---------------------------------------------------------------------------
# Lint / format (optional; no-ops if tools absent)
# ---------------------------------------------------------------------------
.PHONY: lint
lint:  ## Lint with ruff if installed
	@command -v ruff >/dev/null 2>&1 && ruff check src tests benchmarks || \
		echo "ruff not installed; skipping"

.PHONY: format
format:  ## Format with ruff if installed
	@command -v ruff >/dev/null 2>&1 && ruff format src tests benchmarks || \
		echo "ruff not installed; skipping"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean:  ## Remove build artifacts and caches
	rm -rf build dist *.egg-info src/*.egg-info
	rm -f src/deform_conv2d_mps/*.so src/deform_conv2d_mps/_C/*.metallib
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
