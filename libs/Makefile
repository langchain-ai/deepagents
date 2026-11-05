MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LIB_PACKAGE_DIRS := $(patsubst %/,%,$(dir $(patsubst $(MAKEFILE_DIR)%,%,$(wildcard $(MAKEFILE_DIR)*/Makefile $(MAKEFILE_DIR)partners/*/Makefile))))
EXAMPLE_PACKAGE_DIRS := $(patsubst %/,%,$(patsubst $(MAKEFILE_DIR)%,%,$(dir $(wildcard $(MAKEFILE_DIR)../examples/*/pyproject.toml))))
PACKAGE_DIRS := $(sort $(LIB_PACKAGE_DIRS) $(EXAMPLE_PACKAGE_DIRS))

# Map package dirs to their required Python version
# acp requires 3.14, everything else uses 3.12
python_version = $(if $(filter acp,$1),3.14,3.12)

.PHONY: help lock no-cache lock-bump lock-check hygiene lint format bench-all

LOCK_ARGS := $(if $(filter no-cache,$(MAKECMDGOALS)),--no-cache,)

BENCH_PACKAGES := deepagents code

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

lock: ## Update all lockfiles (append 'no-cache' to bypass uv's cache)
	@set -e; \
	$(foreach pkg,$(PACKAGE_DIRS), \
		echo "🔒 Locking $(pkg)"; \
		uv lock --directory '$(pkg)' --python '$(call python_version,$(pkg))' $(LOCK_ARGS); \
	)
	@echo "✅ All lockfiles updated!"

no-cache:
	@:

lock-bump: ## Bump a dependency across all lockfiles (usage: make lock-bump DEP=requests)
	@if [ -z "$(DEP)" ]; then \
		echo "Usage: make lock-bump DEP=<package>"; \
		exit 1; \
	fi
	@set -e; \
	$(foreach pkg,$(PACKAGE_DIRS), \
		echo "⬆️  Bumping $(DEP) in $(pkg)"; \
		uv lock --directory '$(pkg)' --python '$(call python_version,$(pkg))' -P '$(DEP)'; \
	)
	@echo "✅ Bumped $(DEP) across all lockfiles!"

lock-check: ## Check all lockfiles are up-to-date
	@set -e; \
	$(foreach pkg,$(PACKAGE_DIRS), \
		echo "🔍 Checking $(pkg)"; \
		uv lock --check --directory '$(pkg)' --python '$(call python_version,$(pkg))'; \
	)
	@echo "✅ All lockfiles are up-to-date!"

# File-hygiene pre-commit hooks, run on files changed since HYGIENE_FROM.
# CI's Text Hygiene job calls this target with the PR's base and head, so this
# list is the single source of truth (explicit, so a new `language: system`
# hook is not pulled in by accident). Changed files only: `main` still has
# older violations, some intentional (test fixtures, vendored data).
HYGIENE_HOOKS := check-yaml check-toml end-of-file-fixer trailing-whitespace fix-smartquotes fix-spaces
HYGIENE_FROM ?= origin/main
HYGIENE_TO ?= HEAD
PRE_COMMIT ?= uvx pre-commit@4.6.0

hygiene: ## Fix text hygiene in files changed since HYGIENE_FROM (default origin/main)
	@cd '$(MAKEFILE_DIR)..' && status=0; \
	for hook in $(HYGIENE_HOOKS); do \
		$(PRE_COMMIT) run "$$hook" --from-ref '$(HYGIENE_FROM)' --to-ref '$(HYGIENE_TO)' --show-diff-on-failure || status=1; \
	done; \
	exit $$status

lint: ## Lint all packages
	@set -e; \
	for dir in $(LIB_PACKAGE_DIRS); do \
		echo "🔍 Linting $$dir"; \
		$(MAKE) -C "$$dir" lint; \
	done
	@echo "✅ All packages linted!"

format: ## Check text hygiene, then format all packages
	@set -e; \
	status=0; \
	$(MAKE) hygiene || status=$$?; \
	for dir in $(LIB_PACKAGE_DIRS); do \
		echo "🎨 Formatting $$dir"; \
		$(MAKE) -C "$$dir" format; \
	done; \
	exit $$status
	@echo "✅ All packages formatted!"

bench-all: ## Run benchmarks across all benched packages
	@set -e; \
	for pkg in $(BENCH_PACKAGES); do \
		echo "--> bench: $$pkg"; \
		$(MAKE) -C $$pkg bench; \
	done
