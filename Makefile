.DEFAULT_GOAL := help
.PHONY: help sync test lint format lock lock-check lock-bump bench-all no-cache

PACKAGES := $(patsubst libs/%/Makefile,%,$(wildcard libs/*/Makefile libs/partners/*/Makefile))

ifneq ($(strip $(PACKAGE)),)
ifneq ($(words $(PACKAGE)),1)
$(error Set PACKAGE to one package directory under libs/)
endif
ifneq ($(filter $(PACKAGE),$(PACKAGES)),$(PACKAGE))
$(error Unknown PACKAGE '$(PACKAGE)'; choose one of: $(PACKAGES))
endif
endif

help: ## Show root commands (PACKAGE=code or PACKAGE=partners/quickjs selects a package)
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install all dependency groups for PACKAGE (required)
	$(if $(PACKAGE),,$(error Set PACKAGE, e.g. make sync PACKAGE=code))
	uv sync --directory 'libs/$(PACKAGE)' --all-groups

test: ## Run PACKAGE's unit tests (required; accepts TEST_FILE)
	$(if $(PACKAGE),,$(error Set PACKAGE, e.g. make test PACKAGE=code))
	$(MAKE) -C 'libs/$(PACKAGE)' test

lint: ## Lint all packages, or only PACKAGE
format: ## Format all packages, or only PACKAGE
lint format:
	$(MAKE) -C 'libs$(if $(PACKAGE),/$(PACKAGE))' $@

lock: ## Update all lockfiles (append no-cache to bypass uv's cache)
lock-check: ## Check all lockfiles
lock-bump: ## Bump DEP across all lockfiles
bench-all: ## Run the existing SDK and code benchmark sweep
lock lock-check lock-bump bench-all:
	$(MAKE) -C libs $@ $(if $(filter no-cache,$(MAKECMDGOALS)),no-cache)

no-cache:
	@:
