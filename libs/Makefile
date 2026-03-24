PACKAGE_DIRS := $(sort $(patsubst %/,%,$(dir $(wildcard */Makefile partners/*/Makefile))))

# Map package dirs to their required Python version
# acp requires 3.14, everything else uses 3.12
python_version = $(if $(filter acp,$1),3.14,3.12)

.PHONY: help lock lock-check lint format

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

lock: ## Update all lockfiles
	@set -e; \
	$(foreach pkg,$(PACKAGE_DIRS), \
		echo "🔒 Locking $(pkg)"; \
		uv lock --directory $(pkg) --python $(call python_version,$(pkg)); \
	)
	@echo "✅ All lockfiles updated!"

lock-check: ## Check all lockfiles are up-to-date
	@set -e; \
	$(foreach pkg,$(PACKAGE_DIRS), \
		echo "🔍 Checking $(pkg)"; \
		uv lock --check --directory $(pkg) --python $(call python_version,$(pkg)); \
	)
	@echo "✅ All lockfiles are up-to-date!"

lint: ## Lint all packages
	@set -e; \
	for dir in $(PACKAGE_DIRS); do \
		echo "🔍 Linting $$dir"; \
		$(MAKE) -C $$dir lint; \
	done
	@echo "✅ All packages linted!"

format: ## Format all packages
	@set -e; \
	for dir in $(PACKAGE_DIRS); do \
		echo "🎨 Formatting $$dir"; \
		$(MAKE) -C $$dir format; \
	done
	@echo "✅ All packages formatted!"
