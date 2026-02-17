PACKAGE_DIRS = libs/deepagents libs/cli libs/acp libs/harbor libs/partners/daytona libs/partners/modal libs/partners/runloop

# Map package dirs to their required Python version
# acp requires 3.14, everything else uses 3.12
python_version = $(if $(filter libs/acp,$1),3.14,3.12)

.PHONY: lock lock-check lint format

lock:
	@set -e; \
	for dir in $(PACKAGE_DIRS); do \
		echo "🔒 Locking $$dir"; \
		uv lock --directory $$dir --python $(call python_version,$$dir); \
	done
	@echo "✅ All lockfiles updated!"

lock-check:
	@set -e; \
	for dir in $(PACKAGE_DIRS); do \
		echo "🔍 Checking $$dir"; \
		uv lock --check --directory $$dir --python $(call python_version,$$dir); \
	done
	@echo "✅ All lockfiles are up-to-date!"

lint:
	@set -e; \
	for dir in $(PACKAGE_DIRS); do \
		echo "🔍 Linting $$dir"; \
		$(MAKE) -C $$dir lint; \
	done
	@echo "✅ All packages linted!"

format:
	@set -e; \
	for dir in $(PACKAGE_DIRS); do \
		echo "🎨 Formatting $$dir"; \
		$(MAKE) -C $$dir format; \
	done
	@echo "✅ All packages formatted!"
