PACKAGE_DIRS = libs/deepagents libs/cli libs/acp libs/harbor libs/partners/daytona libs/partners/modal libs/partners/runloop

# Map package dirs to their required Python version
# acp requires 3.14, everything else uses 3.12
python_version = $(if $(filter libs/acp,$1),3.14,3.12)

.PHONY: lock lock-check

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
