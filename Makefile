.PHONY: build-frontends

FRONTEND_SRC := libs/cli/frontend
FRONTEND_DEST := libs/cli/deepagents_cli/deploy/frontend_dist

build-frontends:
	@set -e; \
	echo "--> Building $(FRONTEND_SRC)"; \
	( cd $(FRONTEND_SRC) && npm ci && npm run build ); \
	echo "--> Copying dist into $(FRONTEND_DEST)"; \
	rm -rf $(FRONTEND_DEST); \
	mkdir -p $(FRONTEND_DEST); \
	cp -R $(FRONTEND_SRC)/dist/. $(FRONTEND_DEST)/; \
	echo "Frontend built: $(FRONTEND_DEST)"
