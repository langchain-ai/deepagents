.DEFAULT_GOAL := help

ENV_FILE ?= .env
DOCKER_COMPOSE := docker compose --env-file $(ENV_FILE)

STREAM ?= p1-screen
PLAYER ?= p1
FPS ?= 30
SIZE ?= 1920x1080
DURATION ?= 0
WAIT ?= 5s
NDI_SDK_TARBALL ?= vendor/ndi-sdk-macos.tar.gz

.PHONY: help env up down logs testsrc-up testsrc-down colorbars colorbars-p1 colorbars-p2 build-ffmpeg-ndi discover publish publish-p1 publish-p2

help:
	@echo "ndi-bridge targets"
	@echo
	@echo "  make env            # create .env from .env.example (if missing)"
	@echo "  make up             # start mediamtx"
	@echo "  make down           # stop mediamtx stack"
	@echo "  make logs           # tail mediamtx logs"
	@echo "  make testsrc-up     # start docker test pattern publishers"
	@echo "  make testsrc-down   # stop docker test pattern publishers"
	@echo "  make colorbars      # local ffmpeg color bars (STREAM=..., DURATION=..., FPS=..., SIZE=...)"
	@echo "  make colorbars-p1   # local ffmpeg color bars to p1-screen"
	@echo "  make colorbars-p2   # local ffmpeg color bars to p2-screen"
	@echo "  make build-ffmpeg-ndi # build local macOS ffmpeg with NDI input (defaults NDI_SDK_TARBALL=$(NDI_SDK_TARBALL))"
	@echo "  make discover       # discover NDI sources (WAIT=5s)"
	@echo "  make publish        # publish NDI source (PLAYER=p1|p2)"
	@echo "  make publish-p1     # publish NDI_SOURCE_P1 -> p1-screen"
	@echo "  make publish-p2     # publish NDI_SOURCE_P2 -> p2-screen"

env:
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "$(ENV_FILE) already exists"; \
	else \
		cp .env.example "$(ENV_FILE)"; \
		echo "created $(ENV_FILE) from .env.example"; \
	fi

up:
	$(DOCKER_COMPOSE) up -d mediamtx

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f mediamtx

testsrc-up:
	$(DOCKER_COMPOSE) --profile testsrc up -d

testsrc-down:
	$(DOCKER_COMPOSE) --profile testsrc down

colorbars:
	DURATION=$(DURATION) FPS=$(FPS) SIZE=$(SIZE) ./scripts/publish_color_bars.sh "$(STREAM)"

colorbars-p1:
	DURATION=$(DURATION) FPS=$(FPS) SIZE=$(SIZE) ./scripts/publish_color_bars.sh p1-screen

colorbars-p2:
	DURATION=$(DURATION) FPS=$(FPS) SIZE=$(SIZE) ./scripts/publish_color_bars.sh p2-screen

build-ffmpeg-ndi:
	NDI_SDK_ROOT="$(NDI_SDK_ROOT)" NDI_SDK_TARBALL="$(NDI_SDK_TARBALL)" ./scripts/build_local_ffmpeg_ndi_macos.sh

discover:
	NDI_WAIT_SOURCES=$(WAIT) ./scripts/discover_ndi_sources.sh

publish:
	./scripts/publish_ndi_to_stream.sh "$(PLAYER)"

publish-p1:
	./scripts/publish_ndi_to_stream.sh p1

publish-p2:
	./scripts/publish_ndi_to_stream.sh p2
