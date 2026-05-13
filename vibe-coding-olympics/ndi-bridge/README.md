> [!NOTE]
> Experimented with but **not used during the day-of event**. Kept for reference only.

# `ndi-bridge/` — local ffmpeg NDI bridge + MediaMTX

This path is now local-ffmpeg only.

- `mediamtx` runs in Docker.
- NDI ingest runs on host via local `ffmpeg` (`libndi_newtek`).
- No NDI Docker gateway image is required.

## Files

- `docker-compose.yml` — MediaMTX + optional Docker test patterns
- `mediamtx.yml` — MediaMTX config (includes relaxed read timeout)
- `.env.example` — local runtime config
- `scripts/build_local_ffmpeg_ndi_macos.sh` — build NDI-enabled host ffmpeg (macOS)
- `scripts/discover_ndi_sources.sh` — discover NDI sources with local ffmpeg
- `scripts/publish_ndi_to_stream.sh` — publish `NDI_SOURCE_P1/P2` to MediaMTX
- `scripts/publish_color_bars.sh` — publish SMPTE bars (no NDI)

## Quick Start

1. Configure env:

```bash
cd vibe-coding-olympics/ndi-bridge
cp .env.example .env
```

1. Start MediaMTX:

```bash
docker compose --env-file .env up -d mediamtx
```

1. Validate pipeline without NDI:

```bash
./scripts/publish_color_bars.sh p1-screen
```

1. Build host ffmpeg with NDI support (macOS only, if needed):

```bash
brew install x264 pkg-config nasm yasm
./scripts/build_local_ffmpeg_ndi_macos.sh
```

If your SDK is a tar archive, point the build at it:

```bash
make build-ffmpeg-ndi NDI_SDK_TARBALL=/path/to/ndi-sdk-macos.tar.gz
```

Default behavior:

- If `vendor/ndi-sdk-macos.tar.gz` exists, `make build-ffmpeg-ndi` uses it automatically.
- Otherwise it falls back to `NDI_SDK_ROOT` (default: `/Library/NDI SDK for Apple`).

1. Discover source names:

```bash
./scripts/discover_ndi_sources.sh
```

1. Set `NDI_SOURCE_P1` / `NDI_SOURCE_P2` in `.env`, then publish:

```bash
./scripts/publish_ndi_to_stream.sh p1
./scripts/publish_ndi_to_stream.sh p2
```

Open in browser:

- `http://127.0.0.1:8889/p1-screen`
- `http://127.0.0.1:8889/p2-screen`

## Make Targets

You can run the entire flow with `make`:

```bash
make env
make up
make colorbars-p1 DURATION=20
make discover
make publish-p1
make publish-p2
```

Useful targets:

- `make up` / `make down`
- `make logs`
- `make colorbars` (override `STREAM`, `DURATION`, `FPS`, `SIZE`)
- `make build-ffmpeg-ndi` (macOS only)
- `make discover` (override `WAIT=10s`)
- `make publish` (override `PLAYER=p1|p2`)

## browser-control

In `browser-control` set:

- P1 source mode: `ndi`, stream: `p1-screen`
- P2 source mode: `ndi`, stream: `p2-screen`

## Troubleshooting

- `Broken pipe` from ffmpeg:
  - Check `docker compose logs mediamtx`.
  - If you see `read tcp ... i/o timeout`, increase `readTimeout` in `mediamtx.yml`.
- `Unknown input format: libndi_newtek`:
  - Your ffmpeg is missing NDI input support; build local ffmpeg first.

## Stop

```bash
docker compose --env-file .env down
```

## Hand-off

Transfer this folder without local runtime artifacts:

- `.env`
- `.build/`
- `.local/`

On destination:

1. `cp .env.example .env`
2. `docker compose --env-file .env up -d mediamtx`
3. `./scripts/publish_color_bars.sh p1-screen` (smoke test)
