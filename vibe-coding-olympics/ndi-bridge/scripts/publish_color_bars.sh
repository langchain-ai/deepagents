#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STREAM_NAME="${1:-${STREAM_NAME:-p1-screen}}"
DURATION="${DURATION:-0}"
FPS="${FPS:-30}"
SIZE="${SIZE:-1920x1080}"
RTMP_URL="${RTMP_URL:-rtmp://127.0.0.1:1935/${STREAM_NAME}}"

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
if [[ -x "${ROOT_DIR}/.local/ffmpeg-ndi/bin/ffmpeg" ]]; then
  # Prefer local pinned ffmpeg build when available.
  FFMPEG_BIN="${ROOT_DIR}/.local/ffmpeg-ndi/bin/ffmpeg"
fi

echo "==> Publishing SMPTE color bars"
echo "    stream:   ${STREAM_NAME}"
echo "    url:      ${RTMP_URL}"
echo "    fps:      ${FPS}"
echo "    size:     ${SIZE}"
if [[ "${DURATION}" != "0" ]]; then
  echo "    duration: ${DURATION}s"
else
  echo "    duration: infinite (ctrl+c to stop)"
fi
echo

cmd=(
  "${FFMPEG_BIN}"
  -hide_banner
  -loglevel
  info
  -re
  -f
  lavfi
  -i
  "smptebars=size=${SIZE}:rate=${FPS}"
  -an
  -c:v
  libx264
  -profile:v
  baseline
  -pix_fmt
  yuv420p
  -bf
  0
  -g
  30
  -keyint_min
  30
  -sc_threshold
  0
  -tune
  zerolatency
)

if [[ "${DURATION}" != "0" ]]; then
  cmd+=(-t "${DURATION}")
fi

cmd+=(
  -f
  flv
  "${RTMP_URL}"
)

exec "${cmd[@]}"
