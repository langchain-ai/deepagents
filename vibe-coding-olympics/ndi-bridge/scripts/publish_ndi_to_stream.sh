#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Copy .env.example to .env first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
if [[ -x "${ROOT_DIR}/.local/ffmpeg-ndi/bin/ffmpeg" ]]; then
  FFMPEG_BIN="${ROOT_DIR}/.local/ffmpeg-ndi/bin/ffmpeg"
fi

player="${1:-p1}"
case "${player}" in
  p1)
    source_name="${NDI_SOURCE_P1:-}"
    stream_name="${P1_STREAM_NAME:-p1-screen}"
    ;;
  p2)
    source_name="${NDI_SOURCE_P2:-}"
    stream_name="${P2_STREAM_NAME:-p2-screen}"
    ;;
  *)
    echo "ERROR: first arg must be p1 or p2" >&2
    exit 1
    ;;
esac

if [[ -z "${source_name}" ]]; then
  echo "ERROR: missing source name in .env for ${player}." >&2
  exit 1
fi

rtmp_url="${RTMP_URL:-rtmp://127.0.0.1:1935/${stream_name}}"
extra_ips="${NDI_EXTRA_IPS:-}"

echo "==> Publishing NDI source to MediaMTX"
echo "    player:   ${player}"
echo "    source:   ${source_name}"
echo "    stream:   ${stream_name}"
echo "    url:      ${rtmp_url}"
echo "    extraips: ${extra_ips:-<unset>}"
echo

args=(
  "${FFMPEG_BIN}"
  -hide_banner
  -loglevel
  info
  -f
  libndi_newtek
)

if [[ -n "${extra_ips}" ]]; then
  args+=(-extra_ips "${extra_ips}")
fi

args+=(
  -i "${source_name}"
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
  -f
  flv
  "${rtmp_url}"
)

exec "${args[@]}"
