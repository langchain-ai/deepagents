#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
WAIT_SOURCES="${NDI_WAIT_SOURCES:-5s}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Copy .env.example to .env first." >&2
  exit 1
fi

if [[ -x "${ROOT_DIR}/.local/ffmpeg-ndi/bin/ffmpeg" ]]; then
  # Prefer local pinned ffmpeg build when available.
  FFMPEG_BIN="${ROOT_DIR}/.local/ffmpeg-ndi/bin/ffmpeg"
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if ! "${FFMPEG_BIN}" -hide_banner -demuxers | rg -q 'libndi_newtek'; then
  echo "ERROR: ${FFMPEG_BIN} does not include libndi_newtek input support." >&2
  echo "Build local ffmpeg first with:" >&2
  echo "  ./scripts/build_local_ffmpeg_ndi_macos.sh" >&2
  exit 1
fi

echo "==> Discovering NDI sources via local ffmpeg"
echo "    ffmpeg:   ${FFMPEG_BIN}"
echo "    env-file: ${ENV_FILE}"
echo "    wait:     ${WAIT_SOURCES}"
echo "    server:   ${NDI_DISCOVERY_SERVER:-<unset>}"
echo "    extraips: ${NDI_EXTRA_IPS:-<unset>}"
echo

args=(
  "${FFMPEG_BIN}"
  -hide_banner
  -loglevel
  info
  -f libndi_newtek
  -find_sources 1
)

if [[ -n "${NDI_EXTRA_IPS:-}" ]]; then
  args+=(-extra_ips "${NDI_EXTRA_IPS}")
fi

args+=(
  -wait_sources "${WAIT_SOURCES}"
  -i dummy
  -t 1
  -f null -
)

exec "${args[@]}"
