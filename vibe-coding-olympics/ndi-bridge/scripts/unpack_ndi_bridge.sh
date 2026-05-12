#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/unpack_ndi_bridge.sh [archive.tar.gz] [destination_dir]

Examples:
  ./scripts/unpack_ndi_bridge.sh
  ./scripts/unpack_ndi_bridge.sh ndi-bridge-share-20260511-190034.tar.gz
  ./scripts/unpack_ndi_bridge.sh ndi-bridge-share-20260511-190034.tar.gz /tmp

Notes:
  - If archive is omitted, the newest ./ndi-bridge-share-*.tar.gz in the current
    directory is used.
  - Set FORCE=1 to allow extracting over an existing destination/ndi-bridge.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

archive="${1:-}"
dest_dir="${2:-.}"

if [[ -z "${archive}" ]]; then
  archive="$(ls -1t ./ndi-bridge-share-*.tar.gz 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${archive}" || ! -f "${archive}" ]]; then
  echo "ERROR: archive not found." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${dest_dir}" ]]; then
  mkdir -p "${dest_dir}"
fi

target="${dest_dir%/}/ndi-bridge"
if [[ -e "${target}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${target} already exists." >&2
  echo "Set FORCE=1 to extract over it, or choose a different destination." >&2
  exit 1
fi

echo "==> Extracting archive"
echo "    archive: ${archive}"
echo "    dest:    ${dest_dir}"

tar -xzf "${archive}" -C "${dest_dir}"

echo "==> Done"
echo "    extracted: ${target}"
