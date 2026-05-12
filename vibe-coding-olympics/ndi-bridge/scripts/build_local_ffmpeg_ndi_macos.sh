#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: this script is for macOS only." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

FFMPEG_REF="${FFMPEG_REF:-n5.1}"
PATCH_REPO="${PATCH_REPO:-https://github.com/lplassman/FFMPEG-NDI.git}"
PATCH_REF="${PATCH_REF:-master}"
NDI_SDK_ROOT="${NDI_SDK_ROOT:-/Library/NDI SDK for Apple}"
NDI_SDK_TARBALL="${NDI_SDK_TARBALL:-}"
PREFIX="${PREFIX:-${ROOT_DIR}/.local/ffmpeg-ndi}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/.build/ffmpeg-ndi}"
NPROC="$(sysctl -n hw.ncpu)"
NDI_SDK_LINK="${BUILD_DIR}/ndi-sdk"
NDI_SDK_EXTRACT_DIR="${BUILD_DIR}/ndi-sdk-extracted"

if [[ -z "${NDI_SDK_TARBALL}" ]]; then
  for candidate in \
    "${ROOT_DIR}/vendor/ndi-sdk-macos.tar.gz" \
    "${ROOT_DIR}/vendor/ndi-sdk.tar.gz" \
    "${ROOT_DIR}/ndi-sdk-macos.tar.gz"; do
    if [[ -f "${candidate}" ]]; then
      NDI_SDK_TARBALL="${candidate}"
      break
    fi
  done
fi

echo "==> Building local macOS ffmpeg with NDI support"
echo "    ffmpeg ref: ${FFMPEG_REF}"
echo "    patch ref:  ${PATCH_REF}"
echo "    sdk root:   ${NDI_SDK_ROOT}"
if [[ -n "${NDI_SDK_TARBALL}" ]]; then
  echo "    sdk tar:    ${NDI_SDK_TARBALL}"
fi
echo "    sdk link:   ${NDI_SDK_LINK}"
echo "    prefix:     ${PREFIX}"
echo "    build dir:  ${BUILD_DIR}"

for cmd in git make pkg-config tar; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}" >&2
    exit 1
  fi
done

if ! pkg-config --exists x264; then
  echo "ERROR: x264 not found via pkg-config." >&2
  echo "Install with: brew install x264 pkg-config nasm yasm" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${PREFIX}"
rm -rf "${BUILD_DIR}/ffmpeg" "${BUILD_DIR}/ffmpeg-ndi-patch"

if [[ -n "${NDI_SDK_TARBALL}" ]]; then
  echo "==> SDK tarball mode enabled"

  if [[ ! -f "${NDI_SDK_TARBALL}" ]]; then
    echo "ERROR: sdk tarball not found: ${NDI_SDK_TARBALL}" >&2
    exit 1
  fi

  rm -rf "${NDI_SDK_EXTRACT_DIR}"
  mkdir -p "${NDI_SDK_EXTRACT_DIR}"
  tar -xf "${NDI_SDK_TARBALL}" -C "${NDI_SDK_EXTRACT_DIR}"

  sdk_header_path="$(find "${NDI_SDK_EXTRACT_DIR}" -type f -name 'Processing.NDI.Lib.h' | head -n1 || true)"
  if [[ -z "${sdk_header_path}" ]]; then
    echo "ERROR: could not find Processing.NDI.Lib.h in ${NDI_SDK_TARBALL}" >&2
    exit 1
  fi

  NDI_SDK_ROOT="$(cd "$(dirname "${sdk_header_path}")/.." && pwd)"
  echo "==> Using extracted SDK root: ${NDI_SDK_ROOT}"
fi

if [[ ! -f "${NDI_SDK_ROOT}/include/Processing.NDI.Lib.h" ]]; then
  echo "ERROR: missing ${NDI_SDK_ROOT}/include/Processing.NDI.Lib.h" >&2
  exit 1
fi

if [[ ! -f "${NDI_SDK_ROOT}/lib/macOS/libndi.dylib" ]]; then
  echo "ERROR: missing ${NDI_SDK_ROOT}/lib/macOS/libndi.dylib" >&2
  echo "Expected SDK layout under: ${NDI_SDK_ROOT}" >&2
  exit 1
fi

ln -sfn "${NDI_SDK_ROOT}" "${NDI_SDK_LINK}"

if [[ ! -f "${NDI_SDK_LINK}/include/Processing.NDI.Lib.h" ]]; then
  echo "ERROR: missing ${NDI_SDK_LINK}/include/Processing.NDI.Lib.h via symlink" >&2
  exit 1
fi

if [[ ! -f "${NDI_SDK_LINK}/lib/macOS/libndi.dylib" ]]; then
  echo "ERROR: missing ${NDI_SDK_LINK}/lib/macOS/libndi.dylib via symlink" >&2
  exit 1
fi

echo "==> Cloning sources"
git clone --depth 1 --branch "${FFMPEG_REF}" https://github.com/FFmpeg/FFmpeg.git "${BUILD_DIR}/ffmpeg"
git clone --depth 1 --branch "${PATCH_REF}" "${PATCH_REPO}" "${BUILD_DIR}/ffmpeg-ndi-patch"

cd "${BUILD_DIR}/ffmpeg"
git config user.email "builder@example.invalid"
git config user.name "builder"

echo "==> Applying NDI patch"
git am "${BUILD_DIR}/ffmpeg-ndi-patch/libndi.patch"
cp "${BUILD_DIR}/ffmpeg-ndi-patch/libavdevice/libndi_newtek_"* libavdevice/

echo "==> Configuring"
./configure \
  --prefix="${PREFIX}" \
  --enable-gpl \
  --enable-nonfree \
  --enable-libx264 \
  --enable-libndi_newtek \
  --disable-debug \
  --disable-doc \
  --disable-ffplay \
  --extra-cflags="-I${NDI_SDK_LINK}/include" \
  --extra-ldflags="-L${NDI_SDK_LINK}/lib/macOS -Wl,-rpath,${NDI_SDK_LINK}/lib/macOS"

echo "==> Building (jobs: ${NPROC})"
make -j"${NPROC}"
make install

echo "==> Verifying NDI demuxer"
"${PREFIX}/bin/ffmpeg" -hide_banner -demuxers | grep -qi libndi_newtek || {
  echo "ERROR: local ffmpeg build does not expose libndi_newtek demuxer." >&2
  exit 1
}

echo
echo "Done."
echo "Use this binary:"
echo "  ${PREFIX}/bin/ffmpeg"
echo
echo "Quick check:"
echo "  ${PREFIX}/bin/ffmpeg -hide_banner -demuxers | rg -i ndi"
