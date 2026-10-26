#!/usr/bin/env bash
# Install a pinned ripgrep release binary on a Linux x86_64 runner.
#
# Replaces `apt-get update && apt-get install -y ripgrep`: refreshing every apt
# index to install one tool is slow, and the Ubuntu mirrors are the usual cause
# of the install failures that `ci:skip-ripgrep` exists to bypass. The version
# matches what Ubuntu 24.04 (`ubuntu-latest`) ships via apt, and the archive is
# verified against a pinned SHA-256, so every run tests against the same bytes.
#
# Exits non-zero on any failure; callers decide whether that fails the job.

set -euo pipefail

version="14.1.0"
target="x86_64-unknown-linux-musl"
sha256="f84757b07f425fe5cf11d87df6644691c644a5cd2348a2c670894272999d3ba7"

arch="$(uname -m)"
if [ "$arch" != "x86_64" ]; then
  echo "::error::install_ripgrep.sh only supports x86_64 runners, got: $arch"
  exit 1
fi

name="ripgrep-${version}-${target}"
url="https://github.com/BurntSushi/ripgrep/releases/download/${version}/${name}.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl --fail --silent --show-error --location --retry 3 \
  --output "$tmp/${name}.tar.gz" "$url"
echo "${sha256}  $tmp/${name}.tar.gz" | sha256sum --check --strict -
tar -xzf "$tmp/${name}.tar.gz" -C "$tmp" "${name}/rg"
sudo install -m 0755 "$tmp/${name}/rg" /usr/local/bin/rg
rg --version
