#!/usr/bin/env bash
# Deploy the canonical Deep Agents system into a continual-learning-bench checkout.
#
# clbench discovers systems by scanning its own src/systems/<name>/ tree
# (see src/registry.py:_discover_system_modules), so the adapter must physically
# live there to be runnable. This copies the canonical source kept here in the
# deepagents repo into <clbench>/src/systems/deepagents/.
#
# Usage: ./sync_to_clbench.sh /path/to/continual-learning-bench
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_dir="${here}/system"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <path-to-clbench-checkout>" >&2
  exit 2
fi

clbench="$1"
if [[ ! -d "${clbench}" ]]; then
  echo "error: '${clbench}' is not a directory" >&2
  exit 1
fi
if [[ ! -d "${clbench}/src/systems" ]]; then
  echo "error: '${clbench}' does not look like a clbench checkout (missing src/systems/)" >&2
  exit 1
fi

dest="${clbench}/src/systems/deepagents"
mkdir -p "${dest}"
cp "${src_dir}/__init__.py" "${src_dir}/system.py" "${dest}/"

echo "Deployed Deep Agents system -> ${dest}"
echo "Run it with:  clbench run <task> --system deepagents"
