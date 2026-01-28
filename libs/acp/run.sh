#!/bin/bash
# Wrapper script to run deepagents-acp with current directory as root
# Captures the current working directory where Zed invoked this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
uv run deepagents-acp --root-dir "${1:-$OLDPWD}"
