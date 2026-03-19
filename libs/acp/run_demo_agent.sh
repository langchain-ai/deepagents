#!/bin/bash
# Wrapper script to run deepagents-acp with the deps in the script directory
# but with the current working directory preserved
SCRIPT_DIR="$(dirname "$0")"
uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/examples/demo_agent.py"
