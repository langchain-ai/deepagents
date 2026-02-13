#!/bin/bash
# Wrapper script to run deepagents-acp with current directory as root
cd "$(dirname "$0")"
uv run python ./examples/demo_agent.py
