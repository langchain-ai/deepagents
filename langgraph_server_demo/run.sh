#!/bin/bash
# Helper script to run LangGraph server

set -e

echo "🚀 Starting LangGraph Server..."
echo ""

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python 3.13 virtual environment..."
    ~/.local/bin/uv venv --python 3.13
fi

# Install with uv
echo "📦 Installing dependencies..."
~/.local/bin/uv pip install "langgraph-cli[inmem]" httpx

echo ""
echo "✅ Starting server on http://localhost:2024"
echo "🎨 LangGraph Studio: http://localhost:2024/studio"
echo ""

# Run the server
~/.local/bin/uv run langgraph dev
