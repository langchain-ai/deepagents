#!/usr/bin/env python3
"""CLI Agent with RAG - This shows how the CLI agent is configured with RAG.

The deepagents CLI automatically includes RAG when OPENAI_API_KEY is set.
This script demonstrates the same configuration that the CLI uses.

To use the CLI agent with RAG:
  1. Set OPENAI_API_KEY environment variable
  2. Run: deepagents
  3. Ask the agent to search code semantically

Example CLI usage:
  $ export OPENAI_API_KEY='your-key'
  $ deepagents
  > Find code that handles authentication
  > Search for error handling patterns
"""

import os
import sys
from pathlib import Path

# This shows how the CLI agent is configured (same as in main.py)
print("=" * 60)
print("DeepAgents CLI Agent with RAG")
print("=" * 60)
print()
print("The CLI agent automatically includes RAG when OPENAI_API_KEY is set.")
print()
print("To use the CLI agent:")
print("  1. Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key'")
print("  2. Run: deepagents")
print("  3. Ask questions like:")
print("     - 'Find code that handles authentication'")
print("     - 'Search for error handling patterns'")
print("     - 'Where is database connection management?'")
print()
print("The semantic_search tool is automatically added to the agent.")
print("See: libs/deepagents-cli/deepagents_cli/main.py (lines 296-305)")
print()

# Check if RAG would be enabled
has_rag = bool(os.getenv("OPENAI_API_KEY"))
if has_rag:
    print("✓ OPENAI_API_KEY is set - RAG will be enabled in CLI")
else:
    print("⚠ OPENAI_API_KEY not set - RAG will be disabled")
    print("  Set it to enable semantic code search")
    print()

print("To start the CLI agent, run:")
print("  deepagents")
print("  # or")
print("  python -m deepagents_cli")
print()
print("For more information, see: examples/README_RAG.md")
