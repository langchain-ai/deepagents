"""Example: Using the ChATLAS CLI

This example demonstrates various ways to use the modernized ChATLAS CLI.

The CLI now integrates MCPMiddleware directly and supports sandbox execution,
making it much more powerful and easier to use than the previous version.
"""

# ============================================================================
# Basic Usage Examples
# ============================================================================

# 1. Simple interactive session (default behavior)
# This is the most common way to use ChATLAS
"""
$ chatlas

This launches an interactive AI assistant with:
- ChATLAS MCP tools for searching ATLAS documentation
- DeepAgents capabilities (file operations, planning, sub-agents)
- Skills system for custom tools
- Memory for conversation persistence
- Human-in-the-loop approval for destructive operations
"""

# 2. Use a custom agent name (for separate conversation history)
"""
$ chatlas --agent my-research-agent

Each agent name gets its own:
- Conversation history
- Memory state
- Skills directory
"""

# 3. Initialize configuration file
"""
$ chatlas init

Creates a .env file with:
- CHATLAS_MCP_URL (default: https://chatlas-mcp.app.cern.ch/mcp)
- CHATLAS_MCP_TIMEOUT (default: 120 seconds)
- CHATLAS_LLM_PROVIDER (default: openai)
- CHATLAS_LLM_MODEL (default: gpt-4)
- OPENAI_API_KEY (required)
- ANTHROPIC_API_KEY (optional)
- TAVILY_API_KEY (optional, for web search)
"""

# 4. Show version
"""
$ chatlas version

ChATLAS Agents v0.1.0
Built on DeepAgents framework with MCP integration
"""

# ============================================================================
# Advanced Usage Examples
# ============================================================================

# 5. Use a specific LLM model
"""
$ chatlas --model gpt-4o

Override the default model (useful for testing different models)
"""

# 6. Override MCP server URL (for custom/local MCP servers)
"""
$ chatlas --mcp-url http://localhost:8080/mcp
"""

# 7. Auto-approve all tool calls (non-interactive mode)
"""
$ chatlas --auto-approve

Useful for:
- Automation scripts
- CI/CD pipelines
- Batch processing
"""

# 8. Enable verbose logging (for debugging)
"""
$ chatlas --verbose

Shows detailed logs including:
- MCP server connection details
- Tool loading progress
- LLM API calls
- Agent state transitions
"""

# ============================================================================
# Sandbox Execution Examples
# ============================================================================

# 9. Docker sandbox (isolated code execution)
"""
$ chatlas --sandbox docker

Benefits:
- Isolated environment for running code
- Secure execution boundaries
- Custom container images supported
- File upload/download capabilities
"""

# 10. Docker sandbox with custom image
"""
$ chatlas --sandbox docker --sandbox-image python:3.11-slim

Use a specific container image (e.g., for different Python versions)
"""

# 11. Apptainer sandbox (for HPC environments like lxplus)
"""
$ chatlas --sandbox apptainer

Ideal for CERN lxplus and other HPC environments where Docker isn't available.
Uses Apptainer/Singularity for containerization.
"""

# 12. Apptainer with custom image
"""
$ chatlas --sandbox apptainer --sandbox-image docker://python:3.13-slim

Apptainer can pull Docker images using the docker:// prefix
"""

# ============================================================================
# Configuration File Examples
# ============================================================================

# 13. Using YAML configuration (advanced)
"""
$ chatlas --config my-config.yaml

Example my-config.yaml:
---
name: "atlas-analysis-agent"
description: "AI assistant for ATLAS physics analysis"

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7

mcp:
  url: "https://chatlas-mcp.app.cern.ch/mcp"
  timeout: 120

max_iterations: 15
verbose: true
"""

# 14. Using environment variables
"""
# Set in your .env file or export directly
export CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
export CHATLAS_MCP_TIMEOUT=120
export CHATLAS_LLM_PROVIDER=openai
export CHATLAS_LLM_MODEL=gpt-4
export OPENAI_API_KEY=sk-...

# Then just run
$ chatlas
"""

# ============================================================================
# Real-World Usage Scenarios
# ============================================================================

# Scenario 1: Research on lxplus with Apptainer
"""
# SSH to lxplus
ssh lxplus.cern.ch

# Setup environment
export OPENAI_API_KEY=sk-...

# Run with Apptainer sandbox
chatlas --sandbox apptainer --sandbox-image docker://python:3.13-slim

# Now you can:
# - Search ATLAS documentation
# - Execute Python code in isolated container
# - Create analysis scripts
# - Generate plots
"""

# Scenario 2: Automated documentation search
"""
# Create a script that queries ChATLAS
echo "Search for ATLAS trigger documentation" | chatlas --auto-approve > results.txt

# Parse results and use in your workflow
"""

# Scenario 3: Development with local MCP server
"""
# Run local MCP server
python -m chatlas_mcp.server --port 8080

# Connect ChATLAS to it
chatlas --mcp-url http://localhost:8080/mcp --verbose
"""

# Scenario 4: Multi-agent workflow
"""
# Agent 1: Research agent
chatlas --agent research --auto-approve

# Agent 2: Analysis agent (separate memory)
chatlas --agent analysis --auto-approve

# Agent 3: Documentation agent
chatlas --agent docs --auto-approve
"""

# ============================================================================
# Integration with Other Tools
# ============================================================================

# Integration 1: With shell scripts
"""
#!/bin/bash
# run_atlas_query.sh

export OPENAI_API_KEY=$(cat ~/.openai_key)
echo "$1" | chatlas --auto-approve --agent research > query_results.txt
cat query_results.txt
"""

# Integration 2: With Python scripts
"""
import subprocess
import os

# Set API key
os.environ['OPENAI_API_KEY'] = 'sk-...'

# Run ChATLAS query
result = subprocess.run(
    ['chatlas', '--auto-approve'],
    input='Search for ATLAS b-tagging algorithms',
    capture_output=True,
    text=True
)

print(result.stdout)
"""

# ============================================================================
# Troubleshooting
# ============================================================================

# Problem: Can't connect to MCP server
"""
$ chatlas --verbose --mcp-url https://chatlas-mcp.app.cern.ch/mcp

Check:
1. MCP server is running
2. Network connectivity
3. Firewall rules
4. Server timeout setting
"""

# Problem: Sandbox not working
"""
$ chatlas --sandbox docker --verbose

Check:
1. Docker is installed and running
2. User has Docker permissions
3. Container image is available
4. Disk space for container
"""

# Problem: LLM API errors
"""
$ chatlas --verbose

Check:
1. API key is set correctly
2. API key has sufficient credits
3. Model name is correct
4. Network can reach LLM API
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nFor more information, run: chatlas --help")
