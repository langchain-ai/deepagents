# Setup Instructions for chatlas-agents Module

This document explains how to set up the `chatlas-agents` module within the deepagents fork repository, including handling local dependencies.

## Dependencies Overview

The `chatlas-agents` module depends on two other modules in this repository:
- `deepagents` (version 0.3.0) - Located in `../deepagents`
- `deepagents-cli` (version 0.0.10) - Located in `../deepagents-cli`

These dependencies are configured to use **local path references** via the `[tool.uv.sources]` section in `pyproject.toml`, which means:
- ✅ No circular dependencies exist
- ✅ You can develop and test all three modules together
- ✅ Changes in `deepagents` or `deepagents-cli` are immediately reflected in `chatlas-agents`

## Dependency Resolution

### Version Requirements

The modules have been configured with compatible version requirements:

- `chatlas-agents` requires:
  - `deepagents>=0.3.0`
  - `deepagents-cli>=0.0.3`

- `deepagents-cli` requires:
  - `deepagents>=0.3.0` (updated from `==0.2.8` to resolve conflicts)

- `deepagents`:
  - Version 0.3.0
  - No dependencies on other local modules

### Local Path Configuration

The `pyproject.toml` file for `chatlas-agents` includes:

```toml
[tool.uv.sources]
deepagents = { path = "../deepagents" }
deepagents-cli = { path = "../deepagents-cli" }
```

This configuration tells `uv` to use the local versions of these packages instead of fetching them from PyPI.

## Installation from libs/chatlas-agents Directory

### Prerequisites

1. Python 3.11 or higher
2. Either `uv` or `pip` installed

### Option 1: Using uv (Recommended)

```bash
# Navigate to the chatlas-agents directory
cd libs/chatlas-agents

# Install uv if not already installed
pip install uv

# Sync dependencies (this will install both local and remote dependencies)
uv sync

# Run the CLI
uv run python -m chatlas_agents.cli --help
```

### Option 2: Using pip with editable installs

```bash
# Navigate to the repository root
cd /path/to/deepagents

# Install deepagents first (as it has no local dependencies)
pip install -e libs/deepagents

# Install deepagents-cli (depends on deepagents)
pip install -e libs/deepagents-cli

# Install chatlas-agents (depends on both above)
pip install -e libs/chatlas-agents
```

### Option 3: Using pip from chatlas-agents directory

```bash
# Navigate to the chatlas-agents directory
cd libs/chatlas-agents

# Install local dependencies first
pip install -e ../deepagents
pip install -e ../deepagents-cli

# Install chatlas-agents itself
pip install -e .
```

## Verifying the Installation

After installation, you can verify that everything is set up correctly:

```bash
# Check that all packages are installed
pip list | grep -E "(deepagents|chatlas)"

# Should show something like:
# chatlas-agents    0.1.0    /path/to/deepagents/libs/chatlas-agents
# deepagents        0.3.0    /path/to/deepagents/libs/deepagents
# deepagents-cli    0.0.10   /path/to/deepagents/libs/deepagents-cli

# Test the CLI
python -m chatlas_agents.cli --help
```

## Development Workflow

When developing across multiple modules:

1. **Make changes** in any of the three modules
2. **No reinstallation needed** if using editable installs (`-e` flag)
3. **Changes are immediately reflected** in dependent modules
4. **Run tests** from the respective module directory

```bash
# Test deepagents
cd libs/deepagents
pytest

# Test deepagents-cli
cd ../deepagents-cli
pytest

# Test chatlas-agents
cd ../chatlas-agents
pytest
```

## Troubleshooting

### Issue: Version conflicts

If you see version conflicts like:
```
ERROR: Cannot install chatlas-agents because these package versions have conflicting dependencies.
```

**Solution**: Ensure you're using the latest versions of all three modules with compatible version requirements.

### Issue: Module not found errors

If you see:
```
ModuleNotFoundError: No module named 'deepagents'
```

**Solution**: Install the dependencies in the correct order (deepagents → deepagents-cli → chatlas-agents) as shown above.

### Issue: Changes not reflected

If changes in `deepagents` or `deepagents-cli` are not reflected in `chatlas-agents`:

**Solution**: Ensure you used editable installs (`pip install -e .`) rather than regular installs.

## CI/CD Considerations

For continuous integration:

1. Install modules in dependency order
2. Use editable installs for development/testing
3. Consider using `uv` for faster dependency resolution
4. Cache the `.venv` directory to speed up subsequent runs

Example GitHub Actions workflow:

```yaml
- name: Install dependencies
  run: |
    pip install uv
    cd libs/deepagents && uv sync
    cd ../deepagents-cli && uv sync
    cd ../chatlas-agents && uv sync
```

## Summary

- ✅ No circular dependencies between the three modules
- ✅ `chatlas-agents` can be set up from its own directory
- ✅ Local path references ensure development changes are immediately available
- ✅ Version requirements have been aligned (deepagents-cli now uses `>=0.3.0` instead of `==0.2.8`)
- ✅ Both `uv` and `pip` installation methods are supported
