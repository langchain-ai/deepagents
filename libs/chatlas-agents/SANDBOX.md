# Sandbox Backend Documentation

This document describes the ChATLAS sandbox implementation and how it integrates with the deepagents framework.

## Overview

The `chatlas_agents.sandbox` module provides container-based sandbox backends for secure code execution in DeepAgents. It supports both Docker and Apptainer (formerly Singularity), making it suitable for both local development and HPC environments like CERN's lxplus.

## Architecture

### Backend Protocol

Our sandbox backends inherit from `deepagents.backends.sandbox.BaseSandbox`, which provides default implementations for file operations using shell commands. Each backend only needs to implement four core methods:

1. **`execute(command: str) -> ExecuteResponse`** - Execute shell commands
2. **`upload_files(files: list[tuple[str, bytes]]) -> list[FileUploadResponse]`** - Upload files to sandbox
3. **`download_files(paths: list[str]) -> list[FileDownloadResponse]`** - Download files from sandbox
4. **`id` property** - Unique identifier for the sandbox instance

BaseSandbox automatically provides:
- `ls_info()` - List directory contents
- `read()` - Read files with line numbers
- `write()` - Create new files
- `edit()` - Edit existing files
- `grep_raw()` - Search file contents
- `glob_info()` - Find files by pattern

### Supported Backends

#### DockerSandboxBackend

Uses Docker containers for isolation. Best for:
- Local development
- Environments with Docker installed
- Quick iteration and testing

```python
from chatlas_agents.sandbox import DockerSandboxBackend

backend = DockerSandboxBackend(
    image="python:3.13-slim",
    working_dir="/workspace",
    auto_remove=True,
)
```

#### ApptainerSandboxBackend

Uses Apptainer/Singularity instances. Best for:
- HPC environments (CERN lxplus)
- Environments without Docker daemon
- ATLAS software environment
- Systems requiring rootless containers

```python
from chatlas_agents.sandbox import ApptainerSandboxBackend

backend = ApptainerSandboxBackend(
    image="docker://python:3.13-slim",
    working_dir="/workspace",
    auto_remove=True,
)
```

## Factory Functions (Recommended)

Following the deepagents-cli pattern, we provide context manager factory functions for better lifecycle management:

### create_docker_sandbox

```python
from chatlas_agents.sandbox import create_docker_sandbox
from deepagents import create_deep_agent

with create_docker_sandbox(
    image="python:3.13-slim",
    working_dir="/workspace",
    auto_remove=True,
) as backend:
    agent = create_deep_agent(backend=backend)
    result = agent.invoke({"messages": [...]})
# Automatic cleanup when exiting context
```

**Benefits:**
- Automatic cleanup on exit
- Support for setup scripts
- Environment variable expansion in setup scripts
- Consistent with deepagents-cli patterns

### create_apptainer_sandbox

```python
from chatlas_agents.sandbox import create_apptainer_sandbox
from deepagents import create_deep_agent

with create_apptainer_sandbox(
    image="docker://python:3.13-slim",
    working_dir="/workspace",
    auto_remove=True,
) as backend:
    agent = create_deep_agent(backend=backend)
    result = agent.invoke({"messages": [...]})
# Automatic cleanup when exiting context
```

## Setup Scripts

Both factory functions support setup scripts that run after the sandbox starts:

```python
# Create setup script
setup_script = """#!/bin/bash
pip install numpy pandas matplotlib
setupATLAS  # For ATLAS containers
"""

with create_apptainer_sandbox(
    image="docker://atlas/athanalysis:latest",
    setup_script_path="/path/to/setup.sh",
) as backend:
    # Sandbox is ready with packages installed
    agent = create_deep_agent(backend=backend)
```

Setup scripts support environment variable expansion using `${VAR}` syntax:

```bash
#!/bin/bash
pip install ${REQUIRED_PACKAGES}
export ATLAS_VERSION=${ATLAS_VERSION}
```

## Integration with DeepAgents

### Using with create_deep_agent

```python
from chatlas_agents.sandbox import create_docker_sandbox
from deepagents import create_deep_agent
from deepagents.llm import init_chat_model

with create_docker_sandbox() as backend:
    model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")

    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt="You are an AI assistant with access to a sandbox environment.",
    )
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Create and run a Python script to analyze data"
        }]
    })
```

The agent automatically gets access to:
- File system tools (ls, read, write, edit, grep, glob)
- Execute tool for running shell commands
- Safe isolation from the host system

### Using with chatlas_agents.graph

```python
from chatlas_agents.config import AgentConfig, LLMConfig
from chatlas_agents.graph import create_chatlas_deep_agent
from chatlas_agents.sandbox import SandboxBackendType

config = AgentConfig(
    name="my-agent",
    llm=LLMConfig(provider="openai", model="gpt-5-mini"),
)

agent = await create_chatlas_deep_agent(
    config,
    use_docker_sandbox=True,
    docker_image="python:3.13-slim",
    sandbox_backend=SandboxBackendType.APPTAINER,  # or DOCKER
)
```

## ATLAS-Specific Usage

For ATLAS physics analysis on lxplus:

```python
from chatlas_agents.sandbox import create_apptainer_sandbox
from deepagents import create_deep_agent

# Use ATLAS software container
with create_apptainer_sandbox(
    image="docker://atlas/athanalysis:latest",
    working_dir="/workspace",
    setup_script_path="/path/to/setup_atlas.sh",
) as backend:
    agent = create_deep_agent(
        backend=backend,
        system_prompt="""You are an ATLAS physics analysis assistant.
        You have access to the ATLAS software environment.
        Use athena, ROOT, and other ATLAS tools as needed."""
    )
    
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Run an athena job on the test dataset"
        }]
    })
```

Example `setup_atlas.sh`:
```bash
#!/bin/bash
setupATLAS
asetup Athena,22.0.25
source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
```

## Comparison with deepagents-cli

Our implementation follows the same patterns as deepagents-cli for consistency:

| Feature | deepagents-cli | chatlas-agents |
|---------|---------------|----------------|
| Base class | BaseSandbox | BaseSandbox ✅ |
| Context managers | ✅ | ✅ |
| Setup scripts | ✅ | ✅ |
| Env var expansion | ✅ | ✅ |
| Auto cleanup | ✅ | ✅ |
| Cloud providers | Modal, Runloop, Daytona | Docker, Apptainer |

## Examples

See the `examples/` directory for complete examples:

- **`sandbox_factory_example.py`** - Factory function patterns
- **`docker_sandbox_example.py`** - Docker-specific examples
- **`apptainer_sandbox_example.py`** - Apptainer-specific examples

## Testing

Run tests with:

```bash
# All sandbox tests
uv run pytest libs/chatlas-agents/tests/test_sandbox.py

# Specific backend (skips if not installed)
uv run pytest libs/chatlas-agents/tests/test_sandbox.py -k docker
uv run pytest libs/chatlas-agents/tests/test_sandbox.py -k apptainer
```

## Troubleshooting

### Docker Issues

**Container fails to start:**
- Check Docker daemon is running: `docker ps`
- Verify image exists: `docker images`
- Check permissions: User must be in `docker` group

**File operations fail:**
- Ensure working directory exists in container
- Check file permissions
- Verify paths are absolute

### Apptainer Issues

**Instance creation fails:**
- Check Apptainer is installed: `apptainer --version`
- Verify image URL format: `docker://...` for Docker images
- Check network access for pulling images

**Setup script fails:**
- Verify script has bash shebang: `#!/bin/bash`
- Check script permissions: `chmod +x script.sh`
- Examine error output in logs

## Best Practices

1. **Always use context managers** (factory functions) for automatic cleanup
2. **Use setup scripts** for complex initialization instead of manual commands
3. **Specify working_dir** explicitly for clarity
4. **Use absolute paths** in file operations
5. **Set auto_remove=True** for ephemeral sandboxes
6. **Log liberally** for debugging sandbox issues
7. **Test locally with Docker** before deploying to HPC with Apptainer

## Future Enhancements

Potential improvements for future versions:

- [ ] Integration with deepagents-cli's sandbox factory registry
- [ ] Support for persistent volumes/mounts
- [ ] Multi-container orchestration
- [ ] Resource limits (CPU, memory)
- [ ] Network isolation options
- [ ] Custom image builders
- [ ] Sandbox pooling/reuse

## References

- [deepagents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [deepagents-cli Sandbox Factory](../../deepagents-cli/deepagents_cli/integrations/sandbox_factory.py)
- [Apptainer Documentation](https://apptainer.org/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [ATLAS Software Documentation](https://atlassoftwaredocs.web.cern.ch/)
