"""Harbor integration with LangChain DeepAgents.

This package provides a complete integration between Harbor (agent benchmarking)
and LangChain DeepAgents (agent framework) with automatic LangSmith tracing.

Quick Start:
    >>> from harbor_deepagents.agents import DeepAgentHarbor
    >>> agent = DeepAgentHarbor(
    ...     logs_dir=Path("logs"),
    ...     model_name="anthropic/claude-sonnet-4-5-20250929"
    ... )

Example Usage with Harbor:
    In configs/job.yaml:
    ```yaml
    agents:
      - import_path: harbor_deepagents.agents:DeepAgentHarbor
        model_name: anthropic/claude-sonnet-4-5-20250929
    ```

    Then run:
    ```bash
    harbor run --config configs/job.yaml
    ```

Features:
    - DeepAgents framework with planning, filesystem, and subagents
    - Automatic LangSmith tracing (zero config)
    - ATIF trajectory format for RL/fine-tuning
    - Works with any Harbor environment (Docker, Modal, E2B, etc.)

See Also:
    - Harbor: https://harborframework.com
    - DeepAgents: https://github.com/langchain-ai/deepagents
    - LangSmith: https://smith.langchain.com
"""

__version__ = "1.0.0"

from harbor_deepagents.agents import DeepAgentHarbor
from harbor_deepagents.utils import LangSmithTrajectoryExporter

__all__ = [
    "DeepAgentHarbor",
    "LangSmithTrajectoryExporter",
]
