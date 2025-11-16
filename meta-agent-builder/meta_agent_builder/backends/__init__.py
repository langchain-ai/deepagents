"""Backend configuration for Meta-Agent Builder."""

from meta_agent_builder.backends.composite_config import (
    create_backend_with_sandbox,
    create_meta_agent_backend,
)

__all__ = [
    "create_meta_agent_backend",
    "create_backend_with_sandbox",
]
