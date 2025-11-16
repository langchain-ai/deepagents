"""Meta-Agent Builder - Automated project specification generator."""

from meta_agent_builder.orchestrator import MetaOrchestrator
from meta_agent_builder.specialists import (
    ArchitectureSpecialist,
    BaseSpecialist,
    DocumentationSpecialist,
)

__version__ = "0.1.0"

__all__ = [
    "MetaOrchestrator",
    "BaseSpecialist",
    "DocumentationSpecialist",
    "ArchitectureSpecialist",
]
