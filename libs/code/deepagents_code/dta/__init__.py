"""Dynamic Tool Allocation (DTA) package.

Provides middleware, indexing, namespace gating, and LLM-based tool selection
to keep the agent's active toolset within a configurable budget.
"""

from deepagents_code.dta.gating import ToolNamespaceRegistry, ToolNamespaceRouterNode
from deepagents_code.dta.indexer import HybridToolIndexer, ToolCandidate
from deepagents_code.dta.middleware import DynamicToolAllocationMiddleware
from deepagents_code.dta.selector import ToolSelectorNode

__all__ = [
    "DynamicToolAllocationMiddleware",
    "HybridToolIndexer",
    "ToolCandidate",
    "ToolNamespaceRegistry",
    "ToolNamespaceRouterNode",
    "ToolSelectorNode",
]
