"""Custom tools for Meta-Agent Builder specialists."""

from meta_agent_builder.tools.architecture_tools import (
    create_mermaid_diagram,
    suggest_middleware_stack,
    validate_agent_hierarchy,
)
from meta_agent_builder.tools.documentation_tools import (
    extract_code_examples,
    internet_search,
    summarize_documentation,
)

__all__ = [
    # Documentation tools
    "internet_search",
    "extract_code_examples",
    "summarize_documentation",
    # Architecture tools
    "create_mermaid_diagram",
    "validate_agent_hierarchy",
    "suggest_middleware_stack",
]
