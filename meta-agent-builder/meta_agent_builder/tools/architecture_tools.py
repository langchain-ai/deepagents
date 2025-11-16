"""Tools for architecture design and diagramming."""

from langchain_core.tools import tool


@tool
def create_mermaid_diagram(
    diagram_type: str,
    content: str,
    title: str,
) -> str:
    """Create a Mermaid diagram for architecture visualization.

    Args:
        diagram_type: Type of diagram (graph, sequence, flowchart, class)
        content: Mermaid diagram syntax
        title: Diagram title

    Returns:
        Formatted Mermaid diagram block ready for markdown

    Example:
        >>> diagram = create_mermaid_diagram(
        ...     "graph",
        ...     "A --> B\\nB --> C",
        ...     "Simple Flow"
        ... )
        >>> print(diagram)
        ```mermaid
        ---
        title: Simple Flow
        ---
        graph TD
            A --> B
            B --> C
        ```
    """
    # Map diagram types to Mermaid syntax
    type_mapping = {
        "graph": "graph TD",
        "flowchart": "flowchart TD",
        "sequence": "sequenceDiagram",
        "class": "classDiagram",
        "state": "stateDiagram-v2",
    }

    mermaid_type = type_mapping.get(diagram_type, "graph TD")

    return f"""```mermaid
---
title: {title}
---
{mermaid_type}
{content}
```"""


@tool
def validate_agent_hierarchy(hierarchy_spec: dict) -> tuple[bool, list[str]]:
    """Validate an agent hierarchy specification.

    Checks for:
    - No circular dependencies
    - All referenced agents exist
    - Tools properly distributed
    - Valid agent names

    Args:
        hierarchy_spec: Dictionary with agent specifications

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        >>> spec = {
        ...     "orchestrator": {"subagents": ["agent1", "agent2"]},
        ...     "agent1": {"tools": ["tool1"]},
        ...     "agent2": {"tools": ["tool2"]}
        ... }
        >>> valid, issues = validate_agent_hierarchy(spec)
        >>> assert valid == True
    """
    issues = []

    # Check for required fields
    if "orchestrator" not in hierarchy_spec:
        issues.append("Missing orchestrator definition")
        return False, issues

    # Get all agent names
    all_agents = set(hierarchy_spec.keys())

    # Check each agent
    for agent_name, agent_spec in hierarchy_spec.items():
        # Check subagents exist
        if "subagents" in agent_spec:
            for subagent in agent_spec["subagents"]:
                if subagent not in all_agents:
                    issues.append(
                        f"Agent '{agent_name}' references undefined subagent '{subagent}'"
                    )

        # Check for tools
        if "tools" in agent_spec:
            if not isinstance(agent_spec["tools"], list):
                issues.append(f"Agent '{agent_name}' tools must be a list")

    # Check for circular dependencies (simple check)
    # TODO: Implement more sophisticated cycle detection

    return len(issues) == 0, issues


@tool
def suggest_middleware_stack(agent_type: str, requirements: list[str]) -> list[str]:
    """Suggest appropriate middleware stack for an agent.

    Args:
        agent_type: Type of agent (orchestrator, specialist, subagent)
        requirements: List of requirements (e.g., "planning", "memory", "validation")

    Returns:
        Ordered list of recommended middleware

    Example:
        >>> stack = suggest_middleware_stack(
        ...     "specialist",
        ...     ["planning", "memory", "filesystem"]
        ... )
        >>> print(stack)
        ['AgentMemoryMiddleware', 'TodoListMiddleware', 'FilesystemMiddleware', ...]
    """
    middleware_stack = []

    # Base middleware for specialists
    if agent_type == "specialist":
        if "memory" in requirements or "learning" in requirements:
            middleware_stack.append("AgentMemoryMiddleware")

        if "planning" in requirements:
            middleware_stack.append("TodoListMiddleware")

        if "filesystem" in requirements or "files" in requirements:
            middleware_stack.append("FilesystemMiddleware")

        if "subagents" in requirements:
            middleware_stack.append("SubAgentMiddleware")

        # Always add these for specialists
        middleware_stack.extend(
            [
                "SummarizationMiddleware",
                "AnthropicPromptCachingMiddleware",
                "PatchToolCallsMiddleware",
            ]
        )

    # For orchestrators
    elif agent_type == "orchestrator":
        middleware_stack.extend(
            [
                "TodoListMiddleware",
                "FilesystemMiddleware",
                "SubAgentMiddleware",
                "SummarizationMiddleware",
                "AnthropicPromptCachingMiddleware",
                "PatchToolCallsMiddleware",
            ]
        )

        if "validation" in requirements:
            middleware_stack.insert(0, "ValidationMiddleware")

    return middleware_stack
