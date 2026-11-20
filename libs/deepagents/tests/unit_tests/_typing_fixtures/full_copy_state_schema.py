"""Type-checking fixture for full-copy state schemas on `create_deep_agent`.

This module is not executed; it is type-checked by `ty` from
`test_graph.py::TestStateSchemaTyping`. `state_schema` must accept a schema
that extends `AgentState` without extending `DeepAgentState`.
"""

from deepagents import FullCopyDeepAgentState, create_deep_agent


class FullCopyStateWithExtraField(FullCopyDeepAgentState):
    """Full-copy state with one field added by the caller."""

    page_url: str


create_deep_agent(state_schema=FullCopyDeepAgentState)
create_deep_agent(state_schema=FullCopyStateWithExtraField)
