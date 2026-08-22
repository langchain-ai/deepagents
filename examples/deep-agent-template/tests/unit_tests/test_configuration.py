from langgraph.pregel import Pregel

from deep_agent.graph import RO_AGENT, SUBAGENTS, SYSTEM_PROMPT


def test_read_only_agent_compiles() -> None:
    assert isinstance(RO_AGENT, Pregel)


def test_subagents_configured() -> None:
    names = {item["name"] for item in SUBAGENTS}
    assert names == {"researcher", "critic"}


def test_system_prompt_is_nonempty() -> None:
    assert len(SYSTEM_PROMPT.strip()) > 0
