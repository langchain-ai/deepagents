"""Unit tests for deepagents.graph module."""

from langchain_core.messages import AIMessage

from deepagents._version import __version__
from deepagents.graph import create_deep_agent

from .chat_model import GenericFakeChatModel


def test_create_deep_agent_includes_version_metadata() -> None:
    """`create_deep_agent` should attach `deepagents_version` to graph config metadata."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    agent = create_deep_agent(model=model)
    assert agent.config is not None
    assert agent.config["metadata"]["deepagents_version"] == __version__
