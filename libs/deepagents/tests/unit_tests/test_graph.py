"""Unit tests for deepagents.graph module."""

from langchain_core.messages import AIMessage

from deepagents._version import __version__
from deepagents.graph import create_deep_agent

from .chat_model import GenericFakeChatModel


def test_create_deep_agent_includes_version_metadata() -> None:
    """`create_deep_agent` should nest SDK version under `metadata.versions`."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    agent = create_deep_agent(model=model)
    assert agent.config is not None
    versions = agent.config["metadata"]["versions"]
    assert versions["deepagents"] == __version__
