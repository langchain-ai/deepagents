"""Unit tests for deepagents.graph module."""

from unittest.mock import call, patch

from langchain_core.messages import AIMessage

from deepagents._version import __version__
from deepagents.graph import create_deep_agent
from deepagents.middleware.summarization import create_summarization_middleware
from tests.unit_tests.chat_model import GenericFakeChatModel


class TestCreateDeepAgentMetadata:
    """Tests for metadata on the compiled graph."""

    def test_versions_metadata_contains_sdk_version(self) -> None:
        """`create_deep_agent` should attach SDK version in metadata.versions."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = create_deep_agent(model=model)
        assert agent.config is not None
        versions = agent.config["metadata"]["versions"]
        assert versions["deepagents"] == __version__

    def test_ls_integration_metadata_preserved(self) -> None:
        """`ls_integration` should still be present alongside versions."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = create_deep_agent(model=model)
        assert agent.config is not None
        assert agent.config["metadata"]["ls_integration"] == "deepagents"


class TestCreateDeepAgentSummarizerModel:
    """Tests for the summarizer_model parameter of create_deep_agent()."""

    def test_default_uses_main_model(self) -> None:
        """Without summarizer_model, all summarization calls use the main model."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        with patch("deepagents.graph.create_summarization_middleware", wraps=create_summarization_middleware) as mock_factory:
            create_deep_agent(model=model)
        assert all(c.args[0] is model for c in mock_factory.call_args_list)

    def test_override_uses_summarizer_model(self) -> None:
        """When summarizer_model is provided, all summarization calls use it."""
        main_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        cheap_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        with patch("deepagents.graph.create_summarization_middleware", wraps=create_summarization_middleware) as mock_factory:
            create_deep_agent(model=main_model, summarizer_model=cheap_model)
        assert len(mock_factory.call_args_list) > 0
        assert all(c.args[0] is cheap_model for c in mock_factory.call_args_list)

    def test_string_summarizer_model_is_resolved(self) -> None:
        """A string summarizer_model is passed through resolve_model."""
        main_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        cheap_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        with patch("deepagents.graph.resolve_model", return_value=cheap_model) as mock_resolve:
            with patch("deepagents.graph.create_summarization_middleware", wraps=create_summarization_middleware):
                create_deep_agent(model=main_model, summarizer_model="anthropic:claude-haiku-4-5")
        assert call("anthropic:claude-haiku-4-5") in mock_resolve.call_args_list

    def test_summarizer_model_applies_to_subagents(self) -> None:
        """summarizer_model is used for declarative subagent summarization too."""
        main_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        cheap_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        subagent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        subagent_spec = {
            "name": "my-subagent",
            "description": "A test subagent",
            "model": subagent_model,
            "tools": [],
            "system_prompt": "You are a test subagent.",
        }
        with patch("deepagents.graph.create_summarization_middleware", wraps=create_summarization_middleware) as mock_factory:
            create_deep_agent(model=main_model, summarizer_model=cheap_model, subagents=[subagent_spec])
        assert len(mock_factory.call_args_list) > 0
        assert all(c.args[0] is cheap_model for c in mock_factory.call_args_list)
