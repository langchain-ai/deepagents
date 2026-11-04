"""dcode compaction keeps native provider context opaque."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.native_compaction import OpenAICompactionMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.offload_middleware import CLICompactionMiddleware


def _native() -> MagicMock:
    engine = MagicMock(spec=OpenAICompactionMiddleware)
    engine.compaction_strategy = "openai"
    return engine


def test_forced_update_preserves_native_message():
    engine = _native()
    opaque = AIMessage(content=[{"type": "compaction", "encrypted_content": "opaque"}])
    engine._build_new_messages_with_path.return_value = [opaque]
    update = CLICompactionMiddleware._forced_compaction_update(
        engine, "serialized", None, 2, "session"
    )
    assert update["_summarization_event"]["summary_message"] is opaque
    assert update["_summarization_event"]["cutoff_index"] == 2


async def test_forced_native_handoff_refuses_lossy_text_conversion(monkeypatch):
    engine = _native()
    compaction = CLICompactionMiddleware(engine)
    monkeypatch.setattr(compaction, "_summarization_for_runtime", lambda _: engine)
    with pytest.raises(ValueError, match="fork the checkpoint"):
        await compaction._aplan_forced_compaction_update(
            {"messages": [HumanMessage(content="work")]},
            SimpleNamespace(context=None),
            handoff=True,
        )
    engine._acreate_summary.assert_not_called()


async def test_async_auto_gate_uses_native_count_without_sync_call():
    engine = _native()
    engine._get_effective_messages.return_value = [HumanMessage(content="work")]
    engine.acount_request_tokens = AsyncMock(return_value=100)
    engine._should_summarize.return_value = True
    engine._determine_cutoff_index.return_value = 1
    request = MagicMock()
    await CLICompactionMiddleware._aauto_compaction_request(request, engine)
    engine.acount_request_tokens.assert_awaited_once_with(request)
    engine.count_request_tokens.assert_not_called()


def test_runtime_native_summary_model_override_is_rejected():
    from deepagents_code._cli_context import CLIContext

    compaction = CLICompactionMiddleware(_native())
    runtime = SimpleNamespace(context=CLIContext(summarization_model="openai:other"))
    with pytest.raises(ValueError, match="separate summarization model"):
        compaction._summarization_for_runtime(runtime)


def test_native_factory_explicitly_gates_missing_sdk(monkeypatch):
    from deepagents_code import config_manifest, offload_middleware

    monkeypatch.setattr(config_manifest, "get_option", lambda _: object())
    monkeypatch.setattr(
        config_manifest, "_resolve_option", lambda _: SimpleNamespace(value="openai")
    )

    def missing_sdk(_) -> None:
        raise ModuleNotFoundError

    monkeypatch.setattr(offload_middleware, "import_module", missing_sdk)
    with pytest.raises(ValueError, match="installed SDK is unsupported"):
        offload_middleware._create_cli_compaction_middleware(
            FakeListChatModel(responses=["unused"]), StateBackend()
        )


def test_default_factory_does_not_require_native_sdk(monkeypatch):
    from deepagents_code import config_manifest
    from deepagents_code.offload_middleware import _create_cli_compaction_middleware

    monkeypatch.setattr(config_manifest, "get_option", lambda _: None)
    compaction = _create_cli_compaction_middleware(
        FakeListChatModel(responses=["summary"]), StateBackend()
    )
    assert getattr(compaction._summarization, "compaction_strategy", "text") == "text"
