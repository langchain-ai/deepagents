"""Integration coverage for resumed-thread compaction."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _write_model_config(home_dir: Path) -> None:
    """Write a temp config that points the server subprocess at the test model."""
    config_dir = home_dir / ".deepagents"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        """
[models.providers.itest]
class_path = "deepagents_cli._testing_models:DeterministicIntegrationChatModel"
models = ["fake"]
""".strip()
        + "\n"
    )


def _build_long_prompt(turn: int) -> str:
    """Build a long user message so the seeded thread is worth compacting."""
    sentence = (
        f"Turn {turn} keeps enough unique detail to make resume-compaction meaningful. "
        "The quick brown fox documents repeatable integration behavior for the CLI. "
    )
    return sentence * 30


async def _run_turn(agent, *, thread_id: str, assistant_id: str, prompt: str) -> None:
    """Execute one real remote agent turn and drain the stream to completion."""
    from deepagents_cli.config import build_stream_config

    config = build_stream_config(thread_id, assistant_id)
    stream_input = {"messages": [{"role": "user", "content": prompt}]}
    async for _chunk in agent.astream(
        stream_input,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        pass


def _event_field(event: object, key: str) -> object | None:
    """Read a summarization-event field from either dict or object form."""
    if isinstance(event, dict):
        return event.get(key)  # ty: ignore[invalid-argument-type]
    return getattr(event, key, None)


# https://github.com/langchain-ai/deepagents/issues/2624
@pytest.mark.xfail(
    condition=os.environ.get("CI") == "true",
    strict=True,
    reason="Polling loop too tight for slow CI runners (#2624)",
)
@pytest.mark.timeout(180)
async def test_compact_resumed_thread_uses_persisted_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compacts a resumed thread after restart using SQLite-backed history.

    The test seeds a real persisted thread on one server instance, restarts the
    server, resumes that thread in a fresh `DeepAgentsApp`, and verifies that
    `/compact` succeeds even when the resumed history comes from the
    checkpointer rather than live in-memory server state.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    backend_root = tmp_path / "compact_backend"
    assistant_id = "itest-compact"

    home_dir.mkdir()
    project_dir.mkdir()
    backend_root.mkdir()

    # Keep config and the global sessions DB fully test-local.
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CLI_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.filesystem import FilesystemBackend

    from deepagents_cli import model_config
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.config import create_model
    from deepagents_cli.server_manager import server_session
    from deepagents_cli.sessions import generate_thread_id, thread_exists
    from deepagents_cli.widgets.messages import AppMessage, ErrorMessage

    config_path = home_dir / ".deepagents" / "config.toml"
    # Some tests import `model_config` earlier in the session, so override the
    # cached default paths explicitly before creating the model.
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model("itest:fake").apply_to_settings()
        thread_id = generate_thread_id()

        # Server 1: create a real persisted thread with enough content to
        # trigger compaction later.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            for turn in range(1, 5):
                await _run_turn(
                    agent,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    prompt=_build_long_prompt(turn),
                )

        assert await thread_exists(thread_id)

        compact_backend = CompositeBackend(
            default=FilesystemBackend(root_dir=backend_root, virtual_mode=True),
            routes={},
        )

        # Server 2: same SQLite DB, but a fresh server process and empty
        # in-memory thread registry.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            config = {"configurable": {"thread_id": thread_id}}
            actual_state = await agent.aget_state(config)
            actual_values = getattr(actual_state, "values", None) or {}

            # Fresh dev servers may return empty state for persisted threads
            # after restart. If that behavior changes upstream, force the
            # empty-state precondition so this test still covers the SQLite
            # fallback path.
            if actual_values:
                agent.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))  # ty: ignore[invalid-assignment]

            app = DeepAgentsApp(
                agent=agent,  # ty: ignore[invalid-argument-type]
                assistant_id=assistant_id,
                backend=compact_backend,
                cwd=project_dir,
                thread_id=thread_id,
            )

            async with app.run_test() as pilot:
                # Let startup history loading settle before asserting on the UI.
                for _ in range(60):
                    await pilot.pause()
                    if app._message_store.total_count > 0:
                        break

                assert app._message_store.total_count > 0
                resume_messages = app.query(AppMessage)
                assert any(
                    "Resumed thread:" in str(widget._content)
                    for widget in resume_messages
                )

                await app._handle_offload()

                # `/compact` posts a success message after the async state write
                # and archive offload finish.
                for _ in range(60):
                    await pilot.pause()
                    if any(
                        "Conversation compacted." in str(widget._content)
                        for widget in app.query(AppMessage)
                    ):
                        break

                app_messages = [
                    str(widget._content) for widget in app.query(AppMessage)
                ]
                error_messages = [
                    str(widget._content) for widget in app.query(ErrorMessage)
                ]

            assert "Nothing to compact" not in "\n".join(app_messages)
            assert any("Conversation compacted." in content for content in app_messages)
            assert not error_messages

            # The summarization event must be checkpointed so subsequent turns
            # see compacted context instead of the full message history.
            channel_values = await DeepAgentsApp._read_channel_values_from_checkpointer(
                thread_id
            )
            summarization_event = channel_values.get("_summarization_event")
            assert summarization_event is not None
            cutoff = _event_field(summarization_event, "cutoff_index")
            assert isinstance(cutoff, int)
            assert cutoff > 0
            assert (
                _event_field(summarization_event, "file_path")
                == f"/conversation_history/{thread_id}.md"
            )

        # The offloaded archive should land in the explicit temp-backed backend,
        # not the host filesystem root.
        archive_path = backend_root / "conversation_history" / f"{thread_id}.md"
        assert archive_path.exists()
        archive_text = archive_path.read_text()
        assert "Compacted at" in archive_text
        assert "keeps enough unique detail" in archive_text
    finally:
        model_config.clear_caches()
