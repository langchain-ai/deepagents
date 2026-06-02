"""Integration coverage for resumed-thread compaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
class_path = "deepagents_code._testing_models:DeterministicIntegrationChatModel"
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
    from deepagents_code.config import build_stream_config

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


@pytest.mark.timeout(180)
async def test_compact_resumed_thread_uses_persisted_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offloads a resumed thread after restart using remote server state.

    The test seeds a real persisted thread on one server instance, restarts the
    server, resumes that thread in a fresh `DeepAgentsApp`, and verifies that
    `/offload` succeeds after the app registers the thread with the server.
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
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.filesystem import FilesystemBackend

    from deepagents_code import model_config
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.config import create_model
    from deepagents_code.server_manager import server_session
    from deepagents_code.sessions import generate_thread_id
    from deepagents_code.widgets.messages import AppMessage, ErrorMessage

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

        compact_backend = CompositeBackend(
            default=FilesystemBackend(root_dir=backend_root, virtual_mode=True),
            routes={},
        )

        # Server 2: same SQLite DB, but a fresh server process.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            config = {"configurable": {"thread_id": thread_id}}

            app = DeepAgentsApp(
                agent=agent,  # ty: ignore[invalid-argument-type]
                assistant_id=assistant_id,
                backend=compact_backend,
                cwd=project_dir,
                thread_id=thread_id,
            )

            async with app.run_test() as pilot:
                # Let startup history loading settle before asserting on the UI.
                # Use a 0.1 s delay per iteration (up to 12 s) so slow CI
                # runners have enough time for the async I/O to complete.
                for _ in range(120):
                    await pilot.pause(0.1)
                    if app._message_store.total_count > 0:
                        break

                assert app._message_store.total_count > 0

                await app._handle_offload()

                # `/offload` posts a success message after the async state write
                # and archive offload finish.
                for _ in range(120):
                    await pilot.pause(0.1)
                    if any(
                        "Offloaded " in str(widget._content)
                        for widget in app.query(AppMessage)
                    ):
                        break

                app_messages = [
                    str(widget._content) for widget in app.query(AppMessage)
                ]
                error_messages = [
                    str(widget._content) for widget in app.query(ErrorMessage)
                ]

            assert "Nothing to offload" not in "\n".join(app_messages)
            assert any("Offloaded " in content for content in app_messages)
            assert not error_messages

            # The summarization event must be visible through server state so
            # subsequent turns see compacted context instead of full history.
            state = await agent.aget_state(config)
            values = getattr(state, "values", None) or {}
            summarization_event = values.get("_summarization_event")
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
        assert "Offloaded at" in archive_text
        assert "keeps enough unique detail" in archive_text
    finally:
        model_config.clear_caches()
