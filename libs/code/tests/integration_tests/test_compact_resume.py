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
        return event.get(key)  # ty: ignore
    return getattr(event, key, None)


async def _read_file_through_agent(agent, *, thread_id: str, file_path: str) -> str:
    """Read `file_path` via the running agent's own `read_file` tool.

    Seeds a `read_file` tool call attributed to the model node and advances the
    graph so the agent's `ToolNode` executes the read against its own backend,
    proving the offloaded archive exists server-side (not in a client dir).
    Auto-approves any HITL interrupt the read raises.
    """
    import uuid

    from langchain.agents.middleware.human_in_the_loop import ApproveDecision
    from langchain_core.messages import AIMessage
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    tool_call_id = str(uuid.uuid4())
    seed = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"file_path": file_path}, "id": tool_call_id}
        ],
    )
    await agent.aensure_thread(config)
    await agent.aupdate_state(config, {"messages": [seed]}, as_node="model")

    interrupt_ids: list[str] = []
    tool_contents: list[str] = []

    async def _drain(stream_input) -> None:
        async for chunk in agent.astream(
            stream_input,
            stream_mode=["messages", "updates"],
            subgraphs=True,
            config=config,
            durability="exit",
        ):
            if not isinstance(chunk, tuple) or len(chunk) != 3:
                continue
            _ns, mode, data = chunk
            if mode == "updates" and isinstance(data, dict):
                for interrupt_obj in data.get("__interrupt__", []) or []:
                    iid = getattr(interrupt_obj, "id", None)
                    if iid:
                        interrupt_ids.append(iid)
            elif mode == "messages" and isinstance(data, tuple):
                msg = data[0]
                if type(msg).__name__ == "ToolMessage":
                    tool_contents.append(str(getattr(msg, "content", "")))

    await _drain(None)
    if interrupt_ids:
        resume = {
            iid: {"decisions": [ApproveDecision(type="approve")]}
            for iid in interrupt_ids
        }
        await _drain(Command(resume=resume))

    return "\n".join(tool_contents)


@pytest.mark.timeout(180)
async def test_compact_resumed_thread_uses_persisted_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offloads a resumed thread after restart using remote server state.

    The test seeds a real persisted thread on one server instance, restarts the
    server, resumes that thread in a fresh `DeepAgentsApp` constructed the
    PRODUCTION way (`backend=None`), and verifies that `/offload` succeeds
    server-side and the archive stays readable through the agent's own backend.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    assistant_id = "itest-compact"

    home_dir.mkdir()
    project_dir.mkdir()

    # Keep config and the global sessions DB fully test-local.
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents_code import model_config
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.client.launch.server_manager import server_session
    from deepagents_code.config import create_model
    from deepagents_code.sessions import generate_thread_id
    from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage

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

            # Production construction: no client-owned backend. Offload runs
            # server-side through the agent's own `compact_conversation` tool.
            app = DeepAgentsApp(
                agent=agent,  # ty: ignore
                assistant_id=assistant_id,
                backend=None,
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
            # In local mode the history prefix lives under a stable per-user
            # `artifacts_root`, so assert the suffix rather than a fixed prefix.
            # The path stays resolvable after restart because `artifacts_root`
            # is deterministic (Server 3 below reuses it).
            archive_path = _event_field(summarization_event, "file_path")
            assert isinstance(archive_path, str)
            assert archive_path.endswith(f"/conversation_history/{thread_id}.md")

            # The archive must be readable THROUGH THE AGENT, proving the bytes
            # live in the agent's own composite backend server-side rather than
            # in a client-local directory the server can never read.
            read_back = await _read_file_through_agent(
                agent, thread_id=thread_id, file_path=archive_path
            )
            assert "keeps enough unique detail" in read_back
            assert "Summarized at" in read_back

        # Server 3: the event and archive path must remain usable after the
        # process that performed the offload has exited.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            read_back = await _read_file_through_agent(
                agent, thread_id=thread_id, file_path=archive_path
            )
            assert "keeps enough unique detail" in read_back
            assert "Summarized at" in read_back
    finally:
        model_config.clear_caches()
