"""Integration coverage for the server-side `/offload` path.

For a server-backed agent `/offload` runs the dedicated `offload` operation
graph, which compacts without a model node or a synthetic tool call. Either way
the offloaded archive lands in the agent's composite backend and is readable via
`read_file` in every run mode — not in a client-local directory the server can
never read. These tests construct the app the PRODUCTION way (`backend=None`)
and prove the archive is readable *through the agent*.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _write_model_config(home_dir: Path) -> None:
    """Write a temp config that points the server subprocess at the test model.

    The fake model's 8k-token default profile overflows once the system
    prompt plus two seeded long turns cross the 85% auto-compaction trigger,
    so auto-compaction fires during seeding and leaves `/offload` nothing
    genuine to compact. Widening the window past the seeded size keeps the
    thread uncompacted until `/offload`, while the fraction-based retention
    window (~800 tokens) stays smaller than the seeded ~4.4k, so the forced
    compaction still has real work to do.
    """
    config_dir = home_dir / ".deepagents"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        """
[models.providers.itest]
class_path = "deepagents_code._testing_models:DeterministicIntegrationChatModel"
models = ["fake"]

[models.providers.itest.profile]
max_input_tokens = 32000
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
    from deepagents_code.config import build_stream_config, settings

    config = build_stream_config(thread_id, assistant_id)
    stream_input = {"messages": [{"role": "user", "content": prompt}]}
    # Send the resolved context limit so the server's compaction/summarization
    # layers see the same window the model profile was widened to; without it
    # the server falls back to its own default and auto-compaction fires early.
    async for _chunk in agent.astream(
        stream_input,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        config=config,
        context={"model_context_limit": settings.model_context_limit},
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
    graph so the agent's `ToolNode` executes the read against its own backend.
    This proves the offloaded archive exists server-side (not merely in a
    client-local directory). Auto-approves any HITL interrupt the read raises.

    Returns:
        The concatenated content of every `ToolMessage` produced by the run.
    """
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
    # `/offload` restores the thread's main-graph association before returning,
    # so this seeds the read straight through the interactive `agent` graph's
    # model node. Seeding against that client (rather than the app's default
    # graph) also keeps the test pinned to the interactive graph `/offload`
    # shares its checkpoint with.
    agent_graph = agent.for_graph("agent")
    await agent_graph.aensure_thread(config)
    await agent_graph.aupdate_state(config, {"messages": [seed]}, as_node="model")

    interrupt_ids: list[str] = []
    tool_contents: list[str] = []

    async def _drain(stream_input) -> None:
        async for chunk in agent_graph.astream(
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


@pytest.mark.timeout(240)
async def test_offload_runs_server_side_and_is_agent_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/offload` compacts server-side with `backend=None` and stays readable.

    Constructs the app the production way (`backend=None`), seeds a thread with
    enough content, runs `/offload`, and asserts:

    - no `ErrorMessage` and an "Offloaded " success message,
    - no HITL interrupt is surfaced, the operation graph having no tool node,
    - a persisted `_summarization_event` with `cutoff > 0` and
      `file_path == /conversation_history/<thread>.md`,
    - the archive is readable THROUGH THE AGENT (via its own `read_file` tool),
      proving the bytes live in the agent's backend server-side, and
    - local archives land in the persistent per-user history directory.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    assistant_id = "itest-offload"

    home_dir.mkdir()
    project_dir.mkdir()

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
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model("itest:fake").apply_to_settings()
        thread_id = generate_thread_id()

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

            config = {"configurable": {"thread_id": thread_id}}

            # Captured before the run so the replay can be checked against it.
            # The `/offload` run input is *authoritative* for the `messages`
            # channel against a real server -- it replaces the conversation
            # rather than merging into it (streaming `{"messages": []}` here
            # empties the thread outright). No unit test can observe that: an
            # in-process checkpointer honors the `add_messages` reducer and
            # leaves the checkpointed list intact either way.
            before_state = await agent.aget_state(config)
            messages_before = list(
                (getattr(before_state, "values", None) or {}).get("messages", [])
            )
            assert messages_before

            # Production construction: no client-owned backend.
            app = DeepAgentsApp(
                agent=agent,  # ty: ignore
                assistant_id=assistant_id,
                backend=None,
                cwd=project_dir,
                thread_id=thread_id,
            )

            offload_interrupts: list[object] = []
            recorded_chunks = 0
            plain_for_graph = agent.for_graph

            def _recording_for_graph(graph_id: str):  # noqa: ANN202
                """Instrument the `offload` client `/offload` actually streams."""
                offload_client = plain_for_graph(graph_id)
                plain_astream = offload_client.astream

                async def _recording_astream(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                    """Record every interrupt the server surfaces to the client."""
                    nonlocal recorded_chunks
                    async for chunk in plain_astream(*args, **kwargs):
                        if isinstance(chunk, tuple) and len(chunk) == 3:
                            recorded_chunks += 1
                            _ns, mode, data = chunk
                            if mode == "updates" and isinstance(data, dict):
                                offload_interrupts.extend(
                                    data.get("__interrupt__") or []
                                )
                        yield chunk

                offload_client.astream = _recording_astream  # ty: ignore
                return offload_client

            async with app.run_test() as pilot:
                for _ in range(120):
                    await pilot.pause(0.1)
                    if app._message_store.total_count > 0:
                        break

                assert app._message_store.total_count > 0

                agent.for_graph = _recording_for_graph  # ty: ignore
                try:
                    await app._handle_offload()

                    for _ in range(120):
                        await pilot.pause(0.1)
                        if any(
                            "Offloaded " in str(widget._content)
                            for widget in app.query(AppMessage)
                        ):
                            break
                finally:
                    agent.for_graph = plain_for_graph  # ty: ignore

                # The operation graph has no HITL middleware and manufactures no
                # tool call, so the slash command is the whole authorization
                # boundary: there is nothing left to approve in any approval
                # mode (this app runs the default Manual mode).
                assert offload_interrupts == []
                # Positive control: the recorder must have seen chunks, so the
                # empty-interrupt assertion above cannot pass vacuously.
                assert recorded_chunks > 0

                app_messages = [
                    str(widget._content) for widget in app.query(AppMessage)
                ]
                error_messages = [
                    str(widget._content) for widget in app.query(ErrorMessage)
                ]

            assert not error_messages
            assert "Nothing to offload" not in "\n".join(app_messages)
            assert any("Offloaded " in content for content in app_messages)

            # The summarization event must be visible through server state.
            state = await agent.aget_state(config)
            values = getattr(state, "values", None) or {}

            # `/offload` frees context by advancing the summarization cutoff, not
            # by deleting messages: the raw conversation stays in the checkpoint
            # so `/context` and resume still see it. Because the replay replaces
            # this channel, a stale or empty input would silently truncate it
            # here and still report success -- so assert identity, not count.
            messages_after = values.get("messages", [])
            assert len(messages_after) == len(messages_before)
            assert [getattr(m, "id", None) for m in messages_after] == [
                getattr(m, "id", None) for m in messages_before
            ]

            summarization_event = values.get("_summarization_event")
            assert summarization_event is not None
            cutoff = _event_field(summarization_event, "cutoff_index")
            assert isinstance(cutoff, int)
            assert cutoff > 0
            # In local mode the history prefix lives under a per-session
            # `artifacts_root`, so assert the suffix rather than a fixed prefix.
            archive_path = _event_field(summarization_event, "file_path")
            assert isinstance(archive_path, str)
            assert archive_path.endswith(f"/conversation_history/{thread_id}.md")

            # CRUCIAL: the archive must be readable THROUGH THE AGENT, proving
            # the bytes exist in the agent's own backend server-side.
            read_back = await _read_file_through_agent(
                agent, thread_id=thread_id, file_path=archive_path
            )
            assert "keeps enough unique detail" in read_back
            # The SDK middleware writes a "## Summarized at" archive header.
            assert "Summarized at" in read_back

        persistent_archive = (
            home_dir / ".deepagents" / "conversation_history" / f"{thread_id}.md"
        )
        assert persistent_archive.exists()
        assert "keeps enough unique detail" in persistent_archive.read_text()
    finally:
        model_config.clear_caches()
