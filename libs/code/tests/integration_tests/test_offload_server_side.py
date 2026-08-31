"""Integration coverage for the server-side `/offload` path.

For a server-backed agent `/offload` runs through dcode's server HTTP operation,
which compacts without a model node or a synthetic tool call. Either way
the offloaded archive lands in the agent's composite backend and is readable via
`read_file` in every run mode — not in a client-local directory the server can
never read. These tests construct the app the PRODUCTION way (`backend=None`)
and prove the archive is readable *through the agent*.
"""

from __future__ import annotations

import json
import re
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
    (config_dir / "prices.json").write_text(
        """
[
  {
    "id": "itest",
    "name": "Integration Test",
    "api_pattern": "itest",
    "models": [
      {
        "id": "fake",
        "match": {"equals": "fake"},
        "prices": {"input_mtok": 1.0, "output_mtok": 2.0}
      }
    ]
  }
]
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
    from deepagents_code.config import build_stream_config, runtime_state

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
        context={"model_context_limit": runtime_state.model_context_limit},
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
    # Offload never changes the thread's graph association, so the same client
    # can immediately seed a read through the interactive graph.
    agent_graph = agent
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
    - the operation succeeds through the custom server route,
    - a persisted `_summarization_event` with `cutoff > 0` and a
      `file_path` of `/conversation_history/session_<uuid4hex>.md` -- the SDK
      names the archive from the summarization session id, not the thread id,
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
        create_model("itest:fake").apply_to_runtime_state()
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

            # Captured before the operation to prove its state-only commit does
            # not replace or otherwise rewrite conversation messages.
            before_state = await agent.aget_state(config)
            messages_before = list(
                (getattr(before_state, "values", None) or {}).get("messages", [])
            )
            cost_before = float(
                (getattr(before_state, "values", None) or {}).get(
                    "_session_cost_usd", 0.0
                )
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

            async with app.run_test() as pilot:
                for _ in range(120):
                    await pilot.pause(0.1)
                    if app._message_store.total_count > 0:
                        break

                assert app._message_store.total_count > 0

                await app._handle_offload()

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

            assert not error_messages
            assert "Nothing to offload" not in "\n".join(app_messages)
            assert any("Offloaded " in content for content in app_messages)

            # The summarization event must be visible through server state.
            state = await agent.aget_state(config)
            values = getattr(state, "values", None) or {}
            assert float(values.get("_session_cost_usd", 0.0)) > cost_before

            # `/offload` frees context by advancing the summarization cutoff,
            # not by deleting messages: raw history stays checkpointed. Assert
            # identity to prove the operation never supplied message input.
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
            # The leaf is the summarization *session* id (`_get_history_path`),
            # which is not the thread id: asserting `{thread_id}.md` here made
            # this test claim a naming scheme the SDK never produces.
            archive_path = _event_field(summarization_event, "file_path")
            assert isinstance(archive_path, str)
            assert re.fullmatch(
                r".*/conversation_history/session_[0-9a-f]{32}\.md", archive_path
            ), archive_path
            archive_name = archive_path.rsplit("/", 1)[1]

            # CRUCIAL: the archive must be readable THROUGH THE AGENT, proving
            # the bytes exist in the agent's own backend server-side.
            read_back = await _read_file_through_agent(
                agent, thread_id=thread_id, file_path=archive_path
            )
            assert "keeps enough unique detail" in read_back
            # The SDK middleware writes a "## Summarized at" archive header.
            assert "Summarized at" in read_back

        persistent_archive = (
            home_dir / ".deepagents" / "conversation_history" / archive_name
        )
        assert persistent_archive.exists()
        assert "keeps enough unique detail" in persistent_archive.read_text()
    finally:
        model_config.clear_caches()


async def _reject_any_hook(  # noqa: RUF029  # must satisfy the async fulfill_hook signature
    request: object,
) -> dict[str, object]:
    """Fail loudly if the offload unexpectedly routes a hook to the client.

    No hooks are configured in this test, so a well-formed operation never
    interrupts. Returning a deny would mask a protocol bug as a hook denial.

    Raises:
        AssertionError: Always — no hook request is expected here.
    """
    msg = f"Unexpected hook request during offload: {request!r}"
    raise AssertionError(msg)


async def _wait_for_file(path: Path) -> None:
    """Poll until `path` exists, so the test can sync with the server process.

    The gate files are written by the server subprocess, whose clock and event
    loop are independent of the test's; polling (with a generous ceiling) is
    the only synchronization primitive available across that boundary.

    Raises:
        TimeoutError: If the file does not appear within 60 seconds.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 60.0
    while not path.exists():  # noqa: ASYNC240  # cheap stat per poll; the gate protocol is file-based by design
        if loop.time() > deadline:
            msg = f"Timed out waiting for the server to create {path}"
            raise TimeoutError(msg)
        await asyncio.sleep(0.05)


@pytest.mark.timeout(240)
async def test_concurrent_run_during_offload_preserves_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run starting mid-offload must not be clobbered by the offload commit.

    The server operation reads state, runs compaction (a model call), then
    commits a state-only update. Between its final idle check and the
    `update_state` write there is a window in which a user run can start. This
    test holds the compaction model call open (via the
    `DCA_TEST_OFFLOAD_GATE_DIR` summary gate) and starts a run inside that
    window, then asserts the invariant the design relies on: LangGraph either
    rejects the offload's write or serializes it before the run — in both
    cases the run's message must be present in the final thread state, and the
    offload either committed cleanly or reported a conflict (it must never
    silently branch from the stale checkpoint).
    """
    import asyncio

    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    gate_dir = tmp_path / "gate"
    assistant_id = "itest-offload-race"

    home_dir.mkdir()
    project_dir.mkdir()
    gate_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    # Reaches the server subprocess through `_build_server_env`'s
    # `os.environ.copy()`; gates only summary-generation model calls.
    monkeypatch.setenv("DCA_TEST_OFFLOAD_GATE_DIR", str(gate_dir))
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents_code import model_config
    from deepagents_code.client.launch.server_manager import server_session
    from deepagents_code.config import create_model
    from deepagents_code.sessions import generate_thread_id

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model("itest:fake").apply_to_runtime_state()
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
            messages_before = list(
                (getattr(await agent.aget_state(config), "values", None) or {}).get(
                    "messages", []
                )
            )
            assert messages_before

            # An offload whose summary call blocks at the gate. Errors are
            # captured rather than raised so the gate release and the invariant
            # check run regardless of how the operation resolves.
            offload_error: list[BaseException] = []

            async def _offload() -> None:
                try:
                    await agent.aoffload(
                        config=config,
                        context={"model": "itest:fake"},
                        fulfill_hook=_reject_any_hook,
                    )
                except BaseException as exc:  # noqa: BLE001  # asserted below
                    offload_error.append(exc)

            offload_task = asyncio.create_task(_offload())

            # Wait until the server is provably mid-summary, i.e. past its idle
            # checks and inside the window the final commit must be safe in.
            await _wait_for_file(gate_dir / "entered")

            # Launch a real run on the same thread while offload is blocked.
            # Its model call is not a summary request, so it passes the gate.
            run_task = asyncio.create_task(
                _run_turn(
                    agent,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    prompt="concurrent turn: the message that must survive",
                )
            )
            # Give the run a beat to register server-side before releasing the
            # offload, so the commit and the run genuinely overlap.
            await asyncio.sleep(1.0)
            (gate_dir / "release").write_text("1")

            await asyncio.wait_for(run_task, timeout=120)
            await asyncio.wait_for(offload_task, timeout=120)

            # The offload either committed or failed with a conflict; both are
            # acceptable outcomes of a genuine race. A hang or an unexpected
            # exception type is not.
            for exc in offload_error:
                text = f"{type(exc).__name__}: {exc}"
                assert "changed" in text or "active" in text or "409" in text, text

            # The invariant: whatever the offload did, the concurrent run's
            # message survived. If LangGraph ever lets the state-only write
            # branch from the stale checkpoint, this fails because the run's
            # appended messages would be missing.
            final_values = getattr(await agent.aget_state(config), "values", None) or {}
            final_contents = [
                str(getattr(m, "content", m.get("content", "")))
                for m in final_values.get("messages", [])
            ]
            assert any(
                "the message that must survive" in content for content in final_contents
            ), final_contents
            # The pre-offload history was not truncated either: the event only
            # advances a cutoff; raw messages stay checkpointed.
            assert len(final_values.get("messages", [])) >= len(messages_before)
    finally:
        # Never leave the server subprocess blocked on the gate.
        (gate_dir / "release").write_text("1")
        model_config.clear_caches()


_TEST_AUTH_MODULE = '''\
"""Minimal token auth backend for the custom-route-auth integration test."""

from langgraph_sdk import Auth

auth = Auth()


@auth.authenticate
async def authenticate(authorization: str | None) -> str:
    """Accept only the fixed test bearer token; reject everything else.

    Returns:
        A user id for the one credential this test server trusts.

    Raises:
        Auth.exceptions.HTTPException: On a missing or wrong token.
    """
    if authorization != "Bearer itest-token":
        raise Auth.exceptions.HTTPException(status_code=401, detail="nope")
    return "itest-user"
'''


@pytest.mark.timeout(240)
async def test_offload_route_respects_configured_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The custom offload routes are gated exactly like the graph routes.

    Production dcode servers run `LANGGRAPH_AUTH_TYPE=noop` (localhost trust),
    but the generated `langgraph.json` sets `enable_custom_route_auth: True`
    so that a deployment which *does* configure an auth backend gets the
    `/dcode/*` operation routes behind the same middleware as `/threads`.
    The threat model asserts that; this test proves it end to end: a real
    server with a token-rejecting auth backend must reject an unauthenticated
    POST to the offload route with the same 401 it gives a protected graph
    route — not with a 404/422 that would mean the route bypassed auth — and
    must accept the request once the credential is supplied.
    """
    import httpx

    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    work_dir = tmp_path / "server_work"
    home_dir.mkdir()
    project_dir.mkdir()
    work_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents_code import model_config
    from deepagents_code.client.launch.server import (
        ServerProcess,
        generate_langgraph_json,
    )
    from deepagents_code.config import create_model

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    server: ServerProcess | None = None
    try:
        create_model("itest:fake").apply_to_runtime_state()

        # The auth module lives in the server work dir (the subprocess's cwd,
        # which `langgraph dev` puts on `sys.path`) so its import path stays
        # relative to the deployment, exactly like a real deployment's
        # `auth.py` next to its `langgraph.json`.
        (work_dir / "itest_auth.py").write_text(_TEST_AUTH_MODULE)
        generate_langgraph_json(
            work_dir,
            auth_path="./itest_auth.py:auth",
        )

        # No scaffold: the workspace is fully prepared above, and a missing
        # langgraph.json here would be a test bug worth failing on.
        server = ServerProcess(config_dir=work_dir, scaffold=None)
        await server.start()

        async with httpx.AsyncClient(base_url=server.url) as http:
            unauthenticated = await http.post(
                "/dcode/threads/thread-1/offload",
                json={"operation_id": "op-1", "context": {}, "hook_responses": {}},
            )
            protected_graph_route = await http.post("/threads", json={})
            assert unauthenticated.status_code == 401, (
                unauthenticated.status_code,
                unauthenticated.text,
            )
            assert protected_graph_route.status_code == 401, (
                protected_graph_route.status_code,
                protected_graph_route.text,
            )

            headers = {"Authorization": "Bearer itest-token"}
            authenticated = await http.post(
                "/dcode/threads/thread-1/offload",
                json={"operation_id": "op-1", "context": {}, "hook_responses": {}},
                headers=headers,
            )
            # 404/409/500 all pass auth and fail inside the operation (the
            # thread does not exist); only 401/403 would mean auth still
            # rejected a credentialed request.
            assert authenticated.status_code not in (401, 403), (
                authenticated.status_code,
                authenticated.text,
            )
            # A malformed context fails at the boundary with a field-naming
            # 422, not a 500 from deep in model resolution.
            malformed = await http.post(
                "/dcode/threads/thread-1/offload",
                json={
                    "operation_id": "op-1",
                    "context": {"model": 123},
                    "hook_responses": {},
                },
                headers=headers,
            )
            assert malformed.status_code == 422, (
                malformed.status_code,
                malformed.text,
            )
            assert "context.model" in malformed.text
    finally:
        if server is not None:
            server.stop()
        model_config.clear_caches()


_TEST_FLUSH_APP = """
import os
from contextlib import asynccontextmanager
from pathlib import Path

from langsmith import run_trees

from deepagents_code.offload_api import app


class _MarkerClient:
    def flush(self) -> None:
        Path(os.environ["ITEST_TRACE_FLUSH_MARKER"]).write_text("flushed")


_original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _marker_lifespan(starlette_app):
    previous = getattr(run_trees, "_CLIENT", None)
    run_trees._CLIENT = _MarkerClient()
    try:
        async with _original_lifespan(starlette_app):
            yield
    finally:
        run_trees._CLIENT = previous


app.router.lifespan_context = _marker_lifespan
"""


@pytest.mark.timeout(60)
async def test_server_shutdown_flushes_existing_tracers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real `langgraph dev` shutdown runs the custom app's flush lifespan."""
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    work_dir = tmp_path / "server_work"
    marker = tmp_path / "trace-flushed"
    home_dir.mkdir()
    project_dir.mkdir()
    work_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("ITEST_TRACE_FLUSH_MARKER", str(marker))
    monkeypatch.chdir(project_dir)
    _write_model_config(home_dir)

    from deepagents_code import model_config
    from deepagents_code.client.launch.server import (
        ServerProcess,
        generate_langgraph_json,
    )
    from deepagents_code.config import create_model

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    server: ServerProcess | None = None
    try:
        create_model("itest:fake").apply_to_runtime_state()
        (work_dir / "itest_flush_app.py").write_text(_TEST_FLUSH_APP)
        generated = generate_langgraph_json(work_dir)
        config = json.loads(generated.read_text())
        config["http"]["app"] = "./itest_flush_app.py:app"
        generated.write_text(json.dumps(config, indent=2))

        server = ServerProcess(config_dir=work_dir, scaffold=None)
        await server.start()
        server.stop()

        assert marker.read_text() == "flushed"
    finally:
        if server is not None:
            server.stop()
        model_config.clear_caches()
