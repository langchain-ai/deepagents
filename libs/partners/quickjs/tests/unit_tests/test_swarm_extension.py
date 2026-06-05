"""Unit tests for the swarm interpreter extension.

Exercises the extension wiring through a real ``_ThreadREPL``: ``on_setup``
registers the dispatch host function and the ``swarm`` module, and guest JS
can ``import { swarm } from "swarm"`` and call it. The dispatch closure is
stubbed so these tests don't make real model calls; a separate test builds
the extension via the public ``swarm()`` factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from quickjs_rs import Runtime, ThreadWorker

from langchain_quickjs import InterpreterExtension, SwarmExtension, swarm
from langchain_quickjs._extensions import has_on_setup, validate_extension_hooks
from langchain_quickjs._repl import _ThreadREPL
from langchain_quickjs._swarm_extension import SwarmExtension as _SwarmExtensionClass

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def worker() -> Iterator[ThreadWorker]:
    w = ThreadWorker()
    try:
        yield w
    finally:
        w.close()


@pytest.fixture
def runtime(worker: ThreadWorker) -> Iterator[Runtime]:
    async def _make() -> Runtime:
        return Runtime()

    rt = worker.run_sync(_make())
    try:
        yield rt
    finally:

        async def _close() -> None:
            rt.close()

        worker.run_sync(_close())


def _make_repl(
    worker: ThreadWorker,
    runtime: Runtime,
    extensions: list[InterpreterExtension],
) -> _ThreadREPL:
    return _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
        extensions=extensions,
    )


def _stub_extension(calls: list[tuple]) -> SwarmExtension:
    """A SwarmExtension whose dispatch records its args and echoes back."""

    async def _dispatch(
        description: str,
        subagent_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> str:
        calls.append((description, subagent_type, response_schema, mode))
        return f"dispatched: {description}"

    return _SwarmExtensionClass(dispatch=_dispatch)


def test_on_setup_registers_module_and_host_fn(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, [_stub_extension([])])
    # Host symbol present and the module importable.
    assert repl.eval_sync("typeof __swarm_dispatch").result == "function"
    assert repl.eval_sync('typeof (await import("swarm")).swarm').result == "function"


def test_swarm_callable_from_guest(worker: ThreadWorker, runtime: Runtime) -> None:
    calls: list[tuple] = []
    repl = _make_repl(worker, runtime, [_stub_extension(calls)])
    outcome = repl.eval_sync(
        'const { swarm } = await import("swarm");'
        ' await swarm({ description: "do a thing", mode: "invoke" })'
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "dispatched: do a thing"
    # camelCase opts marshaled to positional dispatch args.
    assert calls == [("do a thing", None, None, "invoke")]


def test_swarm_forwards_all_options(worker: ThreadWorker, runtime: Runtime) -> None:
    calls: list[tuple] = []
    repl = _make_repl(worker, runtime, [_stub_extension(calls)])
    repl.eval_sync(
        'const { swarm } = await import("swarm");'
        " await swarm({"
        '   description: "triage",'
        '   subagentType: "screener",'
        '   responseSchema: { type: "object" },'
        '   mode: "agent",'
        " })"
    )
    assert calls == [("triage", "screener", {"type": "object"}, "agent")]


def test_swarm_rejects_non_object_payload(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, [_stub_extension([])])
    # A non-object payload makes the host function raise; the guest sees a
    # catchable JS error (host exceptions surface as rejected promises).
    outcome = repl.eval_sync(
        'const { swarm } = await import("swarm");'
        ' let caught = "";'
        ' try { await swarm("oops") } catch (e) { caught = "errored" }'
        " caught"
    )
    assert outcome.error_type is None
    assert outcome.result == "errored"


def test_system_prompt_present() -> None:
    ext = _stub_extension([])
    assert ext.system_prompt is not None
    assert "swarm" in ext.system_prompt


def test_extension_validates_as_a_hook_impl() -> None:
    ext = _stub_extension([])
    assert has_on_setup(ext) is True
    validate_extension_hooks(ext)  # does not raise


def test_factory_builds_extension() -> None:
    # The public factory wires a real dispatch closure (no model call here,
    # just construction) and produces a valid extension.
    ext = swarm(default_model="openai:gpt-4o-mini")
    assert isinstance(ext, SwarmExtension)
    assert has_on_setup(ext) is True
    assert ext.system_prompt is not None
