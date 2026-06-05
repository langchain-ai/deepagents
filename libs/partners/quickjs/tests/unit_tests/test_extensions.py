"""Unit tests for interpreter extensions.

Phase 1 — ``ExtensionContext`` + the Protocol: the context method →
``quickjs_rs`` binding mapping (host-function registration, module
install, setup eval), re-registration overwrite, identifier validation,
and the hook-detection helpers.

Phase 2 — ``on_setup`` wired into the REPL slot lifecycle: an extension's
module + host function are usable through a real ``_ThreadREPL``, and they
survive a snapshot round-trip onto a freshly built REPL (the production
per-turn behavior).

Phase 3 — ``on_eval`` wired into the eval path: an extension re-registers a
host function closing over fresh per-eval scratch, which accumulates within
one eval and resets across evals (the PTC call-budget pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from quickjs_rs import Context, ModuleScope, Runtime, ThreadWorker

if TYPE_CHECKING:
    from collections.abc import Iterator

from langchain_quickjs import (
    ExtensionContext,
    ExtensionError,
    InterpreterExtension,
)
from langchain_quickjs._extensions import (
    has_on_eval,
    has_on_setup,
    validate_extension_hooks,
)
from langchain_quickjs._repl import _ThreadREPL

# ---------------------------------------------------------------------------
# Fixtures (mirror tests/unit_tests/test_ptc.py)
# ---------------------------------------------------------------------------


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


@pytest.fixture
def ctx(worker: ThreadWorker, runtime: Runtime) -> ExtensionContext:
    async def _make() -> Context:
        return runtime.new_context(timeout=5.0)

    qjs_ctx = worker.run_sync(_make())
    return ExtensionContext(ctx=qjs_ctx, runtime=runtime, backend=None)


def _on_worker(worker: ThreadWorker, ctx: ExtensionContext, fn):
    """Run ``fn(ctx)`` on the worker thread.

    ``ExtensionContext`` methods touch the ``!Send`` context, so they must
    run on the owning worker thread — exactly how the REPL drives the hooks
    from inside ``_ainit`` / the eval loop. Tests use this to register
    against the context the way the real wiring does.
    """

    async def _run():
        return fn(ctx)

    return worker.run_sync(_run())


def _eval(worker: ThreadWorker, ctx: ExtensionContext, code: str):
    """Run guest-style code against the underlying context for assertions."""

    async def _run():
        return await ctx._ctx.eval_async(code)

    return worker.run_sync(_run())


def test_function_registers_callable_callable_from_js(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    def register(c: ExtensionContext) -> None:
        @c.function(name="__ext_add")
        def add(a: int, b: int) -> int:
            return a + b

    _on_worker(worker, ctx, register)
    assert _eval(worker, ctx, "__ext_add(2, 3)") == 5


def test_function_bare_decorator_infers_name(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    def register(c: ExtensionContext) -> None:
        @c.function
        def __ext_double(x: int) -> int:
            return x * 2

    _on_worker(worker, ctx, register)
    assert _eval(worker, ctx, "__ext_double(21)") == 42


def test_function_returns_original_callable(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    def register(c: ExtensionContext):
        @c.function(name="__ext_id")
        def ident(x: int) -> int:
            return x

        return ident

    ident = _on_worker(worker, ctx, register)
    # Decorator returns the callable unchanged — still usable from Python.
    assert ident(7) == 7


def test_function_async_host(worker: ThreadWorker, ctx: ExtensionContext) -> None:
    def register(c: ExtensionContext) -> None:
        @c.function(name="__ext_aval", is_async=True)
        async def aval() -> str:
            return "from python"

    _on_worker(worker, ctx, register)
    # eval_async supports top-level await and unwraps the result.
    assert _eval(worker, ctx, "await __ext_aval()") == "from python"


def test_function_rejects_invalid_identifier(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    def register(c: ExtensionContext) -> None:
        @c.function(name="bad-name")
        def f() -> None: ...

    with pytest.raises(ExtensionError, match="valid JS identifier"):
        _on_worker(worker, ctx, register)


def test_function_rejects_leading_digit(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    def register(c: ExtensionContext) -> None:
        @c.function(name="9fn")
        def f() -> None: ...

    with pytest.raises(ExtensionError, match="valid JS identifier"):
        _on_worker(worker, ctx, register)


def test_function_reregister_overwrites(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    # Re-registering the same symbol overwrites (last write wins) — no
    # collision guard.
    def register_first(c: ExtensionContext) -> None:
        @c.function(name="__ext_dup")
        def first() -> int:
            return 1

    _on_worker(worker, ctx, register_first)
    assert _eval(worker, ctx, "__ext_dup()") == 1

    def register_second(c: ExtensionContext) -> None:
        @c.function(name="__ext_dup")
        def second() -> int:
            return 2

    _on_worker(worker, ctx, register_second)
    assert _eval(worker, ctx, "__ext_dup()") == 2


def test_module_single_file_importable(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    # A bare import name is a nested ModuleScope keyed by that name; the
    # scope's files live inside, with an `index.<ext>` entrypoint. This
    # mirrors how the skills loader builds scopes (see _skills.py).
    _on_worker(
        worker,
        ctx,
        lambda c: c.module(
            ModuleScope(
                {
                    "greet": ModuleScope(
                        {
                            "index.js": (
                                "export function hello(n) { return `hi ${n}`; }\n"
                            )
                        }
                    )
                }
            )
        ),
    )
    result = _eval(
        worker,
        ctx,
        'const m = await import("greet"); m.hello("ada")',
    )
    assert result == "hi ada"


def test_module_multi_file_relative_import(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    # Entrypoint plus an internal sibling. Sibling keys are plain file
    # names (no leading ./); the import specifier in source keeps the ./.
    _on_worker(
        worker,
        ctx,
        lambda c: c.module(
            ModuleScope(
                {
                    "calc": ModuleScope(
                        {
                            "index.js": (
                                'import { two } from "./nums.js";\n'
                                "export function addTwo(x) { return x + two(); }\n"
                            ),
                            "nums.js": "export function two() { return 2; }\n",
                        }
                    )
                }
            )
        ),
    )
    result = _eval(
        worker,
        ctx,
        'const m = await import("calc"); m.addTwo(40)',
    )
    assert result == 42


def test_eval_wires_host_symbols_into_global(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    def register(c: ExtensionContext) -> None:
        @c.function(name="__ext_now")
        def now() -> int:
            return 1234

        # Setup eval assembles an idiomatic global from the bare symbol.
        c.eval("globalThis.clock = { now: __ext_now }; undefined")

    _on_worker(worker, ctx, register)
    assert _eval(worker, ctx, "clock.now()") == 1234


def test_eval_returns_completion_value(
    worker: ThreadWorker, ctx: ExtensionContext
) -> None:
    assert _on_worker(worker, ctx, lambda c: c.eval("1 + 2")) == 3


class _SetupOnly(InterpreterExtension):
    system_prompt = None

    def on_setup(self, ctx: ExtensionContext) -> None: ...


class _EvalOnly(InterpreterExtension):
    system_prompt = None

    def on_eval(self, ctx, runtime) -> None: ...


class _Both(InterpreterExtension):
    system_prompt = "use the thing"

    def on_setup(self, ctx: ExtensionContext) -> None: ...

    def on_eval(self, ctx, runtime) -> None: ...


class _Neither(InterpreterExtension):
    system_prompt = None


def test_hook_detection() -> None:
    assert has_on_setup(_SetupOnly()) is True
    assert has_on_eval(_SetupOnly()) is False

    assert has_on_setup(_EvalOnly()) is False
    assert has_on_eval(_EvalOnly()) is True

    assert has_on_setup(_Both()) is True
    assert has_on_eval(_Both()) is True

    assert has_on_setup(_Neither()) is False
    assert has_on_eval(_Neither()) is False


def test_validate_accepts_at_least_one_hook() -> None:
    validate_extension_hooks(_SetupOnly())
    validate_extension_hooks(_EvalOnly())
    validate_extension_hooks(_Both())


def test_validate_rejects_empty_extension() -> None:
    with pytest.raises(ExtensionError, match="at least one"):
        validate_extension_hooks(_Neither())


# ---------------------------------------------------------------------------
# Phase 2: on_setup wired into the REPL slot lifecycle
# ---------------------------------------------------------------------------


class _GreetExtension(InterpreterExtension):
    """Registers a host function plus a module that forwards to it."""

    system_prompt = None

    def on_setup(self, ctx: ExtensionContext) -> None:
        @ctx.function(name="__greet_host")
        def greet(name: str) -> str:
            return f"hello {name}"

        ctx.module(
            ModuleScope(
                {
                    "greet": ModuleScope(
                        {
                            "index.js": (
                                "export function greet(name) {\n"
                                "  return __greet_host(name);\n"
                                "}\n"
                            )
                        }
                    )
                }
            )
        )


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


def test_on_setup_runs_at_repl_build(worker: ThreadWorker, runtime: Runtime) -> None:
    repl = _make_repl(worker, runtime, [_GreetExtension()])
    # The host function and the module that forwards to it are both live
    # immediately after the REPL is built — on_setup ran in _ainit.
    outcome = repl.eval_sync('const m = await import("greet"); m.greet("ada")')
    assert outcome.error_type is None
    assert outcome.result == "hello ada"


def test_on_setup_host_function_directly_callable(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, [_GreetExtension()])
    outcome = repl.eval_sync('__greet_host("bob")')
    assert outcome.result == "hello bob"


def test_on_setup_reruns_on_fresh_repl(worker: ThreadWorker, runtime: Runtime) -> None:
    # Production rebuilds a fresh _ThreadREPL each turn (the middleware
    # evicts + recreates the slot). Host functions don't live in a
    # snapshot, so they must be re-registered by on_setup on every fresh
    # context. Build a second REPL with the same extensions and confirm
    # on_setup ran again — the host function is live on the new context.
    _make_repl(worker, runtime, [_GreetExtension()])
    repl2 = _make_repl(worker, runtime, [_GreetExtension()])
    assert repl2.eval_sync('__greet_host("eve")').result == "hello eve"


def test_module_reinstall_on_same_runtime_is_tolerated(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    # Two REPLs on one Runtime both install the "greet" module specifier in
    # their on_setup. Re-installing the same specifier must not raise (it
    # would break the per-turn rebuild) — this is implementation-plan risk
    # #3, resolved: runtime.install tolerates a repeat specifier.
    _make_repl(worker, runtime, [_GreetExtension()])
    repl2 = _make_repl(worker, runtime, [_GreetExtension()])
    outcome = repl2.eval_sync('const m = await import("greet"); m.greet("zoe")')
    assert outcome.error_type is None
    assert outcome.result == "hello zoe"


def test_no_extensions_is_noop(worker: ThreadWorker, runtime: Runtime) -> None:
    repl = _make_repl(worker, runtime, [])
    assert repl.eval_sync("1 + 1").result == "2"


# ---------------------------------------------------------------------------
# Phase 3: on_eval wired into the eval path
# ---------------------------------------------------------------------------


class _CounterExtension(InterpreterExtension):
    """on_eval-only: re-registers a host fn closing over fresh per-eval state.

    Each eval gets a brand-new counter, so calls within one eval accumulate
    but never carry across evals — the PTC call-budget pattern in miniature.
    """

    system_prompt = None

    def on_eval(self, ctx: ExtensionContext, runtime) -> None:
        count = [0]

        @ctx.function(name="__counter_tick")
        def tick() -> int:
            count[0] += 1
            return count[0]


def test_on_eval_runs_and_host_fn_works(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, [_CounterExtension()])
    # on_eval registered __counter_tick before the guest code ran.
    assert repl.eval_sync("__counter_tick()").result == "1"


def test_on_eval_state_accumulates_within_one_eval(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, [_CounterExtension()])
    outcome = repl.eval_sync("__counter_tick(); __counter_tick(); __counter_tick()")
    assert outcome.result == "3"


def test_on_eval_state_resets_each_eval(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, [_CounterExtension()])
    # Two ticks in the first eval...
    assert repl.eval_sync("__counter_tick(); __counter_tick()").result == "2"
    # ...the second eval starts from a fresh counter (on_eval re-ran).
    assert repl.eval_sync("__counter_tick()").result == "1"


def test_on_eval_only_extension_needs_no_on_setup(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    # _CounterExtension implements only on_eval. Before its first eval the
    # symbol doesn't exist; on_eval establishes it. This confirms the
    # on_eval-only shape works end to end through the REPL.
    repl = _make_repl(worker, runtime, [_CounterExtension()])
    assert has_on_setup(_CounterExtension()) is False
    assert repl.eval_sync("typeof __counter_tick").result == "function"
