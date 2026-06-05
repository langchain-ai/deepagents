"""Interpreter extensions for ``CodeInterpreterMiddleware``.

An *extension* is a reusable unit that wires capability into the QuickJS
interpreter: JS modules the guest imports, host (FFI) functions backed by
Python callables, install-time setup eval, and a system-prompt fragment.
Both are optional, but an extension must implement at least one of two
lifecycle hooks:

- ``on_setup(ctx)`` — runs once per context creation (every turn under
  ``snapshot_between_turns``, since the middleware rebuilds the context).
  Register modules, stateless host functions, and run setup eval here.
- ``on_eval(ctx, runtime)`` — runs at the start of every eval, with that
  eval's ``ToolRuntime``. Re-register host functions that close over
  fresh per-eval scratch here; re-registering a symbol overwrites it.

This module defines the public ``InterpreterExtension`` Protocol and the
``ExtensionContext`` an extension registers against. The wiring that calls
the hooks lives on ``_ThreadREPL`` / the middleware.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from langchain_quickjs._ptc import is_valid_js_identifier

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langgraph.prebuilt import ToolRuntime
    from quickjs_rs import Context, ModuleScope, Runtime

# A host function is a sync or async Python callable invoked from inside
# QuickJS. It receives the guest's arguments (already coerced to Python
# values) and returns a JSON-serializable result. Installed via
# ``Context.register(symbol, fn, is_async=...)``.
HostFunction = Callable[..., Any | Awaitable[Any]]


class ExtensionError(RuntimeError):
    """Raised when an extension does something invalid.

    Covers an invalid JS identifier and the at-least-one-hook check.
    Surfaced at the point of failure so the traceback names the offending
    extension.
    """


class ExtensionContext:
    """The interpreter-wiring handle passed to an extension's hooks.

    Constructed per ``_ThreadREPL`` (i.e. per context creation) and bound
    to that REPL's ``quickjs_rs.Context`` and ``Runtime``.

    Registration maps straight to the binding — ``module`` →
    ``Runtime.install``, ``function`` → ``Context.register``, ``eval`` →
    ``Context.eval``. The context object is ``!Send``, so **every method
    here must be called on the owning worker thread**: the REPL invokes the
    hooks from inside its worker-loop coroutines (``_ainit`` for
    ``on_setup``, the eval driver for ``on_eval``), exactly where
    ``_install_console`` registers the console functions. Callers off the
    worker thread must wrap the hook invocation in ``worker.run_sync``;
    these methods do not re-dispatch (doing so from the worker thread
    deadlocks).
    """

    def __init__(
        self,
        *,
        ctx: Context,
        runtime: Runtime,
        backend: BackendProtocol | None,
    ) -> None:
        self._ctx = ctx
        self._runtime = runtime
        self._backend = backend

    @property
    def backend(self) -> BackendProtocol | None:
        """The deepagents backend.

        Host functions the extension registers can close over this to reach
        the filesystem/store. This hands the *extension* the full backend;
        the guest never sees it — it only reaches whatever narrow surface
        the extension re-exposes through registered host functions. May be
        ``None`` when no backend is configured (a backend-free extension —
        pure host functions / modules — is valid).
        """
        return self._backend

    def module(self, scope: ModuleScope) -> None:
        """Register JS modules the guest can import.

        ``scope`` is a ``ModuleScope``: a bare import name maps to a nested
        ``ModuleScope`` whose files include an ``index.<ext>`` entrypoint
        (the same shape the skills loader builds). Installs onto the
        ``Runtime`` so the name resolves as a bare import.
        """
        self._runtime.install(scope)

    @overload
    def function(self, fn: HostFunction, /) -> HostFunction: ...
    @overload
    def function(
        self, *, name: str | None = ..., is_async: bool | None = ...
    ) -> Callable[[HostFunction], HostFunction]: ...
    def function(
        self,
        fn: HostFunction | None = None,
        *,
        name: str | None = None,
        is_async: bool | None = None,
    ) -> Any:
        """Decorator that registers a Python callable as an FFI host function.

        Bare (``@ctx.function``) infers the symbol from ``fn.__name__``.
        Parameterized (``@ctx.function(name="__x")``) sets it explicitly.
        ``is_async=None`` (default) lets the binding infer from whether the
        callable is a coroutine function. Returns the original callable
        unchanged, so the extension can still call it from Python.

        The symbol lands in one global scope shared by every extension and
        the guest, so a repeat symbol overwrites the prior registration
        (last write wins, matching ``Context.register``). Use a distinctive
        prefix — ``__<ext>_<name>`` — to avoid clobbering another
        extension. Raises ``ExtensionError`` only if the symbol fails
        ``is_valid_js_identifier``.
        """

        def _register(func: HostFunction) -> HostFunction:
            symbol = name if name is not None else getattr(func, "__name__", None)
            if not symbol or not is_valid_js_identifier(symbol):
                msg = (
                    f"host-function symbol {symbol!r} is not a valid JS "
                    "identifier (/^[A-Za-z_$][A-Za-z0-9_$]*$/)"
                )
                raise ExtensionError(msg)
            self._ctx.register(symbol, func, is_async=is_async)
            return func

        # Bare ``@ctx.function`` — called with the function directly.
        if fn is not None:
            return _register(fn)
        return _register

    def eval(self, script: str) -> Any:
        """Evaluate ``script`` against the context, now, and return its value.

        Wire registered host symbols into idiomatic globals — the move the
        REPL makes for ``globalThis.console`` / ``globalThis.tools``. A
        setup eval *assigns* host symbols into a global object; it does not
        *await* a host function, so the synchronous ``Context.eval`` is the
        right call here (and is what ``_install_console`` uses).

        The completion value is marshaled back to Python; a non-marshalable
        return (e.g. a JS object) raises ``MarshalError``, which is why
        setup evals end in ``; undefined``. ``script`` is extension-
        authored, never guest-derived.
        """
        return self._ctx.eval(script)


@runtime_checkable
class InterpreterExtension(Protocol):
    """A reusable unit installed into the QuickJS interpreter.

    An extension declares an optional ``system_prompt`` and implements one
    or both lifecycle hooks the middleware drives against an
    ``ExtensionContext``. Both are optional, but an extension must
    implement at least one — one with neither does nothing, and the
    middleware rejects it at registration (see ``extension_hooks``).

    The Protocol is ``runtime_checkable`` for ``isinstance`` convenience,
    but the check is structural and weak — the middleware uses explicit
    method-presence detection, not ``isinstance``, to decide which hooks
    to call.
    """

    system_prompt: str | None

    def on_setup(self, ctx: ExtensionContext) -> None:
        """Register durable contributions, once per context creation.

        Optional. Omit it if all wiring is per-eval (then implement
        ``on_eval``).
        """

    def on_eval(self, ctx: ExtensionContext, runtime: ToolRuntime | None) -> None:
        """Run at the start of every eval with that eval's ``ToolRuntime``.

        Optional. Re-register host functions that close over fresh
        per-eval scratch (``ctx.function`` overwrites here). Omit it if all
        wiring is one-time (then implement ``on_setup``).

        ``runtime`` is ``None`` for runtime-less evals (e.g. a bare
        ``eval_sync`` in tests); guard for it if you read agent state.
        """


def has_on_setup(ext: InterpreterExtension) -> bool:
    """Return whether ``ext`` provides its own ``on_setup`` implementation.

    The Protocol carries default method bodies, so ``hasattr`` is always
    true. We compare the bound method's underlying function against the
    Protocol default to detect a genuine override.
    """
    return _overrides(ext, "on_setup")


def has_on_eval(ext: InterpreterExtension) -> bool:
    """Return whether ``ext`` provides its own ``on_eval`` implementation."""
    return _overrides(ext, "on_eval")


def _overrides(ext: object, method: str) -> bool:
    impl = getattr(type(ext), method, None)
    if impl is None:
        return False
    return impl is not getattr(InterpreterExtension, method, None)


def validate_extension_hooks(ext: InterpreterExtension) -> None:
    """Raise if ``ext`` implements neither lifecycle hook.

    An extension with neither ``on_setup`` nor ``on_eval`` does nothing;
    reject it at registration (fail-fast) rather than accept it silently.
    """
    if not has_on_setup(ext) and not has_on_eval(ext):
        msg = (
            f"{type(ext).__name__} implements neither on_setup nor on_eval; "
            "an extension must implement at least one lifecycle hook"
        )
        raise ExtensionError(msg)


def run_setup_hooks(
    extensions: list[InterpreterExtension],
    *,
    ctx: Context,
    runtime: Runtime,
    backend: BackendProtocol | None,
) -> None:
    """Run every extension's ``on_setup`` against a fresh context.

    Called by the REPL from inside ``_ainit`` — i.e. **on the worker
    thread**, where the ``!Send`` context lives — after the console is
    installed. One ``ExtensionContext`` is shared across all extensions.

    Each extension's ``on_setup`` is plain synchronous Python; the binding
    calls it makes (``module`` / ``function`` / ``eval``) run directly on
    this worker thread.
    """
    ext_ctx = ExtensionContext(ctx=ctx, runtime=runtime, backend=backend)
    for ext in extensions:
        if has_on_setup(ext):
            ext.on_setup(ext_ctx)


def run_eval_hooks(
    extensions: list[InterpreterExtension],
    *,
    ctx: Context,
    runtime: Runtime,
    backend: BackendProtocol | None,
    tool_runtime: ToolRuntime | None,
) -> None:
    """Run every extension's ``on_eval`` at the start of an eval.

    Called by the REPL from inside ``_aeval_async`` — **on the worker
    thread**, where the ``!Send`` context lives — after per-eval state is
    set up and before the guest code runs. Each ``on_eval`` gets
    ``tool_runtime`` (this eval's ``ToolRuntime``); re-registering a host
    symbol here overwrites the prior one, which is how an extension swaps
    in a closure over fresh per-eval scratch.

    ``tool_runtime`` is ``None`` for runtime-less evals (e.g. a bare
    ``eval_sync`` in tests); extensions that read it must tolerate ``None``.
    Exceptions propagate to ``_aeval_async``'s error handling — do not
    swallow them here.
    """
    ext_ctx = ExtensionContext(ctx=ctx, runtime=runtime, backend=backend)
    for ext in extensions:
        if has_on_eval(ext):
            ext.on_eval(ext_ctx, tool_runtime)


__all__ = [
    "ExtensionContext",
    "ExtensionError",
    "HostFunction",
    "InterpreterExtension",
    "has_on_eval",
    "has_on_setup",
    "run_eval_hooks",
    "run_setup_hooks",
    "validate_extension_hooks",
]
