"""Shared test utilities for StateBackend / StoreBackend config context."""

from contextlib import contextmanager
from types import SimpleNamespace

from langchain.tools import ToolRuntime
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_RUNTIME, CONFIG_KEY_SEND

from deepagents.middleware.filesystem import _file_data_reducer


def make_state_config(files=None, *, store=None, context=None):
    """Create a mock config with CONFIG_KEY_SEND, CONFIG_KEY_READ, and CONFIG_KEY_RUNTIME.

    Returns (config, file_store) where file_store is a mutable dict so tests
    can inspect the resulting state.
    """
    file_store = {"files": files or {}}

    def read(select, fresh=False):  # noqa: ARG001
        if isinstance(select, str):
            return file_store.get(select)
        return {k: file_store.get(k) for k in select}

    def send(writes):
        for channel, value in writes:
            if channel == "files":
                file_store["files"] = _file_data_reducer(file_store.get("files"), value)

    mock_runtime = SimpleNamespace(store=store, context=context, stream_writer=lambda _: None)

    config = {
        "configurable": {
            CONFIG_KEY_SEND: send,
            CONFIG_KEY_READ: read,
            CONFIG_KEY_RUNTIME: mock_runtime,
        },
    }
    return config, file_store


@contextmanager
def state_config_context(files=None, *, store=None, context=None):
    """Context manager that activates a mock config for StateBackend / StoreBackend."""
    config, file_store = make_state_config(files, store=store, context=context)
    token = var_child_runnable_config.set(config)
    try:
        yield file_store
    finally:
        var_child_runnable_config.reset(token)


@contextmanager
def make_tool_runtime(state, *, tool_call_id="", store=None, context=None):
    """Create a ToolRuntime with a config context for StateBackend.

    Sets up both the ToolRuntime and the context variable so that
    ``get_config()`` returns a config with CONFIG_KEY_READ / CONFIG_KEY_SEND
    backed by the same files dict in ``state``.

    Usage::

        with make_tool_runtime(state) as (rt, file_store):
            result = tool.invoke({"runtime": rt, ...})
            # file_store["files"] has the current state
    """
    files = state.get("files") or {}
    config, file_store = make_state_config(files, store=store, context=context)
    rt = ToolRuntime(
        state=state,
        context=context,
        tool_call_id=tool_call_id,
        store=store,
        stream_writer=lambda _: None,
        config=config,
    )
    token = var_child_runnable_config.set(config)
    try:
        yield rt, file_store
    finally:
        var_child_runnable_config.reset(token)
