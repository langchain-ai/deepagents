import base64
import json
import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Any, cast, get_type_hints

import pytest
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import Strict, ValidationError

from deepagents_browser.errors import BrowserErrorCode, BrowserRuntimeError
from deepagents_browser.middleware import BrowserAccessError, BrowserMiddleware
from deepagents_browser.runtime import ScreenshotResult
from deepagents_browser.schemas import (
    ActInput,
    NavigateInput,
    PageScrollAction,
    ScreenshotInput,
    ScrollIntoViewAction,
    TypeAction,
)
from deepagents_browser.state import BrowserState


class _Runtime:
    def __init__(self, state) -> None:
        self.state = state


class _Session:
    async def navigate(self, url, page_ref=None):
        return f"visited:{url}:{page_ref}"

    async def snapshot(self, page_ref=None):
        return json.dumps({"page_ref": page_ref or "p_test", "nodes": []})

    async def act(self, action):
        return json.dumps({"ok": True, "page_ref": "p_test"})

    async def screenshot(self, page_ref=None):
        return ScreenshotResult(page_ref=page_ref or "p_test", media_type="image/png", data=b"png")

    async def tabs(self, operation, page_ref=None):
        return json.dumps({"operation": operation, "page_ref": page_ref})


class _Manager:
    def __init__(self) -> None:
        self.validated = []
        self.session_calls = []
        self.closed = 0
        self.session = _Session()

    async def validate_url(self, url):
        self.validated.append(url)

    @asynccontextmanager
    async def lease_session(self, thread_id):
        self.session_calls.append(thread_id)
        yield self.session

    async def aclose(self):
        self.closed += 1


def _install_fake_blockbuster(monkeypatch) -> ContextVar[bool]:
    skip = ContextVar("blockbuster_skip", default=False)
    module = ModuleType("blockbuster.blockbuster")
    module.blockbuster_skip = skip
    monkeypatch.setitem(sys.modules, "blockbuster.blockbuster", module)
    return skip


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


class _Request:
    def __init__(self, state, tools) -> None:
        self.state = state
        self.tools = tools

    def override(self, *, tools):
        return _Request(self.state, tools)


def test_construction_is_lazy_and_creates_exactly_five_base_tools():
    manager = _Manager()
    middleware = BrowserMiddleware(runtime_manager=manager)
    assert len(middleware.tools) == 5
    assert all(isinstance(tool, BaseTool) for tool in middleware.tools)
    assert [tool.name for tool in middleware.tools] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_act",
        "browser_screenshot",
        "browser_tabs",
    ]
    assert all("runtime" not in tool.args for tool in middleware.tools)
    configs = [cast("Any", tool.args_schema).model_config for tool in middleware.tools]
    assert all(config["extra"] == "forbid" for config in configs)
    assert all(config["strict"] is True for config in configs)
    assert middleware.tools[3].response_format == "content_and_artifact"
    assert manager.validated == []
    assert manager.session_calls == []


def test_private_activation_state_is_strict_boolean_and_private():
    annotation = get_type_hints(BrowserState, include_extras=True)["_browser_enabled"]
    annotated_boolean = annotation.__args__[0]
    assert PrivateStateAttr in annotated_boolean.__metadata__
    assert any(isinstance(item, Strict) for item in annotated_boolean.__metadata__)


def test_strict_schemas_reject_extra_fields_and_unsafe_actions():
    with pytest.raises(ValidationError):
        NavigateInput(url="https://example.com", selector="body")
    with pytest.raises(ValidationError):
        ScreenshotInput(full_page=True)
    with pytest.raises(ValidationError):
        ActInput(action={"kind": "javascript", "ref": "e_12345678"})
    scroll = ActInput(action={"kind": "scroll", "direction": "down"})
    assert isinstance(scroll.action, PageScrollAction)
    assert scroll.action.distance == "half_page"
    into_view = ActInput(action={"kind": "scroll_into_view", "ref": "e_12345678"})
    assert isinstance(into_view.action, ScrollIntoViewAction)
    with pytest.raises(ValidationError):
        ActInput(action={"kind": "scroll", "direction": "down", "ref": "e_12345678"})
    with pytest.raises(ValidationError):
        ActInput(action={"kind": "scroll", "direction": "down", "amount": 100})
    with pytest.raises(ValidationError):
        ActInput(action={"kind": "scroll", "direction": "diagonal"})
    with pytest.raises(ValidationError):
        ActInput(action={"kind": "scroll_into_view", "ref": "e_12345678", "selector": "body"})


def test_sync_wrapper_filters_only_exact_names_when_inactive():
    middleware = BrowserMiddleware(runtime_manager=_Manager())
    request = _Request(
        {},
        [
            _Named("browser_navigate"),
            _Named("browser_navigation"),
            {"name": "browser_snapshot"},
            {"name": "browser_snapshot_extra"},
        ],
    )
    result = middleware.wrap_model_call(request, lambda received: received)
    assert [tool.get("name") if isinstance(tool, dict) else tool.name for tool in result.tools] == [
        "browser_navigation",
        "browser_snapshot_extra",
    ]


async def test_async_wrapper_requires_exact_true():
    middleware = BrowserMiddleware(runtime_manager=_Manager())
    tools = [_Named("browser_navigate"), _Named("other")]

    async def handler(request):
        return request

    inactive = await middleware.awrap_model_call(_Request({"_browser_enabled": 1}, tools), handler)
    active = await middleware.awrap_model_call(_Request({"_browser_enabled": True}, tools), handler)
    assert [tool.name for tool in inactive.tools] == ["other"]
    assert active.tools == tools


async def test_tool_fails_closed_before_manager_access():
    manager = _Manager()
    middleware = BrowserMiddleware(runtime_manager=manager)
    with pytest.raises(BrowserAccessError):
        await middleware.tools[0].coroutine(
            url="https://example.com",
            page_ref=None,
            runtime=_Runtime({"_browser_enabled": "true"}),
        )
    assert manager.validated == []
    assert manager.session_calls == []


async def test_active_navigation_runs_through_langgraph_tool_node():
    manager = _Manager()
    middleware = BrowserMiddleware(runtime_manager=manager)
    builder = StateGraph(cast("Any", BrowserState))
    builder.add_node("tools", ToolNode(middleware.tools))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    result = await graph.ainvoke(
        cast(
            "Any",
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "browser_navigate",
                                "args": {"url": "https://example.com"},
                                "id": "call-browser-navigate",
                                "type": "tool_call",
                            }
                        ],
                    )
                ],
                "_browser_enabled": True,
            },
        ),
        config={"configurable": {"thread_id": "browser-tool-node-test"}},
    )

    assert result["messages"][-1].content == "visited:https://example.com:None"


async def test_screenshot_returns_standard_blocks_and_separate_raw_artifact():
    manager = _Manager()
    middleware = BrowserMiddleware(runtime_manager=manager)
    builder = StateGraph(cast("Any", BrowserState))
    builder.add_node("tools", ToolNode(middleware.tools))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    result = await graph.ainvoke(
        cast(
            "Any",
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "browser_screenshot",
                                "args": {"page_ref": "p_current_page"},
                                "id": "call-browser-screenshot",
                                "type": "tool_call",
                            }
                        ],
                    )
                ],
                "_browser_enabled": True,
            },
        ),
        config={"configurable": {"thread_id": "browser-screenshot-test"}},
    )

    message = result["messages"][-1]
    assert message.content == [
        {
            "type": "text",
            "text": '{"page_ref":"p_current_page","media_type":"image/png","bytes":3}',
        },
        {
            "type": "image",
            "base64": base64.b64encode(b"png").decode("ascii"),
            "mime_type": "image/png",
        },
    ]
    assert "base64" not in message.content[0]["text"]
    assert message.artifact == {
        "page_ref": "p_current_page",
        "media_type": "image/png",
        "bytes": 3,
        "image": {
            "type": "image",
            "base64": base64.b64encode(b"png").decode("ascii"),
            "mime_type": "image/png",
        },
    }
    assert json.loads(json.dumps(message.artifact)) == message.artifact


async def test_tabs_operation_uses_manager_session_lease():
    manager = _Manager()
    middleware = BrowserMiddleware(runtime_manager=manager)

    result = json.loads(
        await middleware.tools[4].coroutine(
            operation="list",
            page_ref=None,
            runtime=_Runtime({"_browser_enabled": True}),
        )
    )

    assert result == {"operation": "list", "page_ref": None}
    assert len(manager.session_calls) == 1


async def test_active_navigation_validates_before_session_access():
    events = []

    class OrderedManager(_Manager):
        async def validate_url(self, url):
            events.append("validate")

        @asynccontextmanager
        async def lease_session(self, thread_id):
            events.append("session")
            yield self.session

    middleware = BrowserMiddleware(runtime_manager=OrderedManager())
    await middleware.tools[0].coroutine(
        url="https://example.com",
        page_ref=None,
        runtime=_Runtime({"_browser_enabled": True}),
    )
    assert events == ["validate", "session"]


async def test_playwright_blockbuster_exemption_is_scoped_to_browser_calls(monkeypatch):
    skip = _install_fake_blockbuster(monkeypatch)
    events = []

    class ScopedSession(_Session):
        async def navigate(self, url, page_ref=None):
            events.append(("navigate", skip.get()))
            return await super().navigate(url, page_ref)

    class ScopedManager(_Manager):
        def __init__(self) -> None:
            super().__init__()
            self.session = ScopedSession()

        async def validate_url(self, url):
            events.append(("validate", skip.get()))

        @asynccontextmanager
        async def lease_session(self, thread_id):
            events.append(("session", skip.get()))
            yield self.session

    middleware = BrowserMiddleware(runtime_manager=ScopedManager())
    result = await middleware.tools[0].coroutine(
        url="https://example.com",
        page_ref=None,
        runtime=_Runtime({"_browser_enabled": True}),
    )

    assert result == "visited:https://example.com:None"
    assert events == [("validate", False), ("session", True), ("navigate", True)]
    assert skip.get() is False


async def test_playwright_blockbuster_exemption_resets_after_error(monkeypatch):
    skip = _install_fake_blockbuster(monkeypatch)

    class FailingSession(_Session):
        async def navigate(self, url, page_ref=None):
            assert skip.get() is True
            msg = "navigation failed"
            raise RuntimeError(msg)

    manager = _Manager()
    manager.session = FailingSession()
    middleware = BrowserMiddleware(runtime_manager=manager)

    with pytest.raises(RuntimeError, match="navigation failed"):
        await middleware.tools[0].coroutine(
            url="https://example.com",
            page_ref=None,
            runtime=_Runtime({"_browser_enabled": True}),
        )

    assert skip.get() is False


async def test_snapshot_retries_once_after_navigation_race():
    class NavigatingSession(_Session):
        def __init__(self) -> None:
            self.calls = 0

        async def snapshot(self, page_ref=None):
            self.calls += 1
            if self.calls == 1:
                msg = "Page navigated while the snapshot was being captured"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.PAGE_NAVIGATED)
            return json.dumps({"page_ref": page_ref, "nodes": []})

    manager = _Manager()
    manager.session = NavigatingSession()
    middleware = BrowserMiddleware(runtime_manager=manager)

    result = json.loads(
        await middleware.tools[1].coroutine(
            page_ref="p_current_page",
            runtime=_Runtime({"_browser_enabled": True}),
        )
    )

    assert result == {"page_ref": "p_current_page", "nodes": []}
    assert manager.session.calls == 2


async def test_repeated_snapshot_navigation_race_returns_recovery_result():
    class NavigatingSession(_Session):
        async def snapshot(self, page_ref=None):
            msg = "Page navigated while the snapshot was being captured"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.PAGE_NAVIGATED)

    manager = _Manager()
    manager.session = NavigatingSession()
    middleware = BrowserMiddleware(runtime_manager=manager)

    result = json.loads(
        await middleware.tools[1].coroutine(
            page_ref="p_current_page",
            runtime=_Runtime({"_browser_enabled": True}),
        )
    )

    assert result == {
        "ok": False,
        "error": {
            "code": "page_navigated",
            "message": "Page navigated while the snapshot was being captured",
        },
        "next": "Retry browser_snapshot after the page finishes loading.",
    }


async def test_successful_action_instructs_model_to_refresh_references():
    middleware = BrowserMiddleware(runtime_manager=_Manager())

    result = json.loads(
        await middleware.tools[2].coroutine(
            action=TypeAction(kind="type", ref="e_current_reference", text="Bali"),
            runtime=_Runtime({"_browser_enabled": True}),
        )
    )

    assert result == {
        "ok": True,
        "page_ref": "p_test",
        "next": "Call browser_snapshot before the next browser_act.",
    }


async def test_stale_action_reference_returns_recovery_instructions_to_model():
    class StaleSession(_Session):
        async def act(self, action):
            msg = "Unknown or stale element reference; take a new snapshot"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.STALE_ELEMENT_REFERENCE)

    manager = _Manager()
    manager.session = StaleSession()
    middleware = BrowserMiddleware(runtime_manager=manager)

    result = json.loads(
        await middleware.tools[2].coroutine(
            action=TypeAction(kind="type", ref="e_stale_reference", text="Bali"),
            runtime=_Runtime({"_browser_enabled": True}),
        )
    )

    assert result == {
        "ok": False,
        "error": {
            "code": "stale_element_reference",
            "message": "Unknown or stale element reference; take a new snapshot",
        },
        "next": "Call browser_snapshot, then retry with a new element reference.",
    }
    assert "browser_snapshot after every successful action" in middleware.tools[2].description


@pytest.mark.parametrize(
    "code",
    [
        BrowserErrorCode.ELEMENT_DISABLED,
        BrowserErrorCode.ELEMENT_NOT_EDITABLE,
        BrowserErrorCode.ACTION_TARGET_MISMATCH,
        BrowserErrorCode.ACTION_TIMEOUT,
        BrowserErrorCode.SCROLL_FAILED,
        BrowserErrorCode.ACTION_FAILED,
    ],
)
async def test_operational_action_errors_are_recoverable(code):
    class OperationalFailureSession(_Session):
        async def act(self, action):
            msg = "Stable operational failure"
            raise BrowserRuntimeError(msg, code=code)

    manager = _Manager()
    manager.session = OperationalFailureSession()
    middleware = BrowserMiddleware(runtime_manager=manager)

    result = json.loads(
        await middleware.tools[2].coroutine(
            action=PageScrollAction(kind="scroll", direction="down"),
            runtime=_Runtime({"_browser_enabled": True}),
        )
    )

    assert result == {
        "ok": False,
        "error": {"code": code.value, "message": "Stable operational failure"},
        "next": "Call browser_snapshot, then retry with a new element reference.",
    }


async def test_nonrecoverable_action_error_remains_fail_closed():
    class SensitiveSession(_Session):
        async def act(self, action):
            msg = "Actions on password or payment controls are blocked"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.SENSITIVE_CONTROL_BLOCKED)

    manager = _Manager()
    manager.session = SensitiveSession()
    middleware = BrowserMiddleware(runtime_manager=manager)

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await middleware.tools[2].coroutine(
            action=TypeAction(kind="type", ref="e_sensitive_ref", text="secret"),
            runtime=_Runtime({"_browser_enabled": True}),
        )

    assert exc_info.value.code == "sensitive_control_blocked"


async def test_playwright_blockbuster_exemption_covers_cleanup_and_resets(monkeypatch):
    skip = _install_fake_blockbuster(monkeypatch)

    class ScopedManager(_Manager):
        async def aclose(self):
            assert skip.get() is True
            await super().aclose()

    manager = ScopedManager()
    middleware = BrowserMiddleware(runtime_manager=manager)

    await middleware.aclose()

    assert manager.closed == 1
    assert skip.get() is False
