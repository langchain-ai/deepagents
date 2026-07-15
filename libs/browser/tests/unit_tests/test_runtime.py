import asyncio
import inspect
import json

import pytest

from deepagents_browser.network import NetworkPolicy
from deepagents_browser.runtime import BrowserLimits, BrowserRuntimeError, BrowserRuntimeManager
from deepagents_browser.schemas import ClickAction, TypeAction


async def _public_resolver(host: str, port: int):
    return ["93.184.216.34"]


class FakeElement:
    def __init__(self, *, text="Button", attrs=None) -> None:
        self.text = text
        self.attrs = {"role": "button", "aria-label": "Go", **(attrs or {})}
        self.visible = True
        self.attached = True
        self.actions = []

    async def get_attribute(self, name):
        return self.attrs.get(name)

    async def text_content(self):
        return self.text

    async def is_visible(self):
        return self.visible and self.attached

    async def element_handle(self):
        return self if self.attached else None

    async def count(self):
        return int(self.attached)

    async def click(self):
        self.actions.append(("click", None))

    async def fill(self, text):
        self.actions.append(("fill", text))

    async def press(self, key):
        self.actions.append(("press", key))

    async def select_option(self, value):
        self.actions.append(("select", value))


class FakeLocator:
    def __init__(self, elements) -> None:
        self.elements = elements

    async def count(self):
        return len(self.elements)

    def nth(self, index):
        return self.elements[index]


class FakePage:
    def __init__(self, context, elements=None, screenshot=b"png") -> None:
        self.context = context
        self.url = "about:blank"
        self.elements = elements or []
        self.screenshot_payload = screenshot
        self.timeout = None
        self.main_frame = object()
        self.event_handlers = {}

    def set_default_timeout(self, timeout):
        self.timeout = timeout

    def on(self, event, handler):
        self.event_handlers.setdefault(event, []).append(handler)

    def emit_frame_navigated(self, frame):
        for handler in self.event_handlers.get("framenavigated", []):
            handler(frame)

    async def goto(self, url, *, wait_until):
        self.url = url
        self.emit_frame_navigated(self.main_frame)

    async def title(self):
        return "Fake title"

    def locator(self, selector):
        assert "button" in selector
        return FakeLocator(self.elements)

    async def screenshot(self, *, type, full_page):
        assert type == "png"
        assert full_page is False
        return self.screenshot_payload

    async def close(self):
        self.context.pages.remove(self)


class FakeContext:
    def __init__(self, *, page_elements=None, screenshot=b"png") -> None:
        self.pages = []
        self.page_elements = page_elements
        self.screenshot = screenshot
        self.routes = []
        self.closed = 0

    async def route(self, pattern, handler):
        self.routes.append((pattern, handler))

    async def new_page(self):
        page = FakePage(self, self.page_elements, self.screenshot)
        self.pages.append(page)
        return page

    async def close(self):
        self.closed += 1
        self.pages.clear()


class PlaywrightWrappingContext(FakeContext):
    async def route(self, pattern, handler):
        if inspect.ismethod(handler):
            setattr(handler.__self__, f"_pw_impl_instance_{handler.__name__}", handler)
        else:
            handler._pw_impl_instance = handler
        await super().route(pattern, handler)


class FakeBrowser:
    def __init__(self, context_factory=None) -> None:
        self.context_factory = context_factory or FakeContext
        self.contexts = []
        self.kwargs = []
        self.closed = 0

    async def new_context(self, **kwargs):
        self.kwargs.append(kwargs)
        context = self.context_factory()
        self.contexts.append(context)
        return context

    async def close(self):
        self.closed += 1


class FakePlaywright:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop(self):
        self.stopped += 1


def make_manager(*, browser=None, limits=None):
    playwright = FakePlaywright()
    fake_browser = browser or FakeBrowser()
    starts = []

    async def factory():
        starts.append("start")
        await asyncio.sleep(0)
        return playwright, fake_browser

    manager = BrowserRuntimeManager(
        limits=limits,
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    return manager, playwright, fake_browser, starts


async def test_route_handler_supports_playwright_callback_wrapping():
    browser = FakeBrowser(context_factory=PlaywrightWrappingContext)
    manager, _, _, _ = make_manager(browser=browser)

    await manager.get_session("thread")

    assert len(browser.contexts[0].routes) == 1


async def test_runtime_is_lazy_single_flight_isolated_and_bounded():
    manager, _, browser, starts = make_manager(limits=BrowserLimits(max_contexts=2))
    assert starts == []
    sessions = await asyncio.gather(*(manager.get_session("one") for _ in range(10)))
    assert len(starts) == 1
    assert len(browser.contexts) == 1
    assert len({id(session) for session in sessions}) == 1
    assert browser.kwargs[0]["service_workers"] == "block"
    assert browser.kwargs[0]["accept_downloads"] is False
    assert await manager.get_session("two") is not sessions[0]
    with pytest.raises(BrowserRuntimeError, match="context limit"):
        await manager.get_session("three")


def test_limits_enforce_hard_caps():
    with pytest.raises(ValueError, match="max_contexts"):
        BrowserLimits(max_contexts=65)
    with pytest.raises(ValueError, match="max_requests_per_context"):
        BrowserLimits(max_requests_per_context=10_001)


async def test_navigation_and_tabs_are_bounded():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_tabs_per_context=1))
    session = await manager.get_session("thread")
    result = json.loads(await session.navigate("https://example.com"))
    assert result["url"] == "https://example.com"
    assert browser.contexts[0].pages[0].timeout == 15_000
    with pytest.raises(BrowserRuntimeError, match="Tab limit"):
        await session.tabs("new")


async def test_request_budget_fails_closed():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_requests_per_context=1))
    await manager.get_session("thread")
    handler = browser.contexts[0].routes[0][1]

    class Request:
        url = "https://example.com"

    class Route:
        request = Request()

        def __init__(self) -> None:
            self.continued = 0
            self.aborted = 0

        async def continue_(self):
            self.continued += 1

        async def abort(self, error_code="failed"):
            assert error_code == "blockedbyclient"
            self.aborted += 1

    allowed, denied = Route(), Route()
    await handler(allowed)
    await handler(denied)
    assert allowed.continued == 1
    assert denied.aborted == 1


async def test_snapshot_refs_are_opaque_stale_and_fingerprint_checked():
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    await session.navigate("https://example.com")
    snapshot = json.loads(await session.snapshot())
    ref = snapshot["nodes"][0]["ref"]
    assert ref.startswith(f"e_{snapshot['generation']}_")
    await session.act(ClickAction(kind="click", ref=ref))
    with pytest.raises(BrowserRuntimeError, match="stale"):
        await session.act(ClickAction(kind="click", ref=ref))

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    element.text = "changed"
    with pytest.raises(BrowserRuntimeError, match="changed"):
        await session.act(ClickAction(kind="click", ref=ref))

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    element.attached = False
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "element_identity_unavailable"


async def test_element_refs_keep_identity_across_reorder_and_reject_replacement():
    first = FakeElement(text="same")
    second = FakeElement(text="same")
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[first, second]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    snapshot = json.loads(await session.snapshot())
    first_ref = snapshot["nodes"][0]["ref"]
    page = browser.contexts[0].pages[0]

    page.elements[:] = [second, first]
    await session.act(ClickAction(kind="click", ref=first_ref))
    assert first.actions == [("click", None)]
    assert second.actions == []

    replacement_ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    replaced = page.elements[0]
    replaced.attached = False
    page.elements[0] = FakeElement(text="same")
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=replacement_ref))
    assert exc_info.value.code == "element_identity_unavailable"
    assert page.elements[0].actions == []


async def test_sensitive_controls_are_omitted_and_blocked_at_action_time():
    password = FakeElement(attrs={"type": "password", "name": "password"})
    card = FakeElement(attrs={"autocomplete": "section-checkout cc-number"})
    card_without_autocomplete = FakeElement(attrs={"id": "credit-card-number"})
    safe = FakeElement(attrs={"type": "text", "name": "search"})
    browser = FakeBrowser(
        context_factory=lambda: FakeContext(
            page_elements=[password, card, card_without_autocomplete, safe]
        )
    )
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    snapshot = json.loads(await session.snapshot())
    assert len(snapshot["nodes"]) == 1
    assert snapshot["nodes"][0]["label"] == "Go"

    ref = snapshot["nodes"][0]["ref"]
    safe.attrs["type"] = "password"
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(TypeAction(kind="type", ref=ref, text="secret"))
    assert exc_info.value.code == "sensitive_control_blocked"
    assert safe.actions == []


async def test_only_top_level_navigation_invalidates_element_refs():
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    page = browser.contexts[0].pages[0]
    page.emit_frame_navigated(object())
    await session.act(ClickAction(kind="click", ref=ref))

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    page.emit_frame_navigated(page.main_frame)
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "navigation_invalidated_reference"


async def test_new_element_failure_paths_have_stable_codes():
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    element.text = "changed"
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "element_changed"

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref="e_unknown_reference"))
    assert exc_info.value.code == "stale_element_reference"


async def test_tab_refs_are_opaque_and_stale_refs_fail():
    manager, _, _, _ = make_manager()
    session = await manager.get_session("thread")
    tabs = json.loads(await session.tabs("new"))["tabs"]
    page_ref = tabs[0]["page_ref"]
    assert page_ref.startswith("p_")
    await session.tabs("close", page_ref)
    with pytest.raises(BrowserRuntimeError, match="stale"):
        await session.tabs("select", page_ref)


def _oversized_context():
    return FakeContext(
        page_elements=[FakeElement(text="x" * 300) for _ in range(20)],
        screenshot=b"x" * 11,
    )


async def test_snapshot_and_screenshot_outputs_are_bounded():
    browser = FakeBrowser(context_factory=_oversized_context)
    limits = BrowserLimits(
        max_snapshot_nodes=20,
        max_snapshot_chars=600,
        max_screenshot_bytes=10,
    )
    manager, _, _, _ = make_manager(browser=browser, limits=limits)
    session = await manager.get_session("thread")
    snapshot = await session.snapshot()
    assert len(snapshot) <= 600
    assert json.loads(snapshot)["truncated"] is True
    with pytest.raises(BrowserRuntimeError, match="Screenshot exceeds"):
        await session.screenshot()


async def test_screenshot_rejects_projected_response_before_base64_allocation(monkeypatch):
    browser = FakeBrowser(context_factory=lambda: FakeContext(screenshot=b"x" * 100))
    limits = BrowserLimits(
        max_screenshot_bytes=100,
        max_screenshot_output_chars=100,
    )
    manager, _, _, _ = make_manager(browser=browser, limits=limits)
    session = await manager.get_session("thread")
    await session.tabs("new")

    def unexpected_encode(_payload):
        pytest.fail("base64 allocation should not occur for an oversized response")

    monkeypatch.setattr("deepagents_browser.runtime.base64.b64encode", unexpected_encode)
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.screenshot()
    assert exc_info.value.code == "screenshot_too_large"


async def test_runtime_startup_is_bounded_retryable_and_coded():
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)
        return FakePlaywright(), FakeBrowser()

    manager = BrowserRuntimeManager(
        limits=BrowserLimits(startup_timeout_seconds=0.01),
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.get_session("first")
    assert exc_info.value.code == "startup_timeout"

    session = await manager.get_session("second")
    assert session is not None
    assert calls == 2


async def test_invalid_runtime_factory_result_is_rejected_and_cleaned_up():
    playwright = FakePlaywright()

    async def factory():
        return playwright, object()

    manager = BrowserRuntimeManager(
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.get_session("thread")
    assert exc_info.value.code == "invalid_runtime_factory"
    assert playwright.stopped == 1
    assert manager._playwright is None
    assert manager._browser is None


async def test_runtime_factory_failure_preserves_cause_and_code():
    failure = OSError("launch failed")

    async def factory():
        raise failure

    manager = BrowserRuntimeManager(
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.get_session("thread")
    assert exc_info.value.code == "startup_failed"
    assert exc_info.value.__cause__ is failure


async def test_cleanup_is_idempotent():
    manager, playwright, browser, _ = make_manager()
    await manager.get_session("thread")
    context = browser.contexts[0]
    await asyncio.gather(manager.aclose(), manager.aclose())
    await manager.aclose()
    assert context.closed == 1
    assert browser.closed == 1
    assert playwright.stopped == 1
