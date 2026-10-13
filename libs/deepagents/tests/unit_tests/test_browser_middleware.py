"""The browser tools must stay a faithful port of Stagehand's facade contract.

Models are prompted against these descriptions, and Stagehand pins them
string-exact for that reason. A reworded description or a renamed action field
makes this agent run a different contract from every published benchmark of
it, with no test failure to say so -- hence this file.

Reference: `packages/integrations/core/src/facade/contract.ts` in
browserbase/stagehand.
"""

import pytest

from deepagents.middleware.browser import (
    _ACTION_FIELDS,
    CDP,
    Browserbase,
    BrowserContext,
    BrowserMiddleware,
)

RUN = (
    "Browse and automate websites in the persistent browser by executing JavaScript "
    "against a Playwright-shaped API. The code runs inside an async function with "
    "page, context, and browser in scope (Playwright Page, BrowserContext, and "
    "Browser); use await directly and return a JSON-serializable value when useful. "
    'Navigate with await page.goto("https://example.com"); there is no separate '
    "navigate or start tool. Alternatively, provide a batch of actions using IDs from "
    "the latest snapshot. Provide exactly one of code or actions. Each action must use "
    '"op" (never "kind") and "id" (never "ref"). Copy the bracketed snapshot ID as a '
    'string. Examples: {"actions":[{"op":"click","id":"1-42"}]}, '
    '{"actions":[{"op":"fill","id":"2-14","value":"Miami"}]}, '
    '{"actions":[{"op":"select","id":"3-9","values":"Lowest price"}]}.'
)
SNAPSHOT = (
    "Capture the active page's Stagehand accessibility tree and hydrate its displayed "
    "IDs for subsequent run actions. Every call replaces the active page's ID map."
)
SCREENSHOT = (
    "Capture a screenshot of the active page. For size-constrained MCP clients, prefer "
    'a viewport JPEG: {"type":"jpeg","quality":40,"fullPage":false}.'
)


def _tools(**kwargs: object) -> dict:
    return {tool.name: tool for tool in BrowserMiddleware(Browserbase(), **kwargs).tools}


def test_tool_names_match_the_facade() -> None:
    assert set(_tools()) == {"snapshot", "run", "screenshot"}


def test_descriptions_are_the_contract_verbatim() -> None:
    tools = _tools()
    assert tools["run"].description.strip() == RUN
    assert tools["snapshot"].description.strip() == SNAPSHOT
    assert tools["screenshot"].description.strip() == SCREENSHOT


def test_arguments_match_the_wire_schema() -> None:
    tools = _tools()
    assert list(tools["snapshot"].args) == ["includeIframes"]
    assert list(tools["run"].args) == ["code", "actions"]
    assert list(tools["screenshot"].args) == ["fullPage", "type", "quality"]


def test_action_fields_are_the_contract_names() -> None:
    """`type` carries `text` and `press` carries `key`, not a shared `value`."""
    assert _ACTION_FIELDS == {"fill": "value", "type": "text", "press": "key"}


@pytest.mark.asyncio
async def test_cdp_does_not_close_a_browser_it_attached_to() -> None:
    """`browser.close()` on a connected browser sends `Browser.close`."""

    class _Spy:
        closed = False

        async def close(self) -> None:
            self.spy_closed = True

    browser = _Spy()
    ctx = BrowserContext(thread_id="t", runtime=None)
    await CDP(url="ws://127.0.0.1:9222/x").close(browser, ctx)
    assert not hasattr(browser, "spy_closed")

    await Browserbase().close(browser, ctx)
    assert browser.spy_closed is True


def test_javascript_can_be_disabled() -> None:
    assert BrowserMiddleware(Browserbase(), allow_javascript=False).allow_javascript is False
