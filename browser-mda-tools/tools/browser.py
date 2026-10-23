"""Typed browser tools backed by the `browse` CLI inside the agent's sandbox.

Every tool shells out through `runtime.backend.aexecute`, so the browser stays
sandbox-local -- but the model never writes shell. That matters for more than
tidiness: values lifted off a page routinely contain quotes, `$`, backticks and
semicolons, and a model pasting them into a command string is both a correctness
bug and an injection risk. Here every interpolated value goes through
`shlex.quote`, in one place, rather than relying on an instruction to remember.

The tools also trim and parse what `browse` prints. Raw output is JSON wrapped in
exit-code chatter; returning it verbatim spends context on punctuation.
"""

from __future__ import annotations

import json
import shlex

from langchain.tools import tool
from managed_deepagents import ManagedDeepAgentRuntime

# A page of markdown is the single most useful thing the agent reads, but a long
# article can be enormous. Cap it and say so, so the model knows to narrow rather
# than assume it saw everything.
_MAX_CHARS = 20_000


async def _sh(runtime: ManagedDeepAgentRuntime, command: str, timeout: int = 120) -> str:
    """Run one command in the sandbox and return its output, or raise with it."""
    if runtime.backend is None:
        raise RuntimeError("this agent requires a sandbox; none is configured")
    result = await runtime.backend.aexecute(command, timeout=timeout)
    output = (result.output or "").strip()
    if result.exit_code not in (0, None):
        raise RuntimeError(f"command failed (exit {result.exit_code}): {output[:600]}")
    if result.truncated:
        output += "\n[output truncated by the sandbox]"
    return output


def _json(output: str) -> object:
    """`browse` prints JSON; fall back to raw text when it prints something else."""
    try:
        return json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return output


def _clip(text: str) -> str:
    if len(text) <= _MAX_CHARS:
        return text
    return (
        text[:_MAX_CHARS]
        + f"\n\n[truncated at {_MAX_CHARS} characters -- narrow with a selector "
        "or ask for a specific part of the page]"
    )


@tool(parse_docstring=True)
async def navigate(url: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Open a URL in the browser. Use this for every navigation.

    Args:
        url: Absolute URL to open, including the scheme.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    data = _json(await _sh(runtime, f"web-open {shlex.quote(url)}", timeout=180))
    if isinstance(data, dict):
        return f"Opened {data.get('url', url)} -- {data.get('title', '(no title)')}"
    return str(data)[:500]


@tool(parse_docstring=True)
async def read_page(runtime: ManagedDeepAgentRuntime, selector: str = "") -> str:
    """Read the current page as markdown text. Prefer this over snapshot/click when you only need information.

    Args:
        selector: Optional CSS selector to read just one part of the page.
    """
    cmd = "browse get markdown"
    if selector:
        cmd = f"browse get text {shlex.quote(selector)}"
    data = _json(await _sh(runtime, cmd))
    if isinstance(data, dict):
        data = data.get("markdown") or data.get("text") or json.dumps(data)
    return _clip(str(data))


@tool(parse_docstring=True)
async def snapshot(runtime: ManagedDeepAgentRuntime, full: bool = False) -> str:
    """List the interactive elements on the page with refs like @0-4, for clicking and filling.

    Refs go stale whenever the page changes -- take a fresh snapshot after any
    navigation or submit rather than reusing an old ref.

    Args:
        full: Also return XPath and URL maps, when refs alone are ambiguous.
    """
    out = await _sh(runtime, "browse snapshot" + (" --full" if full else ""))
    return _clip(out)


@tool(parse_docstring=True)
async def click(ref: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Click an element.

    Args:
        ref: A snapshot ref such as @0-4, an XPath, or a CSS selector.
    """
    return str(_json(await _sh(runtime, f"browse click {shlex.quote(ref)}")))[:400]


@tool(parse_docstring=True)
async def fill(ref: str, value: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Type a value into an input.

    Args:
        ref: A snapshot ref such as @0-7, an XPath, or a CSS selector.
        value: Text to enter. Any characters are safe; quoting is handled here.
    """
    cmd = f"browse fill {shlex.quote(ref)} {shlex.quote(value)}"
    return str(_json(await _sh(runtime, cmd)))[:400]


@tool(parse_docstring=True)
async def press(key: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Press a key, e.g. Enter or Escape.

    Args:
        key: Key name to press.
    """
    return str(_json(await _sh(runtime, f"browse press {shlex.quote(key)}")))[:300]


@tool(parse_docstring=True)
async def select_option(ref: str, value: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Choose an option in a dropdown.

    Args:
        ref: A snapshot ref, XPath, or CSS selector for the select element.
        value: Visible label of the option to choose.
    """
    cmd = f"browse select {shlex.quote(ref)} {shlex.quote(value)}"
    return str(_json(await _sh(runtime, cmd)))[:400]


@tool(parse_docstring=True)
async def wait_for(runtime: ManagedDeepAgentRuntime, selector: str = "", seconds: float = 0) -> str:
    """Wait for the page to settle before reading it.

    Prefer a selector over a fixed delay: it returns as soon as the element is
    there, and is more reliable when the page is slow.

    Args:
        selector: CSS selector to wait for.
        seconds: Fixed delay instead, when no selector will do.
    """
    if selector:
        cmd = f"browse wait selector {shlex.quote(selector)}"
    elif seconds:
        cmd = f"browse wait timeout {int(seconds * 1000)}"
    else:
        cmd = "browse wait load"
    return str(_json(await _sh(runtime, cmd)))[:300]


@tool(parse_docstring=True)
async def current_url(runtime: ManagedDeepAgentRuntime) -> str:
    """Report the URL the browser is actually on. Use to confirm where a click landed."""
    return str(_json(await _sh(runtime, "browse get url")))[:400]


@tool(parse_docstring=True)
async def run_javascript(expression: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Evaluate a JavaScript expression on the page and return its result.

    Often the precise route when a page exposes its own API, and far cheaper than
    aiming at a canvas with the mouse.

    Args:
        expression: JavaScript expression to evaluate.
    """
    cmd = f"browse eval {shlex.quote(expression)}"
    return _clip(str(_json(await _sh(runtime, cmd))))


@tool(parse_docstring=True)
async def screenshot(runtime: ManagedDeepAgentRuntime) -> list[dict]:
    """Take a screenshot and return it so you can see the page.

    Only for things with no DOM to read -- canvas, maps, charts. A screenshot is a
    slow, expensive call, so reach for `read_page` first.
    """
    path = "/workspace/_shot.png"
    await _sh(runtime, f"browse screenshot --path {shlex.quote(path)}", timeout=180)
    shot = (await runtime.backend.adownload_files([path]))[0]
    if shot.error or shot.content is None:
        raise RuntimeError(shot.error or "screenshot could not be read back")
    import base64

    encoded = base64.b64encode(shot.content).decode("ascii")
    return [
        {"type": "text", "text": "Screenshot of the current page:"},
        {"type": "image", "url": f"data:image/png;base64,{encoded}"},
    ]


@tool(parse_docstring=True)
async def mouse(
    action: str,
    x: float,
    y: float,
    runtime: ManagedDeepAgentRuntime,
    to_x: float = 0,
    to_y: float = 0,
) -> str:
    """Click, drag, or scroll at pixel coordinates, for canvas and maps with no DOM target.

    Screenshot first to judge coordinates. Budget these tightly -- predict where
    the target is from what you already know rather than hunting pixel by pixel.

    Args:
        action: One of click, hover, drag, scroll.
        x: Horizontal pixel, from the left edge.
        y: Vertical pixel, from the top edge.
        to_x: Destination horizontal pixel, for drag.
        to_y: Destination vertical pixel, for drag.
    """
    if action not in ("click", "hover", "drag", "scroll"):
        raise ValueError("action must be click, hover, drag or scroll")
    nums = [x, y] + ([to_x, to_y] if action == "drag" else [])
    if not all(isinstance(v, (int, float)) and v == v for v in nums):
        raise ValueError("coordinates must be finite numbers")
    coords = " ".join(str(int(v)) for v in nums)
    return str(_json(await _sh(runtime, f"browse mouse {action} {coords}")))[:300]


BROWSER_TOOLS = [
    navigate, read_page, snapshot, click, fill, press, select_option,
    wait_for, current_url, run_javascript, screenshot, mouse,
]
