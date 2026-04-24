"""GPT-5.4 computer-use demo on Browserbase.

Runs a single agent session against a URL, executing the model's computer-use
actions on a cloud Chromium via Playwright. The Browserbase session replay
URL is printed up front and again at the end so the run is shareable.

Env:
    OPENAI_API_KEY: required.
    BROWSERBASE_API_KEY: required.
    BROWSERBASE_PROJECT_ID: required.
    CUA_MODEL: override default model (`gpt-5.4`).
    CUA_MAX_TURNS: override default agent turn cap (`40`).
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from typing import Any

from browserbase import Browserbase
from openai import BadRequestError, OpenAI
from playwright.sync_api import Page, sync_playwright

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_MAX_TURNS = 40
DEFAULT_URL = "https://www.google.com"

_KEY_MAP: dict[str, str] = {
    "CTRL": "Control", "CONTROL": "Control",
    "ALT": "Alt", "OPTION": "Alt",
    "SHIFT": "Shift",
    "CMD": "Meta", "META": "Meta", "SUPER": "Meta", "WIN": "Meta",
    "ENTER": "Enter", "RETURN": "Enter",
    "ESC": "Escape", "ESCAPE": "Escape",
    "TAB": "Tab", "BACKSPACE": "Backspace", "DELETE": "Delete",
    "SPACE": " ",
    "UP": "ArrowUp", "ARROW_UP": "ArrowUp", "ARROWUP": "ArrowUp",
    "DOWN": "ArrowDown", "ARROW_DOWN": "ArrowDown", "ARROWDOWN": "ArrowDown",
    "LEFT": "ArrowLeft", "ARROW_LEFT": "ArrowLeft", "ARROWLEFT": "ArrowLeft",
    "RIGHT": "ArrowRight", "ARROW_RIGHT": "ArrowRight", "ARROWRIGHT": "ArrowRight",
    "HOME": "Home", "END": "End", "PAGEUP": "PageUp", "PAGEDOWN": "PageDown",
}


def _map_key(key: str) -> str:
    """Translate a CUA-style key name to the Playwright equivalent."""
    return _KEY_MAP.get(key.upper(), key)


def _coerce(item: Any) -> dict[str, Any]:
    """Convert an SDK pydantic model (or already-a-dict) into a plain dict."""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return dict(item)


def dispatch(action: dict[str, Any], page: Page) -> None:
    """Execute one computer-use action against a Playwright page.

    Screenshot actions are no-ops here since the caller always captures a
    fresh screenshot after the action batch completes.

    Args:
        action: Action payload from a `computer_call` item.
        page: Playwright page the action targets.
    """
    kind = action.get("type")
    if kind == "click":
        page.mouse.click(action["x"], action["y"], button=action.get("button", "left"))
    elif kind == "double_click":
        page.mouse.dblclick(action["x"], action["y"])
    elif kind == "move":
        page.mouse.move(action["x"], action["y"])
    elif kind == "drag":
        path = action.get("path", [])
        if not path:
            return
        page.mouse.move(path[0]["x"], path[0]["y"])
        page.mouse.down()
        for pt in path[1:]:
            page.mouse.move(pt["x"], pt["y"])
        page.mouse.up()
    elif kind == "scroll":
        page.mouse.move(action["x"], action["y"])
        page.mouse.wheel(action.get("scroll_x", 0), action.get("scroll_y", 0))
    elif kind == "keypress":
        keys = [_map_key(k) for k in action["keys"]]
        page.keyboard.press("+".join(keys))
    elif kind == "type":
        page.keyboard.type(action["text"])
    elif kind == "wait":
        time.sleep(1)
    elif kind == "screenshot":
        return
    else:
        print(f"[warn] unhandled action: {kind}", file=sys.stderr)


def _screenshot(page: Page) -> str:
    """Return a base64-encoded PNG screenshot of the page viewport."""
    return base64.b64encode(page.screenshot(type="png", full_page=False)).decode("ascii")


def _actions_from_call(call: Any) -> list[dict[str, Any]]:
    """Extract the action list from a `computer_call`, tolerating old + new shapes.

    GPT-5.4 returns `actions` (plural list). `computer-use-preview` returns a
    singular `action`. Normalize both into a list of plain dicts.
    """
    batch = getattr(call, "actions", None)
    if batch:
        return [_coerce(a) for a in batch]
    single = getattr(call, "action", None)
    if single is not None:
        return [_coerce(single)]
    return []


def _build_create(oai: OpenAI) -> Any:
    """Return a `responses.create` wrapper that surfaces the real 400 body."""
    def create(**kwargs: Any) -> Any:
        try:
            return oai.responses.create(**kwargs)
        except BadRequestError as exc:
            body = getattr(exc, "body", None)
            print(f"\n[openai 400] body={body}", file=sys.stderr)
            raise
    return create


def run_demo(task: str, url: str, *, model: str, max_turns: int) -> str:
    """Drive `url` with an agent pursuing `task`, return the replay URL.

    Args:
        task: Natural-language instruction for the agent.
        url: Starting URL the browser loads before the agent takes over.
        model: OpenAI model id with computer-use access.
        max_turns: Hard cap on response turns to prevent runaway sessions.

    Returns:
        The Browserbase session replay URL.
    """
    bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])
    session = bb.sessions.create(project_id=os.environ["BROWSERBASE_PROJECT_ID"])
    replay = f"https://browserbase.com/sessions/{session.id}"
    print(f"Live view / replay: {replay}\n")

    oai = OpenAI()
    create = _build_create(oai)
    tool: dict[str, Any] = {"type": "computer"}

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(session.connect_url)
        ctx = browser.contexts[0]
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url, wait_until="domcontentloaded")
        print(f"[viewport] {page.viewport_size}")

        resp = create(
            model=model,
            tools=[tool],
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": task}],
                }
            ],
            truncation="auto",
        )

        for turn in range(1, max_turns + 1):
            calls = [o for o in resp.output if o.type == "computer_call"]
            if not calls:
                text = " ".join(
                    c.text
                    for o in resp.output if o.type == "message"
                    for c in o.content if c.type == "output_text"
                )
                if text:
                    print(f"\nagent: {text}")
                break

            call = calls[0]
            actions = _actions_from_call(call)
            if not actions:
                print("[warn] computer_call had no actions — stopping")
                break
            for action in actions:
                print(f"[turn {turn}] {action.get('type')} {action}")
                dispatch(action, page)
                page.wait_for_timeout(250)

            output: dict[str, Any] = {
                "type": "computer_call_output",
                "call_id": call.call_id,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{_screenshot(page)}",
                },
            }
            checks = getattr(call, "pending_safety_checks", None)
            if checks:
                output["acknowledged_safety_checks"] = [
                    {"id": c.id, "code": c.code, "message": c.message} for c in checks
                ]

            resp = create(
                model=model,
                previous_response_id=resp.id,
                tools=[tool],
                input=[output],
                truncation="auto",
            )
        else:
            print(f"[warn] hit max_turns={max_turns}")

        browser.close()

    print(f"\nReplay: {replay}")
    return replay


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="GPT CUA demo via Browserbase")
    parser.add_argument("task", help="What the agent should do.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Starting URL.")
    parser.add_argument(
        "--model",
        default=os.environ.get("CUA_MODEL", DEFAULT_MODEL),
        help="OpenAI model id with computer-use access.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("CUA_MAX_TURNS", DEFAULT_MAX_TURNS)),
        help="Cap on agent response turns.",
    )
    args = parser.parse_args()
    run_demo(args.task, args.url, model=args.model, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
