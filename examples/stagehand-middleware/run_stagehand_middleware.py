"""Run a basic Deep Agent with StagehandBrowserToolsMiddleware and Playwright."""

from __future__ import annotations

import argparse
from typing import Any

from deepagents import StagehandBrowserToolsMiddleware, create_deep_agent
from playwright.sync_api import sync_playwright


class PlaywrightBrowserAdapter:
    """Small adapter exposing the method names StagehandBrowserToolsMiddleware expects."""

    def __init__(self, page: Any) -> None:
        self.page = page

    def current_url(self) -> str:
        return self.page.url

    def goto(self, url: str) -> str:
        self.page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {self.page.url}"

    def aria_tree(self, instruction: str | None = None) -> str:
        del instruction
        snapshot = self.page.accessibility.snapshot(interesting_only=True)
        return _format_accessibility_tree(snapshot)

    def screenshot(self) -> bytes:
        return self.page.screenshot(full_page=False, type="jpeg", quality=60)

    def scroll(self, direction: str | None = None, pixels: int = 700) -> str:
        dx, dy = 0, 0
        if direction == "up":
            dy = -pixels
        elif direction == "down" or direction is None:
            dy = pixels
        elif direction == "left":
            dx = -pixels
        elif direction == "right":
            dx = pixels
        self.page.mouse.wheel(dx, dy)
        return f"Scrolled {direction or 'down'} by {pixels}px"

    def click(self, x: int, y: int) -> str:
        self.page.mouse.click(x, y)
        return f"Clicked at ({x}, {y})"

    def type(self, text: str, x: int | None = None, y: int | None = None) -> str:
        if x is not None and y is not None:
            self.page.mouse.click(x, y)
        self.page.keyboard.type(text)
        return f"Typed {len(text)} characters"

    def key_press(self, keys: str) -> str:
        self.page.keyboard.press(keys)
        return f"Pressed {keys}"

    def go_back(self) -> str:
        self.page.go_back(wait_until="domcontentloaded")
        return f"Navigated back to {self.page.url}"

    def act(self, action: str) -> str:
        raise NotImplementedError(
            "This example adapter does not implement natural-language act(). "
            "Use mode='hybrid' for coordinate tools, or add your own act() implementation. "
            f"Requested action: {action}"
        )

    def fill_form(self, instruction: str) -> str:
        raise NotImplementedError(
            "This example adapter does not implement fill_form(). "
            f"Requested instruction: {instruction}"
        )

    def extract(self, instruction: str) -> str:
        text = self.page.locator("body").inner_text(timeout=5000)
        return f"Extraction request: {instruction}\n\nPage text:\n{text[:8000]}"


def _format_accessibility_tree(node: dict[str, Any] | None, indent: int = 0) -> str:
    if node is None:
        return "No accessibility snapshot available."
    role = node.get("role", "unknown")
    name = node.get("name")
    value = node.get("value")
    line = "  " * indent + f"- {role}"
    if name:
        line += f": {name}"
    if value:
        line += f" = {value}"
    children = node.get("children") or []
    return "\n".join([line, *[_format_accessibility_tree(child, indent + 1) for child in children]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Try StagehandBrowserToolsMiddleware with Playwright.")
    parser.add_argument("instruction", nargs="?", default="Open https://example.com and summarize the page.")
    parser.add_argument("--model", default="openai:gpt-5.5")
    parser.add_argument("--mode", choices=["dom", "hybrid"], default="hybrid")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        adapter = PlaywrightBrowserAdapter(page)

        agent = create_deep_agent(
            model=args.model,
            middleware=[
                StagehandBrowserToolsMiddleware(
                    browser=adapter,
                    mode=args.mode,
                    tool_timeout=30,
                )
            ],
        )

        result = agent.invoke({"messages": [{"role": "user", "content": args.instruction}]})
        print(result["messages"][-1].text())
        browser.close()


if __name__ == "__main__":
    main()
