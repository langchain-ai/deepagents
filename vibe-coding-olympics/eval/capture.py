"""Headless page capture: HTML + screenshot, with optional axe-core audit.

Runs the axe accessibility audit in the same Playwright session as the
screenshot to avoid booting Chromium twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AXE_CDN_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.0/axe.min.js"

_VIEWPORT_WIDTH = 1280
_VIEWPORT_HEIGHT = 900
_NAV_TIMEOUT_MS = 15_000
_AXE_TIMEOUT_MS = 20_000


@dataclass
class AccessibilityReport:
    """Subset of axe-core results we care about for scoring."""

    violations: list[dict[str, Any]] = field(default_factory=list)
    passes_count: int = 0
    incomplete_count: int = 0

    @property
    def violation_count(self) -> int:
        """Total number of WCAG violations detected."""
        return len(self.violations)

    @property
    def serious_violation_count(self) -> int:
        """Count of violations with impact `serious` or `critical`."""
        return sum(
            1
            for v in self.violations
            if v.get("impact") in {"serious", "critical"}
        )


@dataclass
class Capture:
    """Result of a single-site capture."""

    html: str
    screenshot_png: bytes
    accessibility: AccessibilityReport | None = None


async def capture_site(url: str, *, run_axe: bool = True) -> Capture:
    """Capture HTML, a full-page screenshot, and an optional axe audit.

    Uses a headless Chromium browser at 1280x900. Waits for network idle
    before capturing (15s timeout). If `run_axe` is true, injects axe-core
    from CDN and runs it against the loaded page.

    Args:
        url: The website URL to capture.
        run_axe: If `True`, run the axe-core accessibility audit. Audit
            failures are swallowed — the returned `Capture` simply has
            `accessibility=None`.

    Returns:
        `Capture` with html, PNG bytes, and optionally an
        `AccessibilityReport`.

    Raises:
        RuntimeError: If page navigation or screenshot capture fails.
    """
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page(
                    viewport={
                        "width": _VIEWPORT_WIDTH,
                        "height": _VIEWPORT_HEIGHT,
                    },
                )
                await page.goto(
                    url, wait_until="networkidle", timeout=_NAV_TIMEOUT_MS
                )
                html = await page.content()
                png = await page.screenshot(full_page=True)

                accessibility: AccessibilityReport | None = None
                if run_axe:
                    accessibility = await _run_axe(page)
            finally:
                await browser.close()
    except Exception as exc:
        msg = f"Failed to capture {url}: {exc}"
        raise RuntimeError(msg) from exc
    else:
        return Capture(
            html=html, screenshot_png=png, accessibility=accessibility
        )


async def _run_axe(page: Any) -> AccessibilityReport | None:
    """Inject axe-core from CDN and run a full-page audit.

    Swallows all errors — accessibility scoring is best-effort signal, not
    a hard requirement. Callers see `None` if the audit could not run.

    Args:
        page: An active Playwright `Page` already navigated to the target.

    Returns:
        `AccessibilityReport` on success, `None` on any failure.
    """
    try:
        await page.add_script_tag(url=AXE_CDN_URL)
        raw = await page.evaluate(
            """async () => {
                const result = await axe.run(document, {
                    runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] },
                });
                return {
                    violations: result.violations,
                    passes: result.passes.length,
                    incomplete: result.incomplete.length,
                };
            }""",
            timeout=_AXE_TIMEOUT_MS,
        )
    except Exception:
        return None
    return AccessibilityReport(
        violations=raw.get("violations", []),
        passes_count=raw.get("passes", 0),
        incomplete_count=raw.get("incomplete", 0),
    )
