---
name: textual-screenshot
description: Capture a Textual terminal UI as an SVG using its headless test harness. Use when asked to make, attach, or preview a screenshot of deepagents-code/dcode or another Textual app, visually verify a TUI state, or render a modal, screen, or widget without a desktop or browser.
license: MIT
compatibility: designed for deepagents-code and Textual apps
---

# Textual Screenshot

Render the real app with Textual's headless test harness, drive it to the requested state, and save the composed terminal as SVG. Prefer this over browser automation or OS-level screenshot tools.

## Capture workflow

1. Identify the `App` class and the shortest trusted local setup that reaches the requested UI.
2. Write a temporary Python script outside the repository. Keep generated screenshots outside the repository unless the user explicitly requests a committed artifact.
3. Start the app with a deterministic terminal size using `app.run_test(size=(columns, rows))`.
4. Call `await pilot.pause()` after startup and after every action that changes visible state. Use `pilot.press(...)` for a realistic interaction path when practical; direct app methods are acceptable for a focused preview.
5. Call `app.save_screenshot(output_path)` while the desired state is visible. Use an `.svg` path.
6. Inspect the resulting SVG and confirm its file size is reasonable before sharing it.
7. Delete temporary scripts and captures when they are no longer needed.

Minimal deepagents-code example:

```python
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from deepagents_code.app import DeepAgentsApp


async def main() -> None:
    output = Path("/tmp/dcode-ui.svg")
    app = DeepAgentsApp(agent=MagicMock())
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        await app._handle_command("/model")
        await pilot.pause()
        app.save_screenshot(output)
    print(output)


asyncio.run(main())
```

Run from `libs/code` so the project environment and editable package resolve:

```bash
uv run python /tmp/capture_dcode_ui.py
```

Adapt only the app constructor and action that opens the target state. For a standalone Textual app, import its `App` subclass and use the same `run_test`/`save_screenshot` sequence.

## Reliability and safety

- Use trusted local application code only. A headless app can still run startup hooks, subprocesses, or network calls; mock or disable unrelated integrations.
- Never capture secrets, credentials, private conversation content, or unrelated user data. Seed only synthetic content needed for the preview.
- Choose a bounded terminal size; start with `(110, 36)` and adjust only when the target clips or wastes substantial space.
- Wait for workers, animations, and modal transitions to settle. Add another `await pilot.pause()` rather than using arbitrary sleeps.
- Do not treat a generated SVG as proof that interaction works. Use normal Textual tests for behavior and the screenshot for visual inspection.
- Do not hand-edit the SVG. Reproduce the intended app state and recapture it.
