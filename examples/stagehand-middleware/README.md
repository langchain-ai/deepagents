# Stagehand Browser Middleware Example

This example runs a basic Deep Agent with `StagehandBrowserToolsMiddleware` and a small Playwright-backed browser adapter.

It demonstrates the middleware-only shape: browser tools + Stagehand-style prompt guidance, without a Stagehand-like `.execute()` wrapper or forced `done` completion semantics.

## Requirements

Environment variables:

- `OPENAI_API_KEY` if using the default `--model openai:gpt-5.5`
- Or the provider key for whichever LangChain model string you pass with `--model`

Python packages:

- `deepagents` from this checkout
- `playwright`
- provider package for your selected model, for example `langchain-openai` for `openai:*`

Browser install:

- Playwright Chromium browser binaries

## Setup from this repository

From the repository root:

```bash
cd libs/deepagents
uv sync --group test
uv pip install playwright langchain-openai
uv run playwright install chromium
```

If you use a non-OpenAI model, install the matching LangChain provider package instead of `langchain-openai` and set the corresponding API key.

## Run

From `libs/deepagents`:

```bash
OPENAI_API_KEY=... uv run python ../../examples/stagehand-middleware/run_stagehand_middleware.py \
  "Open https://example.com and summarize the page" \
  --model openai:gpt-5.5 \
  --mode hybrid \
  --headless
```

For a visible browser window, omit `--headless`.

## Notes

- `--mode hybrid` is the default because the example adapter implements coordinate tools like `click`, `type`, `scroll`, plus `ariaTree`, `screenshot`, `goto`, and `extract`.
- The adapter intentionally does not implement full natural-language `act()` or `fillForm()` semantics. For DOM mode, add your own implementations of `act()` and `fill_form()` on `PlaywrightBrowserAdapter`.
- This is a smoke-test example for the middleware API, not a full Stagehand replacement.
