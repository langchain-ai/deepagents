# Browser agent with `create_agent`

A minimal Playwright browser agent built with LangChain's [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents). It adapts the browser loop from [`ndrezn/ts-browser-agent`](https://github.com/ndrezn/ts-browser-agent): instead of a custom `Agent.run()` loop choosing each operation, `create_agent` orchestrates typed browser tools directly.

## Setup

```bash
cd examples/browser-agent
uv sync
uv run playwright install chromium
export OPENAI_API_KEY=...
```

## Run

```bash
uv run python agent.py \
  https://en.wikipedia.org/wiki/Main_Page \
  "Open the article about the Rosetta Stone and tell me its first sentence."
```

The browser is visible by default. Pass `--headless` for unattended runs or `--model provider:model-name` to use another model supported by LangChain.

## How it works

`browser_agent()` owns one Playwright session and guarantees cleanup. It passes six tools to `create_agent`: observe, navigate, click, type, select, and scroll. Every observation assigns short-lived integer references to visible controls; actions resolve only references from the latest browser snapshot.

Navigation is restricted to HTTP(S) URLs resolving to public addresses. The guard runs for model-requested URLs and page-initiated navigations, blocking private, loopback, link-local, reserved, multicast, and unspecified addresses. Keep `allow_private=False` for untrusted tasks.

This example intentionally uses standard model tool calling rather than `langchain-typesafe`, making the `create_agent` refactor easy to understand. For the source project's speculative classifier approach and reusable package, see [`ts-browser-agent`](https://github.com/ndrezn/ts-browser-agent).
