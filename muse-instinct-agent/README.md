# Agentic shopper (Managed Deep Agent)

A [Managed Deep Agent](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview)
that browses the web with **Stagehand** (Browserbase) and checks out with the
**Stripe Link CLI**, using secure one-time-use payment credentials instead of a
stored real card. LangSmith runs the harness and hosted runtime; this project is
just the agent's business logic.

## Layout

```
muse-instinct-agent/
├── agent.py            # exports `agent` via define_deep_agent (model + tools)
├── instructions.md     # system prompt (synced to LangSmith Context Hub)
├── identity.py         # caller identity — LangSmith API key
├── tools/
│   └── browser.py      # Stagehand navigate/act/extract/observe/screenshot
├── sandbox/
│   ├── __init__.py     # define_sandbox
│   └── setup.sh        # installs Node + @stripe/link-cli into the snapshot
├── pyproject.toml
└── .env.example        # copy to .env and fill in
```

## Why the Link CLI runs in the sandbox (not over MCP)

MDA connects to MCP servers over **remote HTTP**, and the Link CLI's `--mcp`
mode is **stdio** while `link-cli serve` binds **loopback** (`127.0.0.1`, per
Stripe's docs) — neither is reachable from the hosted runtime, and running an
MCP server does **not** change how auth is piped. The MDA **sandbox**, however,
is co-located with the agent: `setup.sh` installs `link-cli` into the thread's
image, and the agent drives it through the built-in `execute` tool. No tunnel,
no reachability problem.

## Auth: interactive vs. headless

- **Interactive (default).** `link-cli auth login` is a device-approval flow;
  the CLI stores its session on the per-thread sandbox filesystem. Works in
  `mda dev` and in the cloud when a human approves in the Link app.
- **Headless (cloud).** Supply a Link OAuth access token as an MDA **managed
  connection**, not a plain env var. Managed connections live in Agent Auth,
  resolve per-run, and are never cached to disk or baked into the shared
  snapshot:

  ```bash
  mda connections create link-access-token --secret-from-env LINK_ACCESS_TOKEN
  ```

  Resolve it inside a tool with `connections.get("link-access-token",
  {"type": "agent"})` and pass it as `LINK_ACCESS_TOKEN` in the sandbox
  `execute` call's env (the CLI reads `LINK_ACCESS_TOKEN`/`LINK_REFRESH_TOKEN`
  directly). For `mda dev`, the same slug resolves from
  `MDA_DEV_LINK_ACCESS_TOKEN` in `.env`. Wiring the resolver as an authored
  tool is a documented follow-up.

## Payments: what is and isn't possible

- **Test mode** (`--test`) returns a **test card** and never charges a real
  payment method. `instructions.md` requires `--test` on every spend request.
- **Auth still needs a logged-in Link account.** `link-cli auth login` is a
  device-approval flow in the Link app; the agent **cannot self-approve** a
  spend. For a real (test-mode) transaction, a human approves in the app.
- **Wallet limits are set in the Link app, not the CLI.** To cap spend (e.g.
  **$20/day**), set the limit in the Link app; the agent *reads*
  `agent_wallet_spend_limits` via `user-info retrieve` and refuses purchases
  over it, but it cannot change limits. Stripe's platform hard caps apply on
  top: $500 per request, $500/day, $20,000/30-day.
- **US-only** (Link account requirement).

## Setup

```bash
cp .env.example .env    # fill LANGSMITH_API_KEY, ANTHROPIC_API_KEY, BROWSERBASE_API_KEY
uv sync
```

## Run locally, then deploy

```bash
uv run mda dev        # local Agent Server + LangSmith Studio; sandbox falls back
                      # to a local temp dir, so you can `link-cli auth login`
                      # interactively and approve a test spend on your phone.
uv run mda deploy     # hosted deployment on LangSmith Agent Server
uv run mda logs       # tail deployment logs
```

### Test matrix (what works where)

| Capability     | `mda dev` (local)                          | `mda deploy` (cloud)                        |
| -------------- | ------------------------------------------ | ------------------------------------------- |
| Browser-use    | ✅ (Browserbase reaches out either way)     | ✅                                           |
| Link install   | ✅ (or local temp-dir sandbox)              | ✅ (baked by `setup.sh`)                     |
| Link auth      | ✅ interactive `auth login` + app approval  | app approval, or headless via the `link-access-token` connection |
| Test spend     | ✅ end-to-end with your approval            | ✅ once auth is present + you approve        |

Try in Studio:

```
Find "Working in Public" on press.stripe.com, take it to checkout,
and pay for it in test mode.
```

## Follow-ups

- **Headless Link auth** is wired through an MDA managed connection
  (`link-access-token`); resolving it inside an authored `link_exec` tool that
  injects `LINK_ACCESS_TOKEN` into the sandbox `execute` env is the next step.
- **Per-user payments** need per-caller Link wallets. That requires Supabase
  identity (see `identity.py`) plus a frontend, and per-user grants — the same
  `connections.get(..., {"type": "user"})` path, one grant per caller. Documented,
  not wired.
- **MPP merchants** (HTTP 402, `method="stripe"`) can be paid without a browser
  via `link-cli mpp decode` + `mpp pay --credential-type shared_payment_token`.
