# DeepAgent inside a LangSmith sandbox

This example runs a DeepAgent **inside a LangSmith sandbox**: the agent process
and its LLM calls execute on the box, and it edits files and runs commands
against the box's own filesystem. A host-side driver boots the sandbox, hands it
the agent and a small coding task, runs the agent, and verifies the result with
`pytest`, all inside the box.

The model API key is never placed in the box. Instead, the sandbox
[auth proxy](https://docs.langchain.com/langsmith/sandbox-auth-proxy) injects it
into outbound requests to the provider, and the box runs with a dummy key.

## How it works

```txt
host (run_in_sandbox.py)                 LangSmith sandbox
────────────────────────                 ─────────────────────────────
boot box + auth-proxy rule  ──────────▶  box boots (deepagents baked in)
upload agent.py + task      ──────────▶  /app/agent.py, /app/solution.py, ...
run agent in box            ──────────▶  python agent.py
                                           └─ create_deep_agent + LocalShellBackend
                                           └─ edits /app/solution.py, runs pytest
                                           └─ LLM call ─▶ proxy injects real key ─▶ Anthropic
verify with pytest in box   ──────────▶  python -m pytest
delete box
```

Two details make the in-box LLM call work:

- **Auth proxy injects the key.** The driver registers an `opaque` proxy rule
  for `api.anthropic.com` carrying the real `ANTHROPIC_API_KEY`. The box runs
  with `ANTHROPIC_API_KEY=foo`; the proxy replaces the header on egress.
- **`truststore` trusts the proxy CA.** The proxy terminates TLS with its own
  CA, which the sandbox installs into the box's OS trust store at boot.
  `agent.py` calls `truststore.inject_into_ssl()` so httpx (and the Anthropic
  SDK) verify against the OS store rather than the bundled `certifi` roots.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `run_in_sandbox.py` | host | Boots the box, uploads files, runs the agent, verifies, cleans up |
| `agent.py` | box | The DeepAgent (`create_deep_agent` + `LocalShellBackend`) — swap in your own |
| `Dockerfile` | box image | Bakes `deepagents`, `langchain-anthropic`, `truststore`, `pytest` into the snapshot |
| `task/` | box | The coding task: a `roman_to_int` stub plus its `pytest` suite |

## Run it

```bash
cp .env.example .env   # fill in LANGSMITH_PROFILE (or LANGSMITH_API_KEY) and ANTHROPIC_API_KEY
uv run --with-requirements <(uv export) python run_in_sandbox.py
# or: uv sync && uv run python run_in_sandbox.py
```

Expected tail:

```txt
[driver] running the DeepAgent inside the box...
[agent] final message: ...
[driver] verifying with pytest inside the box...
6 passed
==== RESULT: PASS ✅ ====
```

The first run builds the snapshot (a minute or two); later runs reuse it.

## Make it your own

- **Swap the agent:** replace `agent.py` with any `deepagents` agent. Keep
  `LocalShellBackend(root_dir="/app")` so it acts on the box filesystem.
- **Swap the task:** drop your own files and tests into `task/`.
- **Other providers:** add a proxy rule for the provider's host in
  `run_in_sandbox.py` and point the agent's model at it.
