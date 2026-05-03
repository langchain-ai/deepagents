# hitl-rejection-profile

Two patterns for shipping a tuned `rejection_response` for `HumanInTheLoopMiddleware`
(`langchain-ai/langchain` upstream, alpha) through Deep Agents profiles:

1. **`HarnessProfile.extra_middleware` ships a fully-configured HITL middleware**
   when the policy is uniform across an org for a given model.
2. **A standalone helper exports the tuned factory** when the *list of tools
   requiring HITL* is app-specific but the *rejection wording* should be
   uniform per provider.

The motivation is GitHub issue
[langchain-ai/deepagents#2947](https://github.com/langchain-ai/deepagents/issues/2947):
the upstream `HumanInTheLoopMiddleware` returns `ToolMessage(status="error")`
for a `reject` decision, which several models (gpt-4o, current Anthropic tiers)
treat as a transient tool failure and immediately re-emit the same call. The
alpha `rejection_response` knob lets callers swap in `status="success"` plus
retry-discouraging content on a per-tool basis. Profiles are a natural place
to ship that wiring once and let every agent in an org pick it up.

## Run

```bash
uv run python example.py
```

## Caveats

- `rejection_response` is alpha upstream — surface and semantics may shift.
- Pattern 1 doesn't compose with caller-supplied `HumanInTheLoopMiddleware`
  passed via `create_deep_agent(middleware=[...])` — `extra_middleware`
  appends, so two HITL middlewares would coexist. Pattern 2 sidesteps this.
