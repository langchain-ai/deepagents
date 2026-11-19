<!-- markdownlint-disable MD024 -->

# Changelog

## [0.4.0](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.12...langchain-quickjs==0.4.0) (2026-11-19)

langchain-quickjs 0.4.0 is out! Code mode gets a real filesystem, tools that actually update your agent's state, and a much lighter footprint.

### 📁 `fs` in code mode

Code can now read and write the agent's files directly, the same files its file tools see, through a Node-style `fs` global. Pass `fs=True` with `create_deep_agent` (or pass any backend yourself):

```python
agent = create_deep_agent(
    model,
    middleware=[CodeInterpreterMiddleware(ptc=True, fs=True)],
)
```

One eval can now do what used to take a dozen file-tool round trips:

```js
const files = await fs.readdir("/reports");
const totals = {};
for (const name of files.filter((f) => f.endsWith(".json"))) {
  const { region, revenue } = JSON.parse(await fs.readFile(`/reports/${name}`));
  totals[region] = (totals[region] ?? 0) + revenue;
}
await fs.writeFile("/summary.json", JSON.stringify(totals, null, 2));
```

It has `readFile`, `writeFile`, `readdir`, `access` and `rm`, all async, and errors come back as Node errors (`err.code === "ENOENT"`).

### 🔁 Tools called from code now respect `Command`

Before, when a tool returned `Command(update=...)` and you called it from code, the update was silently dropped. Now it lands in graph state just like a normal tool call, later calls in the same eval see it, and your JS gets the tool's message back:

```python
@tool
def save_plan(plan: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Save the plan to state."""
    return Command(update={
        "plan": plan,
        "messages": [ToolMessage("plan saved", tool_call_id=tool_call_id)],
    })
```

```js
await tools.savePlan({ plan: "1. research 2. draft" }); // "plan saved", and state.plan is set
```

### ⚡ Faster startup, lower overhead

Interpreters are now created on the first eval, not when the agent starts, so threads that never run code never pay for one. Closing them no longer blocks the event loop, and runs that error or get cancelled release their interpreter cleanly. The code-mode system prompt is cached by content, so it isn't re-rendered every turn.

Also in this release: one `tools.*` proxy (with `defer_loading` tools still callable), typed signatures for MCP tools, and quickjs-rs 0.3.0.

> **Heads up:** JS state saved by 0.3.x won't restore, so existing threads start from a fresh JS context on their next eval.

### ⚠ BREAKING CHANGES

* keep QuickJS slots warm in a process-wide LRU ([#7790](https://github.com/langchain-ai/deepagents/issues/7790))
* upgrade to quickjs-rs 0.3.0

### Features

* add `writeFile`, `readdir`, `access`, and `rm` to `fs` ([#7638](https://github.com/langchain-ai/deepagents/issues/7638)) ([0cef0a8](https://github.com/langchain-ai/deepagents/commit/0cef0a8e3ec34de25e826305dce076b86cedc8bb))
* add an opt-in `fs.readFile` over the agent's backend ([#7637](https://github.com/langchain-ai/deepagents/issues/7637)) ([e4c8748](https://github.com/langchain-ai/deepagents/commit/e4c8748d4cc5fd86c5569d82464927912524bd67))
* let `fs=True` use the agent's backend ([#7669](https://github.com/langchain-ai/deepagents/issues/7669)) ([94657a4](https://github.com/langchain-ai/deepagents/commit/94657a47f8d47acfc17c19157f823b7bdd902994))
* upgrade to quickjs-rs 0.3.0 ([d60e40e](https://github.com/langchain-ai/deepagents/commit/d60e40e30731cdc3bd6a1e1ee2412316d97b290e))


### Bug Fixes

* apply state updates from `tools.*` calls in eval ([#7838](https://github.com/langchain-ai/deepagents/issues/7838)) ([c6bde12](https://github.com/langchain-ai/deepagents/commit/c6bde125f5974a12f64d56e0fb1058100ed7d59b))
* close slots left open by failed runs and idle slots ([#7463](https://github.com/langchain-ai/deepagents/issues/7463)) ([1a7d602](https://github.com/langchain-ai/deepagents/commit/1a7d602a7ef6b54269bbb268e4bbc71f27ada0f9))
* don't evict a slot while a cancelled eval still runs ([#7850](https://github.com/langchain-ai/deepagents/issues/7850)) ([533e4b3](https://github.com/langchain-ai/deepagents/commit/533e4b3046dfa5e915c5eed86e8a7d3b01319446))
* keep QuickJS slots warm in a process-wide LRU ([#7790](https://github.com/langchain-ai/deepagents/issues/7790)) ([beeb0a0](https://github.com/langchain-ai/deepagents/commit/beeb0a032e65a86910a280a1037bbf3f8e24fe26)), refs [#7262](https://github.com/langchain-ai/deepagents/issues/7262) [#5387](https://github.com/langchain-ai/deepagents/issues/5387) [#4546](https://github.com/langchain-ai/deepagents/issues/4546) [#6311](https://github.com/langchain-ai/deepagents/issues/6311)
* parent PTC calls and `task()` subagents on the eval run ([#7665](https://github.com/langchain-ai/deepagents/issues/7665)) ([f2a6e02](https://github.com/langchain-ai/deepagents/commit/f2a6e02dc8e2dcb05885c5133b94102ccc79b794))
* render dict (MCP) `args_schema` inline in PTC signatures ([#7464](https://github.com/langchain-ai/deepagents/issues/7464)) ([41841fc](https://github.com/langchain-ai/deepagents/commit/41841fc2e2424e5b78bc2875718fb1feb6e2dbc4))
* steer code mode to `Promise.allSettled` for parallel calls ([#7871](https://github.com/langchain-ai/deepagents/issues/7871)) ([40cdd33](https://github.com/langchain-ai/deepagents/commit/40cdd334d831e5c6737708811cc0335ec6105ac6))

## [0.3.12](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.11...langchain-quickjs==0.3.12) (2026-11-07)

### Features

- Added support for disabling the per-call `timeout` ([#6958](https://github.com/langchain-ai/deepagents/pull/6958)).

### Bug Fixes

- Fixed interpreter shutdown blocking in `CodeInterpreterMiddleware.__del__` ([#7460](https://github.com/langchain-ai/deepagents/pull/7460)).
- Fixed resolution of nested schema `$ref`s in PTC signatures ([#7441](https://github.com/langchain-ai/deepagents/pull/7441)).

## [0.3.11](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.10...langchain-quickjs==0.3.11) (2026-10-27)

### Bug Fixes

- Defer QuickJS runtime initialization until evaluation ([#7301](https://github.com/langchain-ai/deepagents/pull/7301)).

## [0.3.10](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.9...langchain-quickjs==0.3.10) (2026-10-26)

### Bug Fixes

- Fixed PTC prompt caching to cache prompts per slot and tool identity. ([#7296](https://github.com/langchain-ai/deepagents/pull/7296))

## [0.3.9](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.8...langchain-quickjs==0.3.9) (2026-10-13)

### Features

- Configure the number of concurrent subagents with `CodeInterpreterMiddleware(max_subagent_concurrency=16)`. The default remains 32 per REPL; requests above the cap queue until a slot opens. Lower limits can reduce memory use, load, and cost, while higher limits allow more parallel work. The prompt now recommends batches of around 10 rather than requiring them, even at the default setting. ([#6926](https://github.com/langchain-ai/deepagents/pull/6926))

## [0.3.8](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.7...langchain-quickjs==0.3.8) (2026-09-29)

### Bug Fixes

- Preserve subagent identity when replaying interrupts. ([#6370](https://github.com/langchain-ai/deepagents/pull/6370))

## [0.3.7](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.6...langchain-quickjs==0.3.7) (2026-09-06)

### Features

- Stream PTC tool calls natively. ([#5735](https://github.com/langchain-ai/deepagents/issues/5735))

### Bug Fixes

- Keep private state out of subagent propagation. ([#3543](https://github.com/langchain-ai/deepagents/issues/3543))

## [0.3.6](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.5...langchain-quickjs==0.3.6) (2026-09-03)

### Features

- Added an HMAC check for QuickJS snapshot bytes ([#5601](https://github.com/langchain-ai/deepagents/issues/5601)) ([749b386](https://github.com/langchain-ai/deepagents/commit/749b386f97ed7b920acfd5936b01f8c425976d92)).

### Performance Improvements

- Omitted middleware trace inputs to improve QuickJS performance ([#6015](https://github.com/langchain-ai/deepagents/issues/6015)) ([673844d](https://github.com/langchain-ai/deepagents/commit/673844d06fe0ee186f2e492c3c0a19ec5facffc2)).

## [0.3.5](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.4...langchain-quickjs==0.3.5) (2026-07-29)

### Bug Fixes

* Require deepagents 0.7.x ([#5151](https://github.com/langchain-ai/deepagents/issues/5151)) ([4cb7e27](https://github.com/langchain-ai/deepagents/commit/4cb7e27abbca6f1bb1d75861b6a1a76e7011ae31))

## [0.3.4](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.3...langchain-quickjs==0.3.4) (2026-07-24)

### Bug Fixes

* Raise minimum quickjs-rs version to 0.2.5 ([#5059](https://github.com/langchain-ai/deepagents/issues/5059)) ([794513b](https://github.com/langchain-ai/deepagents/commit/794513bbc58a859bbd858326e61565bd3f50e694))

## [0.3.3](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.2...langchain-quickjs==0.3.3) (2026-07-16)

### Bug Fixes

* Propagate JS `task()` subagent interrupts ([#4401](https://github.com/langchain-ai/deepagents/issues/4401)) ([0b30e49](https://github.com/langchain-ai/deepagents/commit/0b30e494d59187de2e6e23db062abbf835a6d265))
* Correct `eval` await description ([#4371](https://github.com/langchain-ai/deepagents/issues/4371)) ([4ab860f](https://github.com/langchain-ai/deepagents/commit/4ab860f007832821bd303809d7c2c1d2c86a6528))

## [0.3.2](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.1...langchain-quickjs==0.3.2) (2026-06-25)

### Features

* Dynamic subagents UI ([#4221](https://github.com/langchain-ai/deepagents/issues/4221)) ([10bcba2](https://github.com/langchain-ai/deepagents/commit/10bcba25600e51aba135f170b34aa6315c0f53d6))

### Bug Fixes

* Ensure top-level title on subagent response schemas ([#4155](https://github.com/langchain-ai/deepagents/issues/4155)) ([08f917e](https://github.com/langchain-ai/deepagents/commit/08f917eea79513e7e894731f884e738c2d30383f))
* Normalize nested `undefined` tool args ([#3935](https://github.com/langchain-ai/deepagents/issues/3935)) ([1b461a0](https://github.com/langchain-ai/deepagents/commit/1b461a0d6c6a7a6d323db18505596fcce4326b92))

## [0.3.1](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.3.0...langchain-quickjs==0.3.1) (2026-06-22)

### Bug Fixes

* Persist top-level JS declarations across evals ([#4147](https://github.com/langchain-ai/deepagents/issues/4147)) ([7574fea](https://github.com/langchain-ai/deepagents/commit/7574fea88b51c77f7afa8279c4f561d6ecb47e3f))
* PTC tools in tools namespace are rendered without prepended `tools.` in system prompt and task as ptc duplicated task global ([#4075](https://github.com/langchain-ai/deepagents/issues/4075)) ([014a903](https://github.com/langchain-ai/deepagents/commit/014a9033af70b5b8b08ad2eb36f98590f5f1cca8))

## [0.3.0](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.2.0...langchain-quickjs==0.3.0) (2026-06-18)

### ⚠ BREAKING CHANGES

* Upgrade to 0.2.0 quickjs-rs ([#4067](https://github.com/langchain-ai/deepagents/issues/4067))

### Features

* Prompt tuning on task global ([#4066](https://github.com/langchain-ai/deepagents/issues/4066)) ([a47696f](https://github.com/langchain-ai/deepagents/commit/a47696f6d3e57eccb5ea19fb344305a7995ecc76))
* Upgrade to 0.2.0 quickjs-rs ([#4067](https://github.com/langchain-ai/deepagents/issues/4067)) ([4ffea88](https://github.com/langchain-ai/deepagents/commit/4ffea88690418207b5e4fa800ee8c1abfa454bec))

## [0.2.0](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.4...langchain-quickjs==0.2.0) (2026-06-12)

### ⚠ BREAKING CHANGES

* Add default `subagent` bridge ([#3850](https://github.com/langchain-ai/deepagents/issues/3850))
* Remove `skills_backend` ([#3843](https://github.com/langchain-ai/deepagents/issues/3843))

### Features

* Add default `subagent` bridge ([#3850](https://github.com/langchain-ai/deepagents/issues/3850)) ([85fd7c2](https://github.com/langchain-ai/deepagents/commit/85fd7c283da6744e403a01861e17e99e13e0f481))

### Bug Fixes

* Remove `skills_backend` ([#3843](https://github.com/langchain-ai/deepagents/issues/3843)) ([1159e50](https://github.com/langchain-ai/deepagents/commit/1159e504abaeec4f81d5e777ecde6a6cee641edb))

## [0.1.4](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.3...langchain-quickjs==0.1.4) (2026-06-03)

### Bug Fixes

* Swarm subagent doesn't allow configuring middleware ([#3757](https://github.com/langchain-ai/deepagents/issues/3757)) ([3394a9d](https://github.com/langchain-ai/deepagents/commit/3394a9d9c7c89c0a28fa1328c9f6bae68a83ff14))

## [0.1.3](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.2...langchain-quickjs==0.1.3) (2026-06-01)

### Features

* Add REPL persistence modes ([#3557](https://github.com/langchain-ai/deepagents/issues/3557)) ([0cda6f3](https://github.com/langchain-ai/deepagents/commit/0cda6f3ab28bc83cd16ec9fcc48229bdf6f2dc1a))
* Add swarm task tool ([#3472](https://github.com/langchain-ai/deepagents/issues/3472)) ([2c28b7b](https://github.com/langchain-ai/deepagents/commit/2c28b7b8c2ac7571fc3a1f0d8d00f5697fe3e90e))

### Bug Fixes

* Auto-await final-expression Promise in eval REPL ([#3499](https://github.com/langchain-ai/deepagents/issues/3499)) ([f7f894a](https://github.com/langchain-ai/deepagents/commit/f7f894aa9f313cf8157bc6d7711013f5509d0b46))
* Scope REPL prompt sandbox bullet to the runtime ([#3528](https://github.com/langchain-ai/deepagents/issues/3528)) ([1b395ab](https://github.com/langchain-ai/deepagents/commit/1b395ab9699b1f384a85efeeef732ea7e4fc523a))
* Update system prompt snapshots ([#3450](https://github.com/langchain-ai/deepagents/issues/3450)) ([9f9220d](https://github.com/langchain-ai/deepagents/commit/9f9220d80737208faa9262c0bdfb3eeafc0e13c8))
* Stable `HumanMessage` IDs across resumed threads ([#3591](https://github.com/langchain-ai/deepagents/issues/3591)) ([82c3194](https://github.com/langchain-ai/deepagents/commit/82c31947f9dc938ffc71e1cea96d162a39aec3a1))

## [0.1.2](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.1...langchain-quickjs==0.1.2) (2026-05-11)

### Features

* Rename middleware ([#3334](https://github.com/langchain-ai/deepagents/issues/3334)) ([fc80075](https://github.com/langchain-ai/deepagents/commit/fc80075c65c3b4beb8f672b6bb27464fee6d79c2))

## [0.1.1](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.0...langchain-quickjs==0.1.1) (2026-05-08)

### Features

* Propagate return types ([#3210](https://github.com/langchain-ai/deepagents/issues/3210)) ([e26bccb](https://github.com/langchain-ai/deepagents/commit/e26bccbe81b4e3ff2f0332f56f683106e0bafd88))

## [0.1.0](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.0.1...langchain-quickjs==0.1.0) (2026-05-05)

This release introduces a new QuickJS runtime implementation backed by `quickjs-rs`.

---

## Prior Releases

Versions prior to 0.0.2 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=langchain-quickjs) for details on previous versions.
