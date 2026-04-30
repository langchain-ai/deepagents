# Changelog

## [0.5.5](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.4...deepagents==0.5.5) (2026-04-30)


### Bug Fixes

* **cli:** prevent stdin hang by passing `DEVNULL` ([#2427](https://github.com/langchain-ai/deepagents/issues/2427)) ([5bf5fae](https://github.com/langchain-ai/deepagents/commit/5bf5fae8d93beba90628f2f71e3e79817a36ac9e))
* **sdk:** harden `FilesystemBackend` against symlink loops ([#3035](https://github.com/langchain-ai/deepagents/issues/3035)) ([abd02f9](https://github.com/langchain-ai/deepagents/commit/abd02f99ef12030bdfe429fdc3ad80a2785bea61))
* **sdk:** re-export filesystem permission for backwards compatibility ([#3036](https://github.com/langchain-ai/deepagents/issues/3036)) ([e04b50a](https://github.com/langchain-ai/deepagents/commit/e04b50ae291abefa64ee2750a0c1bbfd93954b32))
* **sdk:** surface EOF-newline mismatch in `edit_file` ([#3031](https://github.com/langchain-ai/deepagents/issues/3031)) ([d30686e](https://github.com/langchain-ai/deepagents/commit/d30686ec82d36a0e9430f7c512c34835aba2c079))

## [0.5.4](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.3...deepagents==0.5.4) (2026-04-29)

### Highlights

Until this release, `deepagents` shipped a single set of prompts, tools, and middleware tuned to work across *all* model families. Today's headline change is **harness profiles** — a declarative override layer for the parts of the harness that vary per model (system-prompt prefix/suffix, tool inclusion and naming, middleware selection, subagent configuration, skills). `ProviderProfile` shapes how `resolve_model` builds the client; `HarnessProfile` / `HarnessProfileConfig` shape how `create_deep_agent` assembles the agent. Both are keyed by provider (`"openai"`) or full `provider:model` spec (`"openai:gpt-5.4"`), and registrations are additive — your call site doesn't change when you swap models.

We currently ship built-ins for OpenAI and Anthropic out of the box, drawn directly from each vendor's published prompting guides (Codex's `apply_patch` / `shell_command` conventions, Claude's tool-result reflection and active-investigation patterns, etc.). On a curated tau2-bench subset, applying these profiles produced a 10–20 point lift over the default harness (GPT-5.3 Codex: 33% -> 53%; Claude Opus 4.7: 43% -> 53%). Third-party plugins can register via the `deepagents.provider_profiles` and `deepagents.harness_profiles` entry points.

See the [profiles documentation](https://docs.langchain.com/oss/python/deepagents/profiles) for the full field surface, merge semantics, and plugin packaging.

### Features

* Profiles API ([#2892](https://github.com/langchain-ai/deepagents/issues/2892)) ([7365ad1](https://github.com/langchain-ai/deepagents/commit/7365ad1600064eec616c5de970320104189ddf80))
* `ls_agent_type` configurable tag on subagent runs ([#2788](https://github.com/langchain-ai/deepagents/issues/2788)) ([3bcc51a](https://github.com/langchain-ai/deepagents/commit/3bcc51a95da80094cfc8bc4bcaf25dc1e2ad8f44))

### Bug Fixes

* Normalize Windows backslash paths before `PurePosixPath` processing ([#1859](https://github.com/langchain-ai/deepagents/issues/1859)) ([e1c1d50](https://github.com/langchain-ai/deepagents/commit/e1c1d5024729f5205eaa42bf6a9bc1c93a30d043))
* Preserve CRLF line endings in sandbox edit ([#2899](https://github.com/langchain-ai/deepagents/issues/2899)) ([291aebe](https://github.com/langchain-ai/deepagents/commit/291aebe21f8a53604a2bf47daa120761dace2536))
* Support read-your-writes in `StateBackend` ([#2991](https://github.com/langchain-ai/deepagents/issues/2991)) ([0924869](https://github.com/langchain-ai/deepagents/commit/0924869bc3d946577e7c3cbc79a86e4aaf522edd))
* Treat boundary-truncated UTF-8 in `read()` prefix check as text ([#2980](https://github.com/langchain-ai/deepagents/issues/2980)) ([c36ebc7](https://github.com/langchain-ai/deepagents/commit/c36ebc7be5840e9008279992741c67a8377ffc01))
* Use `configurable` directly instead of tracing context for subagent tagging ([#2845](https://github.com/langchain-ai/deepagents/issues/2845)) ([bd6ec6b](https://github.com/langchain-ai/deepagents/commit/bd6ec6bcebcdcc26f6b79e2c55611074b0e01631))

### Performance Improvements

* Add cache breakpoint to `MemoryMiddleware` ([#2713](https://github.com/langchain-ai/deepagents/issues/2713)) ([1699f3a](https://github.com/langchain-ai/deepagents/commit/1699f3aea710985087b16318bb8e6f6e80e02a1b))

## [0.5.3](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.2...deepagents==0.5.3) (2026-04-14)

### Features

* **sdk:** add static structured output to subagent response ([#2437](https://github.com/langchain-ai/deepagents/issues/2437)) ([6e57731](https://github.com/langchain-ai/deepagents/commit/6e57731fc6d908ac1ebe131e782696a4776147e9))
* **sdk:** deprecate `model=None` in `create_deep_agent` ([#2677](https://github.com/langchain-ai/deepagents/issues/2677)) ([149df41](https://github.com/langchain-ai/deepagents/commit/149df415d17f3cf3b7eb0bd1e78460112bfa9b04))

### Bug Fixes

* **sdk:** skill loading should default to 1000 lines ([#2721](https://github.com/langchain-ai/deepagents/issues/2721)) ([badc4d3](https://github.com/langchain-ai/deepagents/commit/badc4d3921ae0ede4305f44f85fa7266df9465e7))

## [0.5.2](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.1...deepagents==0.5.2) (2026-04-10)

### Features

* Permissions system for filesystem access control ([#2633](https://github.com/langchain-ai/deepagents/issues/2633)) ([41dc759](https://github.com/langchain-ai/deepagents/commit/41dc7597deb3fc036f1f850e68edc3c0870f27da))
  * Scope permissions to routes for composite backends with sandbox default ([#2659](https://github.com/langchain-ai/deepagents/issues/2659)) ([6dd6122](https://github.com/langchain-ai/deepagents/commit/6dd612237a7ee707726c4cafc4b691704e4cdb37))
  * Raise `ValueError` for permission paths without leading slash and path traversal ([#2665](https://github.com/langchain-ai/deepagents/issues/2665)) ([723d27d](https://github.com/langchain-ai/deepagents/commit/723d27dcdce03cc9ffaa757c70533f0134a43a44))
* Implement `upload_files` for `StateBackend` ([#2661](https://github.com/langchain-ai/deepagents/issues/2661)) ([5798345](https://github.com/langchain-ai/deepagents/commit/579834513a4ba1a024a52fc4edf918f526eab5f2))

### Bug Fixes

* Catch `PermissionError` in `FilesystemBackend` ripgrep ([#2571](https://github.com/langchain-ai/deepagents/issues/2571)) ([3d5d673](https://github.com/langchain-ai/deepagents/commit/3d5d67349c8e88e33af98137db9634742f018cb0))

## [0.5.1](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.0...deepagents==0.5.1) (2026-04-07)

### Features

* **sdk:** `BASE_AGENT_PROMPT` tweaks ([#2541](https://github.com/langchain-ai/deepagents/issues/2541)) ([812eef1](https://github.com/langchain-ai/deepagents/commit/812eef185ffda7bc9e6f11425eb5eddc3d3b32e8))
* **sdk:** add `artifacts_root` to `CompositeBackend` and middleware ([#2490](https://github.com/langchain-ai/deepagents/issues/2490)) ([753ee56](https://github.com/langchain-ai/deepagents/commit/753ee567f1cc4d544dc2afea7b414564fd07d37d))

### Bug Fixes

* **sdk:** updates for multimodal ([#2514](https://github.com/langchain-ai/deepagents/issues/2514)) ([a2edf3e](https://github.com/langchain-ai/deepagents/commit/a2edf3ed80e17a87027c41a46283387031ebd3e5))

---

# Prior Releases

Versions prior to 0.5.1 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=deepagents) for details on previous versions.
