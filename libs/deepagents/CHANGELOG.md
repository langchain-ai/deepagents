# Changelog

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
