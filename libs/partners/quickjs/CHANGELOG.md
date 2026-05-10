# Changelog

## [0.1.1](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.0...langchain-quickjs==0.1.1) (2026-05-08)


### Features

* **quickjs:** propagate return types ([#3210](https://github.com/langchain-ai/deepagents/issues/3210)) ([e26bccb](https://github.com/langchain-ai/deepagents/commit/e26bccbe81b4e3ff2f0332f56f683106e0bafd88))


### Reverted Changes

* **quickjs:** release: 0.1.1 ([#3255](https://github.com/langchain-ai/deepagents/issues/3255)) ([8125f71](https://github.com/langchain-ai/deepagents/commit/8125f71a6ffd40b75a25c017e2b255eeb3be48a6))

## [0.1.0](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.0.1...langchain-quickjs==0.1.0) (2026-05-05)

This release introduces a new QuickJS runtime implementation backed by
`quickjs-rs`, replacing the previous interpreter path.

### Features

* **quickjs:** add `max_ptc_calls` budget for ptc calls ([#2994](https://github.com/langchain-ai/deepagents/issues/2994)) ([13f6c2d](https://github.com/langchain-ai/deepagents/commit/13f6c2dec448e270d7e685447292d394d31fe528))
* **quickjs:** add snapshot-based repl persistence between turns ([#3064](https://github.com/langchain-ai/deepagents/issues/3064)) ([c46feed](https://github.com/langchain-ai/deepagents/commit/c46feed66e83876054489a8454904de4c87ddf6b))
* **quickjs:** surface tool exceptions as the original error ([#3049](https://github.com/langchain-ai/deepagents/issues/3049)) ([d96dc8c](https://github.com/langchain-ai/deepagents/commit/d96dc8ce4a1d2d25abc606266cfe14629789c457))
* **quickjs:** use quickjs-rs ([#2979](https://github.com/langchain-ai/deepagents/issues/2979)) ([7491899](https://github.com/langchain-ai/deepagents/commit/7491899074fa1ed7946f2f5a0e84c2b3793d39bd))


### Bug Fixes

* **quickjs:** add `ls_code_input_language` metadata to `eval` tool ([#3062](https://github.com/langchain-ai/deepagents/issues/3062)) ([b9bc674](https://github.com/langchain-ai/deepagents/commit/b9bc674fe67ac40e000b30bbb4a753a9b6e167ed))
* **quickjs:** bound console buffering at capture time ([#2999](https://github.com/langchain-ai/deepagents/issues/2999)) ([251e405](https://github.com/langchain-ai/deepagents/commit/251e405d8b1897ae09ee47b0a8be48edfaaad8a1))
* **quickjs:** handle top-level await in snapshot step ([#3161](https://github.com/langchain-ai/deepagents/issues/3161)) ([b330c22](https://github.com/langchain-ai/deepagents/commit/b330c222ee661f7944a59d7b0ef2e0c7a42b1e29))
* **quickjs:** keep PTC loop and runtime context in `_PTCState` ([#3134](https://github.com/langchain-ai/deepagents/issues/3134)) ([c70cc5a](https://github.com/langchain-ai/deepagents/commit/c70cc5a602bd26191f347dd31cc4e5d17c75f65f))
* **quickjs:** per-thread_id Runtimes ([#2931](https://github.com/langchain-ai/deepagents/issues/2931)) ([4021b03](https://github.com/langchain-ai/deepagents/commit/4021b032f79913ddbfbb233fce5aff3f245fd1db))
* **quickjs:** prompt improvements ([#2564](https://github.com/langchain-ai/deepagents/issues/2564)) ([4999c6b](https://github.com/langchain-ai/deepagents/commit/4999c6b064266a0f0189964241323fd0f0ba345e))
* **quickjs:** remove ptc command buffering ([#3023](https://github.com/langchain-ai/deepagents/issues/3023)) ([ac1218a](https://github.com/langchain-ai/deepagents/commit/ac1218a1b14cde2f066c59804ab86567821f8c9d))

## Changelog

---

## Prior Releases

Versions prior to 0.0.2 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=langchain-quickjs) for details on previous versions.
