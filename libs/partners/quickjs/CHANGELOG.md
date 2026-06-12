<!-- markdownlint-disable MD024 -->

# Changelog

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
