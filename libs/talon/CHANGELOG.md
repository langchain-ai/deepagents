# Changelog

## [0.0.7](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.6...deepagents-talon==0.0.7) (2026-09-07)

### Highlights

- Added Discord channel support, including improved gateway failure reporting after startup. ([#5992](https://github.com/langchain-ai/deepagents/issues/5992), [#6113](https://github.com/langchain-ai/deepagents/issues/6113))
- Added chat-scoped conversation history with configurable storage and optional hybrid search. ([#6105](https://github.com/langchain-ai/deepagents/issues/6105), [#6115](https://github.com/langchain-ai/deepagents/issues/6115), [#6108](https://github.com/langchain-ai/deepagents/issues/6108))
- Added MCP configuration tools, channel-based MCP server authorization, hot reload for MCP configuration, and OAuth/device authentication support for Slack and GitHub MCP integrations. ([#6097](https://github.com/langchain-ai/deepagents/issues/6097), [#6073](https://github.com/langchain-ai/deepagents/issues/6073), [#6084](https://github.com/langchain-ai/deepagents/issues/6084), [#6078](https://github.com/langchain-ai/deepagents/issues/6078), [#6079](https://github.com/langchain-ai/deepagents/issues/6079))
- Added more flexible subagent execution, including background expendable subagents, dcode-style subagents in fork mode, fresh subagents, per-task tool selection, and on-demand subagent configuration reloads. ([#6098](https://github.com/langchain-ai/deepagents/issues/6098), [#6085](https://github.com/langchain-ai/deepagents/issues/6085), [#6129](https://github.com/langchain-ai/deepagents/issues/6129), [#6099](https://github.com/langchain-ai/deepagents/issues/6099))
- Added defensive research defaults and a configuration-hardening self-review skill, with missing research subagent defaults backfilled. ([#6131](https://github.com/langchain-ai/deepagents/issues/6131), [#6136](https://github.com/langchain-ai/deepagents/issues/6136), [#6135](https://github.com/langchain-ai/deepagents/issues/6135))

### New features

- Added a `/help` command. ([#6106](https://github.com/langchain-ai/deepagents/issues/6106))
- Added a `current_time` tool and timezone-aware wall-clock cron schedules. ([#6065](https://github.com/langchain-ai/deepagents/issues/6065), [#6062](https://github.com/langchain-ai/deepagents/issues/6062))
- Added persistent LangGraph checkpoints. ([#6088](https://github.com/langchain-ai/deepagents/issues/6088))
- Added channel debug logging and opt-in agent activity logging. ([#5983](https://github.com/langchain-ai/deepagents/issues/5983), [#5984](https://github.com/langchain-ai/deepagents/issues/5984))
- New messages can now interrupt active turns. ([#6023](https://github.com/langchain-ai/deepagents/issues/6023))
- Long agent turns now keep the typing indicator alive. ([#5993](https://github.com/langchain-ai/deepagents/issues/5993))

### Fixes and improvements

- Improved channel reconnect resilience. ([#6040](https://github.com/langchain-ai/deepagents/issues/6040))
- Improved cron reliability by keeping the ticker alive after failed ticks, and stored cron jobs in a structured, versioned format. ([#6087](https://github.com/langchain-ai/deepagents/issues/6087), [#6086](https://github.com/langchain-ai/deepagents/issues/6086))
- Improved conversation history embeddings and search by correcting token budgets and prompt defaults, making remote embedding settings explicit, keeping search non-blocking, and removing the default Qwen query prefix. ([#6132](https://github.com/langchain-ai/deepagents/issues/6132), [#6133](https://github.com/langchain-ai/deepagents/issues/6133), [#6134](https://github.com/langchain-ai/deepagents/issues/6134))
- Fixed OAuth and MCP integration issues, including TLS hostname normalization, omitted empty optional MCP arguments, persisted OAuth token expiry, secured OAuth discovery, and restarted token refresh. ([#6102](https://github.com/langchain-ai/deepagents/issues/6102), [#6077](https://github.com/langchain-ai/deepagents/issues/6077), [#6090](https://github.com/langchain-ai/deepagents/issues/6090), [#6100](https://github.com/langchain-ai/deepagents/issues/6100))
- Fixed WhatsApp behavior by restoring bridge compatibility, preserving quoted message context and approval loops, handling reactions, and restricting replies to self-chat. ([#5999](https://github.com/langchain-ai/deepagents/issues/5999), [#6025](https://github.com/langchain-ai/deepagents/issues/6025), [#6104](https://github.com/langchain-ai/deepagents/issues/6104), [#6010](https://github.com/langchain-ai/deepagents/issues/6010))
- Treated trailing `[SILENT]` as a suppression marker. ([#6110](https://github.com/langchain-ai/deepagents/issues/6110))

## [0.0.6](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.5...deepagents-talon==0.0.6) (2026-08-28)

### Bug Fixes

- Removed `extract-zip` from the WhatsApp bridge dependency tree. ([#5924](https://github.com/langchain-ai/deepagents/issues/5924))

## [0.0.5](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.4...deepagents-talon==0.0.5) (2026-08-26)

### Bug Fixes

- Migrated MCP discovery to `discover_mcp_config_sources`. ([#5803](https://github.com/langchain-ai/deepagents/issues/5803))

## [0.0.4](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.3...deepagents-talon==0.0.4) (2026-08-24)

### Features

- Require Python 3.12 or greater. ([#5603](https://github.com/langchain-ai/deepagents/issues/5603))

## [0.0.3](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.2...deepagents-talon==0.0.3) (2026-07-06)


### Features

* **sdk:** optional video frame extraction on `read_file` ([#4094](https://github.com/langchain-ai/deepagents/issues/4094)) ([b927147](https://github.com/langchain-ai/deepagents/commit/b927147d026749c6c790bb06c9853515dabf579c))
* **talon:** add Fleet zip import command ([#4493](https://github.com/langchain-ai/deepagents/issues/4493)) ([0289dd0](https://github.com/langchain-ai/deepagents/commit/0289dd0a190e5060e631e840da115dd59c64cf5c))


### Bug Fixes

* **talon:** materialize agents under home ([f2b26a8](https://github.com/langchain-ai/deepagents/commit/f2b26a8915fb70c26d32af6e8240442e5e6118e6))

## [0.0.2](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.1...deepagents-talon==0.0.2) (2026-06-30)


### Features

* **talon:** `DEEPAGENTS_TALON_RECURSION_LIMIT` env var ([#4354](https://github.com/langchain-ai/deepagents/issues/4354)) ([82d1eac](https://github.com/langchain-ai/deepagents/commit/82d1eac59a43f096096e86849733aa716adb18fc))
* **talon:** add reaction approval routing ([#4345](https://github.com/langchain-ai/deepagents/issues/4345)) ([3fe8c0c](https://github.com/langchain-ai/deepagents/commit/3fe8c0c35536626f583df08573469506b9529706))
* **talon:** add Telegram channel adapter, CLI wiring, and offset persistence ([#4097](https://github.com/langchain-ai/deepagents/issues/4097)) ([7c87cec](https://github.com/langchain-ai/deepagents/commit/7c87ceca069874db8555705efab3973301baa1cb))
* **talon:** add tool approval env override ([#4349](https://github.com/langchain-ai/deepagents/issues/4349)) ([d26481d](https://github.com/langchain-ai/deepagents/commit/d26481da615881bae4401dfa485ad925945e667a))
* **talon:** audit reaction approval attempts ([#4348](https://github.com/langchain-ai/deepagents/issues/4348)) ([d7895c4](https://github.com/langchain-ai/deepagents/commit/d7895c4f9b996ad6fe194936bbeaa8beea21e913))
* **talon:** ingest Telegram approval reactions ([#4346](https://github.com/langchain-ai/deepagents/issues/4346)) ([437af0b](https://github.com/langchain-ai/deepagents/commit/437af0bf79332b20ae0c1883c3cc4d91a98c2457))


### Bug Fixes

* **talon:** default workspace to current directory ([#4099](https://github.com/langchain-ai/deepagents/issues/4099)) ([5e337ae](https://github.com/langchain-ai/deepagents/commit/5e337ae50a76bc174b752be187e62698a389cbe6))

## Changelog

All notable changes to this project will be documented in this file.
