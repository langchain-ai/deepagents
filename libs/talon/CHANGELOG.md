# Changelog

## [0.0.8](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.7...deepagents-talon==0.0.8) (2026-09-11)


### Features

* **talon:** add `send_message` for progress updates ([#6264](https://github.com/langchain-ai/deepagents/issues/6264)) ([15606d2](https://github.com/langchain-ai/deepagents/commit/15606d280a8ddb86c8966fef0bab257f8070944f))
* **talon:** add targeted conversation deletion ([#6235](https://github.com/langchain-ai/deepagents/issues/6235)) ([4fb301d](https://github.com/langchain-ai/deepagents/commit/4fb301d647c860d0d8f2108149e58cd9fe623352))
* **talon:** manage tool approvals through `tools.json` ([#6248](https://github.com/langchain-ai/deepagents/issues/6248)) ([a273ad6](https://github.com/langchain-ai/deepagents/commit/a273ad6c6a595c3aff915b744750711b50d544e3))


### Bug Fixes

* **talon:** bound archive scans, query embeddings, and deletion markers ([#6162](https://github.com/langchain-ai/deepagents/issues/6162)) ([79c23e6](https://github.com/langchain-ai/deepagents/commit/79c23e669bf81cddf16e4ab1582ccc9b26d8bea7))
* **talon:** bound the vector rebuild, break the import cycle, pin postgres ([#6165](https://github.com/langchain-ai/deepagents/issues/6165)) ([8462c24](https://github.com/langchain-ai/deepagents/commit/8462c24524541f49a3eb53037580110586973c8c))
* **talon:** close a url swap past the MCP auto-approve guard ([#6173](https://github.com/langchain-ai/deepagents/issues/6173)) ([d886742](https://github.com/langchain-ai/deepagents/commit/d8867426af3ab83c31d64a11e5b84a21ce86a43d))
* **talon:** deliver background subagents launched by a scheduled job ([#6228](https://github.com/langchain-ai/deepagents/issues/6228)) ([b6dc5c5](https://github.com/langchain-ai/deepagents/commit/b6dc5c5a8c6c1124f8e243868a83b9af188b08c7))
* **talon:** fix local voice transcription and embedding retries ([#6255](https://github.com/langchain-ai/deepagents/issues/6255)) ([a97ba32](https://github.com/langchain-ai/deepagents/commit/a97ba320fa797e9546efe3486ce703e8efd30fb4))
* **talon:** harden the MCP OAuth device flow and credential paths ([#6170](https://github.com/langchain-ai/deepagents/issues/6170)) ([e6c5829](https://github.com/langchain-ai/deepagents/commit/e6c58294d6792bcf433eb5f38fec1b5be70cc065))
* **talon:** keep results a discarded turn never reported ([#6231](https://github.com/langchain-ai/deepagents/issues/6231)) ([481773c](https://github.com/langchain-ai/deepagents/commit/481773caed336c86034e1c72b4988e7e93968610))
* **talon:** make background subagents and host start/stop recoverable ([#6166](https://github.com/langchain-ai/deepagents/issues/6166)) ([d4c41b0](https://github.com/langchain-ai/deepagents/commit/d4c41b048132eb27e2337b60b21463403b615390))
* **talon:** make MCP configuration updates bounded and non-destructive ([#6171](https://github.com/langchain-ai/deepagents/issues/6171)) ([f995931](https://github.com/langchain-ai/deepagents/commit/f9959311bfeb788191f971fe32552b01adbac199))
* **talon:** make subagent orchestration state what it enforces ([#6167](https://github.com/langchain-ai/deepagents/issues/6167)) ([05620cc](https://github.com/langchain-ai/deepagents/commit/05620cc72bfa7b81ce41064641cfa5808cb7c74a))
* **talon:** persist local model downloads ([#6137](https://github.com/langchain-ai/deepagents/issues/6137)) ([fc91199](https://github.com/langchain-ai/deepagents/commit/fc91199a44b99990cca49341169aea858da222fc))
* **talon:** report MCP protocol errors to the model ([#6239](https://github.com/langchain-ai/deepagents/issues/6239)) ([edd0bcf](https://github.com/langchain-ai/deepagents/commit/edd0bcfc61dc16eabea0b3a59aa84569e3b240dc))
* **talon:** share one optional-driver loader and close provider clients ([#6164](https://github.com/langchain-ai/deepagents/issues/6164)) ([02cea21](https://github.com/langchain-ai/deepagents/commit/02cea210d7f3e5bc2bf89453e8a6e4af59031f59))
* **talon:** stop losing MCP refresh tokens on a refresh response ([#6172](https://github.com/langchain-ai/deepagents/issues/6172)) ([7dc07cb](https://github.com/langchain-ai/deepagents/commit/7dc07cbb60f144e2b7fc266ee452b06c8b8dd10c))
* **talon:** stop one conversation from stalling or outliving the rest ([#6168](https://github.com/langchain-ai/deepagents/issues/6168)) ([134ffc7](https://github.com/langchain-ai/deepagents/commit/134ffc7bb4eba504ee90aaae3a70df87c6ea3217))
* **talon:** surface indexing failures and bound the indexing worker ([#6163](https://github.com/langchain-ai/deepagents/issues/6163)) ([03653d6](https://github.com/langchain-ai/deepagents/commit/03653d6d95ceb0b1b6739998fd9475e304489d12))
* **talon:** unwind the channel that fails mid-start, and keep teardown safe ([#6169](https://github.com/langchain-ai/deepagents/issues/6169)) ([5406820](https://github.com/langchain-ai/deepagents/commit/5406820ec1c00ce521b0f54bb786c511da092f01))

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
