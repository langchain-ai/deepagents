# Changelog

## [0.0.9](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.8...deepagents-talon==0.0.9) (2026-10-01)


### Features

* **talon:** add `/context-doctor` command ([#6557](https://github.com/langchain-ai/deepagents/issues/6557)) ([60c0228](https://github.com/langchain-ai/deepagents/commit/60c0228aad14d042080743ead04e135994d3fa75))
* **talon:** add `/model` command for per-chat model switching ([#6577](https://github.com/langchain-ai/deepagents/issues/6577)) ([40873ba](https://github.com/langchain-ai/deepagents/commit/40873baa880133ae19ea921239ad9142c1972f4f))
* **talon:** add approved one-off `ask_for_help` consultations ([#6605](https://github.com/langchain-ai/deepagents/issues/6605)) ([15d8f47](https://github.com/langchain-ai/deepagents/commit/15d8f471d8c6ba6868fc6d94beb7fad89ef7406f))
* **talon:** add cron expressions and `until` expiry to scheduled jobs ([#6526](https://github.com/langchain-ai/deepagents/issues/6526)) ([71ae3f6](https://github.com/langchain-ai/deepagents/commit/71ae3f69b161fdb0d009fe91e112e9aa39369a4d))
* **talon:** add general-purpose safety preflight skill ([#6587](https://github.com/langchain-ai/deepagents/issues/6587)) ([86923e1](https://github.com/langchain-ai/deepagents/commit/86923e160c18a6c090045ef1d60a17bd0d65760d))
* **talon:** add sender pairing for DM access ([#6579](https://github.com/langchain-ai/deepagents/issues/6579)) ([135e849](https://github.com/langchain-ai/deepagents/commit/135e849484d15974a5dbc997828c9889b2d098fd))
* **talon:** add Slack channel adapter ([#6575](https://github.com/langchain-ai/deepagents/issues/6575)) ([15bd7fd](https://github.com/langchain-ai/deepagents/commit/15bd7fdcfb5b6a02a650a5faa7aced0faf6dd6d8))
* **talon:** admit paired senders in every chat ([#6592](https://github.com/langchain-ai/deepagents/issues/6592)) ([6744083](https://github.com/langchain-ai/deepagents/commit/6744083d1978b187fce1de0226aad3c5c9a647dc))
* **talon:** allow outbound Slack user mentions ([#6602](https://github.com/langchain-ai/deepagents/issues/6602)) ([b329965](https://github.com/langchain-ai/deepagents/commit/b32996518f9bad4d156d19e35915d51ead7e6019))
* **talon:** batch concurrent tool approvals ([#6435](https://github.com/langchain-ai/deepagents/issues/6435)) ([1660147](https://github.com/langchain-ai/deepagents/commit/16601472424400ab4a90608d7d95e8fc61f23267))
* **talon:** deliver thread cron results to the channel by default ([#6632](https://github.com/langchain-ai/deepagents/issues/6632)) ([07628a5](https://github.com/langchain-ai/deepagents/commit/07628a59010348aca443087a19c477f148997b5b))
* **talon:** opt-in sandboxed execution via `DEEPAGENTS_TALON_SANDBOX` ([#6574](https://github.com/langchain-ai/deepagents/issues/6574)) ([b5f22b0](https://github.com/langchain-ai/deepagents/commit/b5f22b01aff7cfe328d632de086c31247c2b1081))
* **talon:** register chat commands as Discord slash commands ([#6303](https://github.com/langchain-ai/deepagents/issues/6303)) ([f36b641](https://github.com/langchain-ai/deepagents/commit/f36b641ce3e1f44dbd47835d6dc9f5bb1c82324a))
* **talon:** run a scheduled job's subagents inline ([#6425](https://github.com/langchain-ai/deepagents/issues/6425)) ([c3a041e](https://github.com/langchain-ai/deepagents/commit/c3a041e3d8f593e4e4c9bfc273d52264ceff165b))
* **talon:** send Slack mention pairing codes in DMs ([#6584](https://github.com/langchain-ai/deepagents/issues/6584)) ([a31b310](https://github.com/langchain-ai/deepagents/commit/a31b31009b3f7fad90cc567cd794def20e280bd3))
* **talon:** share public Discord thread history by guild channel ([#6606](https://github.com/langchain-ai/deepagents/issues/6606)) ([936a37e](https://github.com/langchain-ai/deepagents/commit/936a37ef7b85a5ea3a2dca989e3e3a218fb96700))
* **talon:** support explicit public MCP OAuth clients ([#6670](https://github.com/langchain-ai/deepagents/issues/6670)) ([7475059](https://github.com/langchain-ai/deepagents/commit/7475059432e9eda77b8b9a5bc7178f5f3a532bd9))
* **talon:** support sender pairing on Slack ([#6582](https://github.com/langchain-ai/deepagents/issues/6582)) ([90cd463](https://github.com/langchain-ai/deepagents/commit/90cd463830418f79dc994f7b84c01f6256409435))


### Bug Fixes

* **talon:** accept any Slack slash command name ([#6589](https://github.com/langchain-ai/deepagents/issues/6589)) ([31ae532](https://github.com/langchain-ai/deepagents/commit/31ae53291c23854912d3237c20e108445c6b6f11))
* **talon:** accept labeled Slack OAuth callbacks ([#6593](https://github.com/langchain-ai/deepagents/issues/6593)) ([0359577](https://github.com/langchain-ai/deepagents/commit/035957706fc34545ae1e7485aa03f8415694cd71))
* **talon:** avoid retranscribing voice on background follow-ups ([#6682](https://github.com/langchain-ai/deepagents/issues/6682)) ([1614a38](https://github.com/langchain-ai/deepagents/commit/1614a389ea48c12182c167d3ecf69224a69d4043))
* **talon:** declare the packages talon imports directly ([#6436](https://github.com/langchain-ai/deepagents/issues/6436)) ([c63eba4](https://github.com/langchain-ai/deepagents/commit/c63eba426b28f1d1685e40767f087dcc3f34334c))
* **talon:** deliver narration before running tools ([#6681](https://github.com/langchain-ai/deepagents/issues/6681)) ([a8bf07b](https://github.com/langchain-ai/deepagents/commit/a8bf07bf604524e98530034027414254ae215f78))
* **talon:** enumerate selectable gateway models ([#6595](https://github.com/langchain-ai/deepagents/issues/6595)) ([5b29445](https://github.com/langchain-ai/deepagents/commit/5b29445dd387bb9e8b8c3fed8f7382a91a990e06))
* **talon:** exclude OAuth callbacks from Slack thread context ([#6687](https://github.com/langchain-ai/deepagents/issues/6687)) ([f566a09](https://github.com/langchain-ai/deepagents/commit/f566a09da6141935ee564caa01c921da08f34036))
* **talon:** improve subagent research handoffs ([#6674](https://github.com/langchain-ai/deepagents/issues/6674)) ([ab57eab](https://github.com/langchain-ai/deepagents/commit/ab57eab7e51121a2677d0c5b5aa396487ed2b9ee))
* **talon:** include preceding Slack replies on thread mentions ([#6634](https://github.com/langchain-ai/deepagents/issues/6634)) ([0e5f6f8](https://github.com/langchain-ai/deepagents/commit/0e5f6f893b91bed16837f2cd7966049c7cafc650))
* **talon:** index only visible conversation text ([#6358](https://github.com/langchain-ai/deepagents/issues/6358)) ([f6850e9](https://github.com/langchain-ai/deepagents/commit/f6850e9d25854b5679010c08f02adae99e6b8a2c))
* **talon:** keep legacy Slack thread archives writable ([#6679](https://github.com/langchain-ai/deepagents/issues/6679)) ([166b0e0](https://github.com/langchain-ai/deepagents/commit/166b0e0ce1e424bb24f49ee5c48155458b4b1dce))
* **talon:** let scheduled jobs read their origin chat's history ([#6583](https://github.com/langchain-ai/deepagents/issues/6583)) ([e70cf09](https://github.com/langchain-ai/deepagents/commit/e70cf09267d93ce54f381d77fb577a76adac6300))
* **talon:** persist history vector deduplication ([#6357](https://github.com/langchain-ai/deepagents/issues/6357)) ([b3d4967](https://github.com/langchain-ai/deepagents/commit/b3d4967bdcda5b42e976b3e97c357deea82d63ac))
* **talon:** persist model selection across conversations ([#6603](https://github.com/langchain-ai/deepagents/issues/6603)) ([156a682](https://github.com/langchain-ai/deepagents/commit/156a6824e53c3f483dca75d08e0d084c0788b3a3))
* **talon:** preserve MIME types when downloading WhatsApp voice messages ([#6590](https://github.com/langchain-ai/deepagents/issues/6590)) ([e739e4d](https://github.com/langchain-ai/deepagents/commit/e739e4d66c367f91b430062ee841055ac311c57c))
* **talon:** retry provider overload and connection errors ([#6580](https://github.com/langchain-ai/deepagents/issues/6580)) ([b3f1a39](https://github.com/langchain-ai/deepagents/commit/b3f1a3919fe39853ee29acee1fb49b077507428f))
* **talon:** retry statusless provider overload errors ([#6304](https://github.com/langchain-ai/deepagents/issues/6304)) ([f8acbd0](https://github.com/langchain-ai/deepagents/commit/f8acbd0ae24278d35aeb1fa014056084a58f70d2))
* **talon:** serialize cron store mutations ([#6688](https://github.com/langchain-ai/deepagents/issues/6688)) ([67b0bef](https://github.com/langchain-ai/deepagents/commit/67b0bef01082bf94e8641bf4e1a94a71b3f6ebc9))
* **talon:** share Slack channel conversation archives across threads ([#6665](https://github.com/langchain-ai/deepagents/issues/6665)) ([6bf7665](https://github.com/langchain-ai/deepagents/commit/6bf76658f88af8328d9a9625d2c6444e01896248))
* **talon:** stop swallowing a turn's cancel during typing cleanup ([#6594](https://github.com/langchain-ai/deepagents/issues/6594)) ([eff90de](https://github.com/langchain-ai/deepagents/commit/eff90dec8c6bfe9636da064c7c9310b660d89b32))
* **talon:** store offloaded artifacts in the assistant home ([#6530](https://github.com/langchain-ai/deepagents/issues/6530)) ([74d768b](https://github.com/langchain-ai/deepagents/commit/74d768bd45dd4eca977dc320ebfe5838f1691682))
* **talon:** transcribe inbound audio attachments ([#6680](https://github.com/langchain-ai/deepagents/issues/6680)) ([184f547](https://github.com/langchain-ai/deepagents/commit/184f547e12f57df358eb49536a02cbe672f95888))
* **talon:** validate archive pagination arguments ([#6442](https://github.com/langchain-ai/deepagents/issues/6442)) ([f84d963](https://github.com/langchain-ai/deepagents/commit/f84d96358a111872fba3f9b69c5b08ad1565c083))


### Performance Improvements

* **talon:** reduce archive replay transactions ([#6320](https://github.com/langchain-ai/deepagents/issues/6320)) ([89bd48f](https://github.com/langchain-ai/deepagents/commit/89bd48f7d531c9affafa3f0dc096083cbc5969dd))

## [0.0.8](https://github.com/langchain-ai/deepagents/compare/deepagents-talon==0.0.7...deepagents-talon==0.0.8) (2026-09-11)

### Features

- Added `send_message` support for progress updates. ([#6264](https://github.com/langchain-ai/deepagents/pull/6264))
- Added targeted conversation deletion. ([#6235](https://github.com/langchain-ai/deepagents/pull/6235))
- Added support for managing tool approvals through `tools.json`. ([#6248](https://github.com/langchain-ai/deepagents/pull/6248))

### Bug Fixes

- Improved MCP reliability and safety by hardening OAuth device-flow and credential handling, preserving refresh tokens, making configuration updates bounded and non-destructive, reporting protocol errors to the model, and preventing URL swaps past the auto-approve guard. ([#6170](https://github.com/langchain-ai/deepagents/pull/6170), [#6172](https://github.com/langchain-ai/deepagents/pull/6172), [#6171](https://github.com/langchain-ai/deepagents/pull/6171), [#6239](https://github.com/langchain-ai/deepagents/pull/6239), [#6173](https://github.com/langchain-ai/deepagents/pull/6173))
- Improved background subagent and conversation reliability, including recoverable start/stop behavior, scheduled-job delivery, clearer orchestration enforcement, preventing one conversation from stalling or outliving the rest, and suppressing results from discarded turns. ([#6166](https://github.com/langchain-ai/deepagents/pull/6166), [#6228](https://github.com/langchain-ai/deepagents/pull/6228), [#6167](https://github.com/langchain-ai/deepagents/pull/6167), [#6168](https://github.com/langchain-ai/deepagents/pull/6168), [#6231](https://github.com/langchain-ai/deepagents/pull/6231))
- Improved indexing and vector maintenance by bounding archive scans, query embeddings, deletion markers, vector rebuilds, and indexing workers, while surfacing indexing failures and resolving import-cycle and PostgreSQL dependency issues. ([#6162](https://github.com/langchain-ai/deepagents/pull/6162), [#6165](https://github.com/langchain-ai/deepagents/pull/6165), [#6163](https://github.com/langchain-ai/deepagents/pull/6163))
- Fixed local voice transcription and embedding retries, and ensured local model downloads persist. ([#6255](https://github.com/langchain-ai/deepagents/pull/6255), [#6137](https://github.com/langchain-ai/deepagents/pull/6137))
- Improved startup, teardown, and provider cleanup by safely unwinding channels that fail mid-start, sharing the optional-driver loader, and closing provider clients. ([#6169](https://github.com/langchain-ai/deepagents/pull/6169), [#6164](https://github.com/langchain-ai/deepagents/pull/6164))

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
