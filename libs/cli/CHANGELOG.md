# Changelog

## [0.0.27](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.26...deepagents-cli==0.0.27) (2026-03-04)

### Features

* Add background PyPI update check ([#1648](https://github.com/langchain-ai/deepagents/issues/1648)) ([2e7a5e7](https://github.com/langchain-ai/deepagents/commit/2e7a5e7d97f64147ab2d000fae833fe681f1d6b2))
* Add install script ([#1649](https://github.com/langchain-ai/deepagents/issues/1649)) ([68f6ef9](https://github.com/langchain-ai/deepagents/commit/68f6ef96e7d66b2c98d1371e91e5d25f107b80fe))
* Add fuzzy search to model switcher ([#1266](https://github.com/langchain-ai/deepagents/issues/1266)) ([a6bbb18](https://github.com/langchain-ai/deepagents/commit/a6bbb182a2336ba748d93a06b9fcf27966321e20))
* Add model usage stats display ([#1587](https://github.com/langchain-ai/deepagents/issues/1587)) ([a1208db](https://github.com/langchain-ai/deepagents/commit/a1208db096761eb54e0fe712a5aa922502575cb6))
* Add substring matching to command history navigation ([#1301](https://github.com/langchain-ai/deepagents/issues/1301)) ([e276d5a](https://github.com/langchain-ai/deepagents/commit/e276d5a64bee9394f53ab993b01447023bcd4c7d))

### Bug Fixes

* Allow Esc to exit command/bash input mode ([#1644](https://github.com/langchain-ai/deepagents/issues/1644)) ([906da72](https://github.com/langchain-ai/deepagents/commit/906da72ea40e16492f8e7f3c35758af486c92b3c))
* Make `!` bash commands interruptible via `Esc`/`Ctrl+C` ([#1638](https://github.com/langchain-ai/deepagents/issues/1638)) ([0c414d1](https://github.com/langchain-ai/deepagents/commit/0c414d154a74cfabebfae8fc2dbb6d7e39da3857))
* Make escape reject pending HITL approval first ([#1645](https://github.com/langchain-ai/deepagents/issues/1645)) ([5d7be0c](https://github.com/langchain-ai/deepagents/commit/5d7be0c1a2fbe54f7fe062c5a43a7591aecb00e4))
* Show cwd on startup ([#1209](https://github.com/langchain-ai/deepagents/issues/1209)) ([23032dd](https://github.com/langchain-ai/deepagents/commit/23032ddd80b0ec8bf58c91776e62b834f6e03b5e))
* Terminate active subprocesses on app quit ([#1646](https://github.com/langchain-ai/deepagents/issues/1646)) ([5f2e614](https://github.com/langchain-ai/deepagents/commit/5f2e614f05912d3278a988cb7366612099105acf))
* Use first-class OpenRouter attribution kwargs ([#1635](https://github.com/langchain-ai/deepagents/issues/1635)) ([9c1ed93](https://github.com/langchain-ai/deepagents/commit/9c1ed93861a52b9ced2c1426131d542f50afa623))

## [0.0.26](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.25...deepagents-cli==0.0.26) (2026-03-03)

### Features

* Compaction hook ([#1420](https://github.com/langchain-ai/deepagents/issues/1420)) ([e87cdad](https://github.com/langchain-ai/deepagents/commit/e87cdaddb9a984c4fd189b4f71303881edb32cb2))
  * `/compact` command ([#1579](https://github.com/langchain-ai/deepagents/issues/1579)) ([46e9e95](https://github.com/langchain-ai/deepagents/commit/46e9e950087e973175d49d6a863cfa9d2f241528))
* Add `--profile-override` CLI flag ([#1605](https://github.com/langchain-ai/deepagents/issues/1605)) ([1984099](https://github.com/langchain-ai/deepagents/commit/1984099ae9ac4b0c13dc08722abb9d56055da7b7))
* Add model profile overrides to config ([#1603](https://github.com/langchain-ai/deepagents/issues/1603)) ([d3d6899](https://github.com/langchain-ai/deepagents/commit/d3d6899209b7cf97447da0eee642b3f55261ffbc))
* Show summarization status and notification    ([#919](https://github.com/langchain-ai/deepagents/issues/919)) ([2e3cb74](https://github.com/langchain-ai/deepagents/commit/2e3cb743eff8e0a33b215359132cee13a673a4df))

### Bug Fixes

* Fix image path pasting qualms ([#1560](https://github.com/langchain-ai/deepagents/issues/1560)) ([8caaf3e](https://github.com/langchain-ai/deepagents/commit/8caaf3e71ae7f5a26c20ca86700cc51f3c6f37ed))
* Load `.agents` skill alias directories at interactive startup ([#1556](https://github.com/langchain-ai/deepagents/issues/1556)) ([af0a759](https://github.com/langchain-ai/deepagents/commit/af0a759ee231cfe8860da34fe39dbcff38726102))
* Coerce execute timeout to int before formatting tool display ([#1588](https://github.com/langchain-ai/deepagents/issues/1588)) ([04b8c72](https://github.com/langchain-ai/deepagents/commit/04b8c72361f7eb60b86fa560ef3f6283912c3395)), closes [#1586](https://github.com/langchain-ai/deepagents/issues/1586)
* Add missing flags to help screen ([#1619](https://github.com/langchain-ai/deepagents/issues/1619)) ([6067749](https://github.com/langchain-ai/deepagents/commit/60677492b3f49adc8535b34156029271a0728923))
* Align compaction messaging across `/compact` and `compact_conversation` ([#1583](https://github.com/langchain-ai/deepagents/issues/1583)) ([d455a6b](https://github.com/langchain-ai/deepagents/commit/d455a6b117dbca2dfb5156050273a84946adc247))
* Apply profile overrides in `/compact` ([#1612](https://github.com/langchain-ai/deepagents/issues/1612)) ([a9dc2c5](https://github.com/langchain-ai/deepagents/commit/a9dc2c5a1ad6d37f3f682491664b3f709cad8552))
* Disambiguate `/tokens` vs `/compact` token reporting ([#1618](https://github.com/langchain-ai/deepagents/issues/1618)) ([51c3347](https://github.com/langchain-ai/deepagents/commit/51c3347e5a402115d4ecbb09f0074c607270f992))
* Make LangSmith URL lookups non-blocking ([#1595](https://github.com/langchain-ai/deepagents/issues/1595)) ([572eaee](https://github.com/langchain-ai/deepagents/commit/572eaeefbe2f9318555733977e4771815879273c))
* Only exit input mode on backspace, not text clear ([#1479](https://github.com/langchain-ai/deepagents/issues/1479)) ([da0965e](https://github.com/langchain-ai/deepagents/commit/da0965ee33e6bdf7aec30865bed44a1bd38a7d12))
* Retry langsmith project url lookup until project exists ([#1562](https://github.com/langchain-ai/deepagents/issues/1562)) ([e137a63](https://github.com/langchain-ai/deepagents/commit/e137a633fdadda205b8e05a9fdabc4b978726a37))
* Show model info in `/tokens` before first usage ([#1607](https://github.com/langchain-ai/deepagents/issues/1607)) ([7b01ae7](https://github.com/langchain-ai/deepagents/commit/7b01ae7258ed079046262d1c174f1c406101294c))
* Support `timeout=0` for sandbox `execute()` ([#1558](https://github.com/langchain-ai/deepagents/issues/1558)) ([ed14443](https://github.com/langchain-ai/deepagents/commit/ed14443b5aec8afde1f74bb2e12a17cb7d1829b6))
* Unreachable `except` block ([#1535](https://github.com/langchain-ai/deepagents/issues/1535)) ([0e17e35](https://github.com/langchain-ai/deepagents/commit/0e17e352fa2ae4e34320a27d272586a10a0a7aec))

### Performance Improvements

* Optimize thread resume path with prefetch and batched hydration ([#1561](https://github.com/langchain-ai/deepagents/issues/1561)) ([068d112](https://github.com/langchain-ai/deepagents/commit/068d1128177de0f0a01f533a01184039c2a2f09f))
* Parallelize detection scripts for faster first-turn ([#1541](https://github.com/langchain-ai/deepagents/issues/1541)) ([dad8b6e](https://github.com/langchain-ai/deepagents/commit/dad8b6e15a78d26921c0cb831579648927caa551))
* Speed up `/threads` first-open ([#1481](https://github.com/langchain-ai/deepagents/issues/1481)) ([b248b15](https://github.com/langchain-ai/deepagents/commit/b248b15fd70de3c4d055b68a0dae04f00e41ea9e))

## [0.0.25](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.24...deepagents-cli==0.0.25) (2026-02-20)

### Features

* Set OpenRouter headers, default to `gemini-3.1-pro-preview` ([#1455](https://github.com/langchain-ai/deepagents/issues/1455)) ([95c0b71](https://github.com/langchain-ai/deepagents/commit/95c0b71c2fafbec8424d92e7698563045a787866)), closes [#1454](https://github.com/langchain-ai/deepagents/issues/1454)

### Bug Fixes

* Duplicate paste issue ([#1460](https://github.com/langchain-ai/deepagents/issues/1460)) ([9177515](https://github.com/langchain-ai/deepagents/commit/9177515c8a968882e980d229fb546c9753475de7)), closes [#1425](https://github.com/langchain-ai/deepagents/issues/1425)
* Remove model fallback to env variables ([#1458](https://github.com/langchain-ai/deepagents/issues/1458)) ([c9b4275](https://github.com/langchain-ai/deepagents/commit/c9b4275e22fda5aa35b3ddce924277ec8aaa9e1f))

## [0.0.24](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.23...deepagents-cli==0.0.24) (2026-02-20)

### Features

* Add single-click link opening for rich-style hyperlinks ([#1433](https://github.com/langchain-ai/deepagents/issues/1433)) ([ef1fd31](https://github.com/langchain-ai/deepagents/commit/ef1fd3115d77cd769e664d2ad0345623f9ce4019))
* Display model name and context window size using `/tokens` ([#1441](https://github.com/langchain-ai/deepagents/issues/1441)) ([ff7ef0f](https://github.com/langchain-ai/deepagents/commit/ff7ef0f87e6dfc6c581edb34b1a57be7ff6e059c))
* Refresh local context after summarization events ([#1384](https://github.com/langchain-ai/deepagents/issues/1384)) ([dcb9583](https://github.com/langchain-ai/deepagents/commit/dcb95839de360f03d2fc30c9144096874b24006f))
* Windowed thread hydration and configurable thread limit ([#1435](https://github.com/langchain-ai/deepagents/issues/1435)) ([9da8d0b](https://github.com/langchain-ai/deepagents/commit/9da8d0b5c86441e87b85ee6f8db1d23848a823ed))
* Add per-command `timeout` override to `execute()` ([#1154](https://github.com/langchain-ai/deepagents/issues/1154)) ([49277d4](https://github.com/langchain-ai/deepagents/commit/49277d45a026c86b5bf176142dcb1dfc2c7643ae))

### Bug Fixes

* Escape `Rich` markup in shell command display ([#1413](https://github.com/langchain-ai/deepagents/issues/1413)) ([c330290](https://github.com/langchain-ai/deepagents/commit/c33029032a1e2072dab2d06e93953f2acaa6d400))
* Load root-level `AGENTS.md` into agent system prompt ([#1445](https://github.com/langchain-ai/deepagents/issues/1445)) ([047fa2c](https://github.com/langchain-ai/deepagents/commit/047fa2cadfb9f005410c21a6e1e3b3d59eadda7d))
* Prevent crash when quitting with queued messages ([#1421](https://github.com/langchain-ai/deepagents/issues/1421)) ([a3c9ae6](https://github.com/langchain-ai/deepagents/commit/a3c9ae681501cd3efca82573a8d20a0dc8c9b338))

## [0.0.23](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.22...deepagents-cli==0.0.23) (2026-02-18)

### Features

* Add drag-and-drop image attachment to chat input ([#1386](https://github.com/langchain-ai/deepagents/issues/1386)) ([cd3d89b](https://github.com/langchain-ai/deepagents/commit/cd3d89b4419b4c164915ff745afff99cb11b55a5))
* Add skill deletion command ([#580](https://github.com/langchain-ai/deepagents/issues/580)) ([40a8d86](https://github.com/langchain-ai/deepagents/commit/40a8d866f952e0cf8d856e2fa360de771721b99a))
* Add visual mode indicators to chat input ([#1371](https://github.com/langchain-ai/deepagents/issues/1371)) ([1ea6159](https://github.com/langchain-ai/deepagents/commit/1ea6159b068b8c7d721d90a5c196e2eb9877c1c5))
* Dismiss completion dropdown on `esc` ([#1362](https://github.com/langchain-ai/deepagents/issues/1362)) ([961b7fc](https://github.com/langchain-ai/deepagents/commit/961b7fc764a7fbf63466d78c1d80b154b5d1692b))
* Expand local context & implement via bash for sandbox support ([#1295](https://github.com/langchain-ai/deepagents/issues/1295)) ([de8bc7c](https://github.com/langchain-ai/deepagents/commit/de8bc7cbbd7780ef250b3838f61ace85d4465c0a))
* Show sdk version alongside cli version ([#1378](https://github.com/langchain-ai/deepagents/issues/1378)) ([e99b4c8](https://github.com/langchain-ai/deepagents/commit/e99b4c864afd01d68c3829304fb93cc0530eedee))
* Strip mode-trigger prefix from chat input text ([#1373](https://github.com/langchain-ai/deepagents/issues/1373)) ([6879eff](https://github.com/langchain-ai/deepagents/commit/6879effb37c2160ef3835cd2d058b79f9d3a5a99))

### Bug Fixes

* Path hardening ([#918](https://github.com/langchain-ai/deepagents/issues/918)) ([fc34a14](https://github.com/langchain-ai/deepagents/commit/fc34a144a2791c75f8b4c11f67dd1adbc029c81e))
* Only navigate prompt history at input boundaries ([#1385](https://github.com/langchain-ai/deepagents/issues/1385)) ([6d82d6d](https://github.com/langchain-ai/deepagents/commit/6d82d6de290e73b897a58d724f3dfc7a32a06cba))
* Substitute image base64 for placeholder in result block ([#1381](https://github.com/langchain-ai/deepagents/issues/1381)) ([54f4d8e](https://github.com/langchain-ai/deepagents/commit/54f4d8e834c4aad672d78b4130cd43f2454424fa))

### Performance Improvements

* Defer more heavy imports to speed up startup ([#1389](https://github.com/langchain-ai/deepagents/issues/1389)) ([4dd10d5](https://github.com/langchain-ai/deepagents/commit/4dd10d5c9f3cfe13cd7b9ac18a1799c0832976ff))

## [0.0.22](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.21...deepagents-cli==0.0.22) (2026-02-17)

### Features

* Add `langchain-openrouter` ([#1340](https://github.com/langchain-ai/deepagents/issues/1340)) ([5b35247](https://github.com/langchain-ai/deepagents/commit/5b35247b126ed328e9562ac3a3c2acd184b39011))
* Update system & default prompt ([#1293](https://github.com/langchain-ai/deepagents/issues/1293)) ([2aeb092](https://github.com/langchain-ai/deepagents/commit/2aeb092e027affd9eaa8a78b33101e1fd930d444))
* Warn when ripgrep is not installed ([#1337](https://github.com/langchain-ai/deepagents/issues/1337)) ([0367efa](https://github.com/langchain-ai/deepagents/commit/0367efa323b7a29c015d6a3fbb5af8894dc724b8))
* Ensure dep group version match for CLI ([#1316](https://github.com/langchain-ai/deepagents/issues/1316)) ([db05de1](https://github.com/langchain-ai/deepagents/commit/db05de1b0c92208b9752f3f03fa5fa54813ab4ef))
* Enable type checking in `deepagents` and resolve most linting issues ([#991](https://github.com/langchain-ai/deepagents/issues/991)) ([5c90376](https://github.com/langchain-ai/deepagents/commit/5c90376c02754c67d448908e55d1e953f54b8acd))

### Bug Fixes

* Handle `None` selection endpoint, `IndexError` in clipboard copy ([#1342](https://github.com/langchain-ai/deepagents/issues/1342)) ([5754031](https://github.com/langchain-ai/deepagents/commit/57540316cf928da3dcf4401fb54a5d0102045d67))

### Performance Improvements

* Defer heavy imports ([#1361](https://github.com/langchain-ai/deepagents/issues/1361)) ([dd992e4](https://github.com/langchain-ai/deepagents/commit/dd992e48feb3e3a9fc6fd93f56e9d8a9cb51c7bf))

## [0.0.21](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.20...deepagents-cli==0.0.21) (2026-02-11)

### Features

* Support piped stdin as prompt input ([#1254](https://github.com/langchain-ai/deepagents/issues/1254)) ([cca61ff](https://github.com/langchain-ai/deepagents/commit/cca61ff5edb5e2424bfc54b2ac33b59a520fdd6a))
* Add `/threads` command switcher ([#1262](https://github.com/langchain-ai/deepagents/issues/1262)) ([45bf38d](https://github.com/langchain-ai/deepagents/commit/45bf38d7c5ca7ca05ec58c320494a692e419b632)), closes [#1111](https://github.com/langchain-ai/deepagents/issues/1111)
* Make thread link clickable when switching ([#1296](https://github.com/langchain-ai/deepagents/issues/1296)) ([9409520](https://github.com/langchain-ai/deepagents/commit/9409520d524c576c3b0b9686c96a1749ee9dcbbb)), closes [#1291](https://github.com/langchain-ai/deepagents/issues/1291)
* Add `/trace` command to open LangSmith thread, link in switcher ([#1291](https://github.com/langchain-ai/deepagents/issues/1291)) ([fbbd45b](https://github.com/langchain-ai/deepagents/commit/fbbd45b51be2cf09726a3cd0adfcb09cb2b1ff46))
* Add `/changelog`, `/feedback`, `/docs` ([#1261](https://github.com/langchain-ai/deepagents/issues/1261)) ([4561afb](https://github.com/langchain-ai/deepagents/commit/4561afbea17bb11f7fc02ae9f19db15229656280))
* Show langsmith thread url on session teardown ([#1285](https://github.com/langchain-ai/deepagents/issues/1285)) ([899fd1c](https://github.com/langchain-ai/deepagents/commit/899fd1cdea6f7b2003992abd3f6173d630849a90))

### Bug Fixes

* Fix stale model settings during model hot-swap ([#1257](https://github.com/langchain-ai/deepagents/issues/1257)) ([55c119c](https://github.com/langchain-ai/deepagents/commit/55c119cb6ce73db7cae0865172f00ab8fc9f8fc1))

## [0.0.20](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.19...deepagents-cli==0.0.20) (2026-02-10)

### Features

* `--quiet` flag to suppress non-agent output w/ `-n` ([#1201](https://github.com/langchain-ai/deepagents/issues/1201)) ([3e96792](https://github.com/langchain-ai/deepagents/commit/3e967926655cf5249a1bc5ca3edd48da9dd3061b))
* Add docs link to `/help` ([#1098](https://github.com/langchain-ai/deepagents/issues/1098)) ([8f8fc98](https://github.com/langchain-ai/deepagents/commit/8f8fc98bd403d96d6ed95fce8906d9c881236613))
* Built-in skills, ship `skill-creator` as first ([#1191](https://github.com/langchain-ai/deepagents/issues/1191)) ([42823a8](https://github.com/langchain-ai/deepagents/commit/42823a88d1eb7242a5d9b3eba981f24b3ea9e274))
* Enrich built-in skill metadata with license and compatibility info ([#1193](https://github.com/langchain-ai/deepagents/issues/1193)) ([b8179c2](https://github.com/langchain-ai/deepagents/commit/b8179c23f9130c92cb1fb7c6b34d98cc32ec092a))
* Implement message queue for CLI ([#1197](https://github.com/langchain-ai/deepagents/issues/1197)) ([c4678d7](https://github.com/langchain-ai/deepagents/commit/c4678d7641785ac4f17045eb75d55f9dc44f37fe))
* Model switcher & arbitrary chat model support ([#1127](https://github.com/langchain-ai/deepagents/issues/1127)) ([28fc311](https://github.com/langchain-ai/deepagents/commit/28fc311da37881257e409149022f0717f78013ef))
* Non-interactive mode w/ shell allow-listing ([#909](https://github.com/langchain-ai/deepagents/issues/909)) ([433bd2c](https://github.com/langchain-ai/deepagents/commit/433bd2cb493d6c4b59f2833e4304eead0304195a))
* Support custom working directories and LangSmith sandbox templates ([#1099](https://github.com/langchain-ai/deepagents/issues/1099)) ([21e7150](https://github.com/langchain-ai/deepagents/commit/21e715054ea5cf48cab05319b2116509fbacd899))

### Bug Fixes

* `-m` initial prompt submission ([#1184](https://github.com/langchain-ai/deepagents/issues/1184)) ([a702e82](https://github.com/langchain-ai/deepagents/commit/a702e82a0f61edbadd78eff6906ecde20b601798))
* Align skill-creator example scripts with agent skills spec ([#1177](https://github.com/langchain-ai/deepagents/issues/1177)) ([199d176](https://github.com/langchain-ai/deepagents/commit/199d17676ac1bfee645908a6c58193291e522890))
* Harden dictionary iteration and HITL fallback handling ([#1151](https://github.com/langchain-ai/deepagents/issues/1151)) ([8b21fc6](https://github.com/langchain-ai/deepagents/commit/8b21fc6105d808ad25c53de96f339ab21efb4474))
* Per-subcommand help screens, short flags, and skills enhancements ([#1190](https://github.com/langchain-ai/deepagents/issues/1190)) ([3da1e8b](https://github.com/langchain-ai/deepagents/commit/3da1e8bc20bf39aba80f6507b9abc2352de38484))
* Port skills behavior from SDK ([#1192](https://github.com/langchain-ai/deepagents/issues/1192)) ([ad9241d](https://github.com/langchain-ai/deepagents/commit/ad9241da6e7e23e4430756a1d5a3afb6c6bfebcc)), closes [#1189](https://github.com/langchain-ai/deepagents/issues/1189)
* Rewrite skills create template to match spec guidance ([#1178](https://github.com/langchain-ai/deepagents/issues/1178)) ([f08ad52](https://github.com/langchain-ai/deepagents/commit/f08ad520172bd114e4cebf69138a10cbf98e157a))
* Terminal virtualize scrolling to stop perf issues ([#965](https://github.com/langchain-ai/deepagents/issues/965)) ([5633c82](https://github.com/langchain-ai/deepagents/commit/5633c825832a0e8bd645681db23e97af31879b65))
* Update splash thread ID on `/clear` ([#1204](https://github.com/langchain-ai/deepagents/issues/1204)) ([23651ed](https://github.com/langchain-ai/deepagents/commit/23651edbc236e4a68fb0d9496506e6293b836cd9))
* Refactor summarization middleware ([#1138](https://github.com/langchain-ai/deepagents/issues/1138)) ([e87001e](https://github.com/langchain-ai/deepagents/commit/e87001eace2852c2df47095ffd2611f09fdda2f5))

## [0.0.19](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.18...deepagents-cli==0.0.19) (2026-02-06)

### Features

* Add click support and hover styling to autocomplete popup ([#1130](https://github.com/langchain-ai/deepagents/issues/1130)) ([b1cc83d](https://github.com/langchain-ai/deepagents/commit/b1cc83d277e01614b0cc4141993cde40ce68d632))
* Add per-command `timeout` override to `execute` tool ([#1158](https://github.com/langchain-ai/deepagents/issues/1158)) ([cb390ef](https://github.com/langchain-ai/deepagents/commit/cb390ef7a89966760f08c5aceb2211220e8653b8))
* Highlight file mentions and support CJK parsing ([#558](https://github.com/langchain-ai/deepagents/issues/558)) ([cebe333](https://github.com/langchain-ai/deepagents/commit/cebe333246f8bea6b04d6283985e102c2ed5d744))
* Make thread id in splash clickable ([#1159](https://github.com/langchain-ai/deepagents/issues/1159)) ([6087fb2](https://github.com/langchain-ai/deepagents/commit/6087fb276f39ed9a388d722ff1be88d94debf49f))
* Use LocalShellBackend, gives shell to subagents ([#1107](https://github.com/langchain-ai/deepagents/issues/1107)) ([b57ea39](https://github.com/langchain-ai/deepagents/commit/b57ea3906680818b94ecca88b92082d4dea63694))

### Bug Fixes

* Disable iTerm2 cursor guide during execution ([#1123](https://github.com/langchain-ai/deepagents/issues/1123)) ([4eb7d42](https://github.com/langchain-ai/deepagents/commit/4eb7d426eaefa41f74cc6056ae076f475a0a400d))
* Dismiss modal screens on escape key ([#1128](https://github.com/langchain-ai/deepagents/issues/1128)) ([27047a0](https://github.com/langchain-ai/deepagents/commit/27047a085de99fcb9977816663e61114c2b008ac))
* Hide resume hint on app error and improve startup message ([#1135](https://github.com/langchain-ai/deepagents/issues/1135)) ([4e25843](https://github.com/langchain-ai/deepagents/commit/4e258430468b56c3e79499f6b7c5ab7b9cd6f45b))
* Propagate app errors instead of masking ([#1126](https://github.com/langchain-ai/deepagents/issues/1126)) ([79a1984](https://github.com/langchain-ai/deepagents/commit/79a1984629847ce067b6ce78ad14797889724244))
* Remove Interactive Features from --help output ([#1161](https://github.com/langchain-ai/deepagents/issues/1161)) ([a296789](https://github.com/langchain-ai/deepagents/commit/a2967898933b77dd8da6458553f49e717fa732e6))
* Rename `SystemMessage` -&gt; `AppMessage` ([#1113](https://github.com/langchain-ai/deepagents/issues/1113)) ([f576262](https://github.com/langchain-ai/deepagents/commit/f576262aeee54499e9970acf76af93553fccfefd))
* Unify spinner API to support dynamic status text ([#1124](https://github.com/langchain-ai/deepagents/issues/1124)) ([bb55608](https://github.com/langchain-ai/deepagents/commit/bb55608b7172f55df38fef88918b2fded894e3ce))
* Update help text to include `Esc` key for rejection ([#1122](https://github.com/langchain-ai/deepagents/issues/1122)) ([8f4bcf5](https://github.com/langchain-ai/deepagents/commit/8f4bcf52547dcd3e38d4d75ce395eb973a7ee2c0))

## [0.0.18](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.17...deepagents-cli==0.0.18) (2026-02-05)

### Features

* Add langsmith sandbox integration ([#1077](https://github.com/langchain-ai/deepagents/issues/1077)) ([7d17be0](https://github.com/langchain-ai/deepagents/commit/7d17be00b59e586c55517eaca281342e1a6559ff))
* Resume thread enhancements ([#1065](https://github.com/langchain-ai/deepagents/issues/1065)) ([e6663b0](https://github.com/langchain-ai/deepagents/commit/e6663b0b314582583afd32cb906a6d502cd8f16b))
* Support  .`agents/skills` dir alias ([#1059](https://github.com/langchain-ai/deepagents/issues/1059)) ([ec1db17](https://github.com/langchain-ai/deepagents/commit/ec1db172c12bc8b8f85bb03138e442353d4b1013))

### Bug Fixes

* `Ctrl+E` for tool output toggle ([#1100](https://github.com/langchain-ai/deepagents/issues/1100)) ([9fa9d72](https://github.com/langchain-ai/deepagents/commit/9fa9d727dbf6b8996a61f2f764675dbc2e23c1b6))
* Consolidate tool output expand/collapse hint placement ([#1102](https://github.com/langchain-ai/deepagents/issues/1102)) ([70db34b](https://github.com/langchain-ai/deepagents/commit/70db34b5f15a7e81ff586dd0adb2bdfd9ac5d4e9))
* Delete `/exit` ([#1052](https://github.com/langchain-ai/deepagents/issues/1052)) ([8331b77](https://github.com/langchain-ai/deepagents/commit/8331b7790fcf0474e109c3c29f810f4ced0f1745)), closes [#836](https://github.com/langchain-ai/deepagents/issues/836) [#651](https://github.com/langchain-ai/deepagents/issues/651)
* Installed default prompt not updated following upgrade ([#1082](https://github.com/langchain-ai/deepagents/issues/1082)) ([bffd956](https://github.com/langchain-ai/deepagents/commit/bffd95610730c668406c485ad941835a5307c226))
* Replace silent exception handling with proper logging ([#708](https://github.com/langchain-ai/deepagents/issues/708)) ([20faf7a](https://github.com/langchain-ai/deepagents/commit/20faf7ac244d97e688f1cc4121d480ed212fe97c))
* Show full shell command in error output ([#1097](https://github.com/langchain-ai/deepagents/issues/1097)) ([23bb1d8](https://github.com/langchain-ai/deepagents/commit/23bb1d8af85eec8739aea17c3bb3616afb22072a)), closes [#1080](https://github.com/langchain-ai/deepagents/issues/1080)
* Support `-h`/`--help` flags ([#1106](https://github.com/langchain-ai/deepagents/issues/1106)) ([26bebf5](https://github.com/langchain-ai/deepagents/commit/26bebf592ab56ffdc5eeff55bb7c2e542ef8f706))

## [0.0.17](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.16...deepagents-cli==0.0.17) (2026-02-03)

### Features

* Add expandable shell command display in HITL approval ([#976](https://github.com/langchain-ai/deepagents/issues/976)) ([fb8a007](https://github.com/langchain-ai/deepagents/commit/fb8a007123d18025beb1a011f2050e1085dcf69b))
* Model identity ([#770](https://github.com/langchain-ai/deepagents/issues/770)) ([e54a0ee](https://github.com/langchain-ai/deepagents/commit/e54a0ee43c7dfc7fd14c3f43d37cc0ee5e85c5a8))
* Sandbox provider interface ([#900](https://github.com/langchain-ai/deepagents/issues/900)) ([d431cfd](https://github.com/langchain-ai/deepagents/commit/d431cfd4a56713434e84f4fa1cdf4a160b43db95))

## [0.0.16](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.15...deepagents-cli==0.0.16) (2026-02-02)

### Features

* Add configurable timeout to `ShellMiddleware` ([#961](https://github.com/langchain-ai/deepagents/issues/961)) ([bc5e417](https://github.com/langchain-ai/deepagents/commit/bc5e4178a76d795922beab93b87e90ccaf99fba6))
* Add timeout formatting to enhance `shell` command display ([#987](https://github.com/langchain-ai/deepagents/issues/987)) ([cbbfd49](https://github.com/langchain-ai/deepagents/commit/cbbfd49011c9cf93741a024f6efeceeca830820e))
* Display thread ID at splash ([#988](https://github.com/langchain-ai/deepagents/issues/988)) ([e61b9e8](https://github.com/langchain-ai/deepagents/commit/e61b9e8e7af417bf5f636180631dbd47a5bb31bb))

### Bug Fixes

* Improve clipboard copy/paste on macOS ([#960](https://github.com/langchain-ai/deepagents/issues/960)) ([3e1c604](https://github.com/langchain-ai/deepagents/commit/3e1c604474bd98ce1e0ac802df6fb049dd049682))
* Make `pyperclip` hard dep ([#985](https://github.com/langchain-ai/deepagents/issues/985)) ([0f5d4ad](https://github.com/langchain-ai/deepagents/commit/0f5d4ad9e63d415c9b80cd15fa0f89fc2f91357b)), closes [#960](https://github.com/langchain-ai/deepagents/issues/960)
* Revert, improve clipboard copy/paste on macOS ([#964](https://github.com/langchain-ai/deepagents/issues/964)) ([4991992](https://github.com/langchain-ai/deepagents/commit/4991992a5a60fd9588e2110b46440337affc80da))
* Update timeout message for long-running commands in `ShellMiddleware` ([#986](https://github.com/langchain-ai/deepagents/issues/986)) ([dcbe128](https://github.com/langchain-ai/deepagents/commit/dcbe12805a3650e63da89df0774dd7e0181dbaa6))

---

## Prior Releases

Versions prior to 0.0.16 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=deepagents-cli) for details on previous versions.
