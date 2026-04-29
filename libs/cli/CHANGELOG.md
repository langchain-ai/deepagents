# Changelog

## [0.0.42](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.41...deepagents-cli==0.0.42) (2026-04-29)

### Features

* `shift+enter` newline on kitty-capable terminals ([#2869](https://github.com/langchain-ai/deepagents/issues/2869)) ([34e6614](https://github.com/langchain-ai/deepagents/commit/34e6614a2c6e6a7e33f763ac5b527c13b324a690))
* Hint `Enter` behavior in `/model` empty state ([#2933](https://github.com/langchain-ai/deepagents/issues/2933)) ([7cffa16](https://github.com/langchain-ai/deepagents/commit/7cffa16a072bc77def78e4cda09c2e5a2cf8ca39))

### Bug Fixes

* Queue `/model` switches during server startup ([#2895](https://github.com/langchain-ai/deepagents/issues/2895)) ([6ed7b65](https://github.com/langchain-ai/deepagents/commit/6ed7b65a40ac5bd926e94edd944bd552e8d6b5b4))

## [0.0.41](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.40...deepagents-cli==0.0.41) (2026-04-21)

### Features

* `--startup-cmd` flag ([#2841](https://github.com/langchain-ai/deepagents/issues/2841)) ([8adcc2c](https://github.com/langchain-ai/deepagents/commit/8adcc2c2e612346c263d02d1ec5c33e0d63da5a3))
* Actionable notifications, update modal ([#2855](https://github.com/langchain-ai/deepagents/issues/2855)) ([5fcd368](https://github.com/langchain-ai/deepagents/commit/5fcd368088079a84f151e0a3f5e4b9ac29c360c1))
* Custom auth via `[auth]` in `deepagents deploy` ([#2734](https://github.com/langchain-ai/deepagents/issues/2734)) ([417ddaa](https://github.com/langchain-ai/deepagents/commit/417ddaab4804a54a8f83721bb8577d65cfa3659f))
* Refresh footer git branch after shell commands ([#2851](https://github.com/langchain-ai/deepagents/issues/2851)) ([ee4fddd](https://github.com/langchain-ai/deepagents/commit/ee4fddde9454ee8f7ede98ee9a346da6c5ccd3d9))
* Rework `/version`, add release-age and editable-install guard ([#2854](https://github.com/langchain-ai/deepagents/issues/2854)) ([1ae053f](https://github.com/langchain-ai/deepagents/commit/1ae053f347679e58562d2b81eb6d6e6e9bbf0b07))

### Bug Fixes

* Gate agent-swap resume hint on server checkpoint ([#2862](https://github.com/langchain-ai/deepagents/issues/2862)) ([f14b9cd](https://github.com/langchain-ai/deepagents/commit/f14b9cd3cc41f92ddb089d4294cba8496d517807))
* Keep thinking spinner visible during text streaming ([#2849](https://github.com/langchain-ai/deepagents/issues/2849)) ([2cb2244](https://github.com/langchain-ai/deepagents/commit/2cb22446bc1351d20e8bbfa9f23b5d668947351f))
* Re-show thinking spinner on hitl/`ask_user` resume ([#2847](https://github.com/langchain-ai/deepagents/issues/2847)) ([34a6167](https://github.com/langchain-ai/deepagents/commit/34a6167b0c275c7e4da067ef276a837f60da2f8a))
* Restore `ctrl+j` newline keybinding ([#2827](https://github.com/langchain-ai/deepagents/issues/2827)) ([2e1a3f4](https://github.com/langchain-ai/deepagents/commit/2e1a3f45a9ff56390e294ea5b0f6a837200874ec))

## [0.0.40](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.39...deepagents-cli==0.0.40) (2026-04-20)

### Features

* `--max-turns` flag on non-interactive mode ([#2832](https://github.com/langchain-ai/deepagents/issues/2832)) ([567bcd8](https://github.com/langchain-ai/deepagents/commit/567bcd8f2065e089f4905a901efe7154add23121))
* `/agents` switcher ([#2558](https://github.com/langchain-ai/deepagents/issues/2558)) ([3a2b777](https://github.com/langchain-ai/deepagents/commit/3a2b77773a19dd48f9c51dfb35cdfc5c33126d77))

### Bug Fixes

* Normalize serialized message dicts in `/offload` ([#2808](https://github.com/langchain-ai/deepagents/issues/2808)) ([0d11bd4](https://github.com/langchain-ai/deepagents/commit/0d11bd46fd682c7aa3f008918771e9759dca7352))

## [0.0.39](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.38...deepagents-cli==0.0.39) (2026-04-17)

### Features

* Inline argument hints for slash commands ([#2181](https://github.com/langchain-ai/deepagents/issues/2181)) ([6b58e06](https://github.com/langchain-ai/deepagents/commit/6b58e06b06f6fb360d85c54eac31953d1e47dd7a))
* Subagents for `deepagents deploy` ([#2786](https://github.com/langchain-ai/deepagents/issues/2786)) ([7dd5565](https://github.com/langchain-ai/deepagents/commit/7dd5565e30ab91a2cd2ca10fcd227e590451f13c))

### Bug Fixes

* Throttle update notification to once per day and fix teardown banner ([#2764](https://github.com/langchain-ai/deepagents/issues/2764)) ([ba31294](https://github.com/langchain-ai/deepagents/commit/ba31294b733d4ef542d18830eec6df30cfa0d23f))

## [0.0.38](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.37...deepagents-cli==0.0.38) (2026-04-15)

### Features

* User scoped memory for `deepagents deploy` ([#2708](https://github.com/langchain-ai/deepagents/issues/2708)) ([23bfca6](https://github.com/langchain-ai/deepagents/commit/23bfca6e46e6f3e4fba6657d858ddd5a0b06626f))

### Performance Improvements

* `O(1)` message lookups in `MessageStore` ([#2350](https://github.com/langchain-ai/deepagents/issues/2350)) ([d39fd5d](https://github.com/langchain-ai/deepagents/commit/d39fd5d3651fd87d1eea8c02cbef2c2f62449e67))

## [0.0.37](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.36...deepagents-cli==0.0.37) (2026-04-10)

### Features

* Permissions for `deepagents deploy` ([#2651](https://github.com/langchain-ai/deepagents/issues/2651)) ([5d93b73](https://github.com/langchain-ai/deepagents/commit/5d93b736af6ffb165f33569233d533ced95a6943))

### Bug Fixes

* Add missing model provider deps to `deepagents deploy` bundler [closes [#2647](https://github.com/langchain-ai/deepagents/issues/2647)] ([#2660](https://github.com/langchain-ai/deepagents/issues/2660)) ([b710a69](https://github.com/langchain-ai/deepagents/commit/b710a69b12e49479045eaa54dfb709326473500b))
* `AGENTS.md` in system prompt twice ([#2652](https://github.com/langchain-ai/deepagents/issues/2652)) ([9052be9](https://github.com/langchain-ai/deepagents/commit/9052be98d9f4ef9b11a88c9b1df3fae5e5ac666c))
* Harden `deepagents deploy` config parsing and add unit tests ([#2636](https://github.com/langchain-ai/deepagents/issues/2636)) ([0469d14](https://github.com/langchain-ai/deepagents/commit/0469d1429d129e604fc1b622263923162f719314))
* Load `deepagents deploy` project `.env` before deploy/dev config validation ([#2644](https://github.com/langchain-ai/deepagents/issues/2644)) ([8299091](https://github.com/langchain-ai/deepagents/commit/829909166606f8a9d9571b00da725845bad08da7))

## [0.0.36](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.35...deepagents-cli==0.0.36) (2026-04-09)

### Features

* `deepagents deploy` ([#2491](https://github.com/langchain-ai/deepagents/issues/2491)) ([01dc60e](https://github.com/langchain-ai/deepagents/commit/01dc60e394ecb56bd5336e447d32caeed8a67ec2))
* Warn on missing tavily key, add `/notifications` ([#2555](https://github.com/langchain-ai/deepagents/issues/2555)) ([3dff3ed](https://github.com/langchain-ai/deepagents/commit/3dff3ed6835eae9f8b420b8a73c054127faaf7d2))

### Bug Fixes

* Fail fast on missing provider credentials ([#2554](https://github.com/langchain-ai/deepagents/issues/2554)) ([50fb8ae](https://github.com/langchain-ai/deepagents/commit/50fb8aefe7e3065024e10f1d5ecd11a54d736641))
* Fix mktemp template in debug script for macOS ([#2603](https://github.com/langchain-ai/deepagents/issues/2603)) ([63fa537](https://github.com/langchain-ai/deepagents/commit/63fa537e9995ca2ead492ee44902227567e9a130))

## [0.0.35](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.34...deepagents-cli==0.0.35) (2026-04-07)

### Highlights

* **Skills:** Invoke SDK skills directly from the CLI with `/skill:name` or at startup via `--skill`. Skills are a composable extension point for domain-specific agent behaviors.
* **Themes & configuration:** New theme system with color overrides, global dotenv at `~/.deepagents/.env`, and a `DEEPAGENTS_CLI_` env var prefix for conflict-free configuration.
* **Auto-updates:** Full update lifecycle — `/update` to upgrade in-place, `/auto-update` to toggle background checks, and a refreshed install script UX.
* **Headless workflows:** Agent-friendly UX for scripted/headless invocations, AgentCore Code Interpreter sandbox provider, and improved non-interactive tracing and guidance.
* **Performance:** Sub-250ms first paint via aggressive import deferral (pydantic, adapters, heavy SDK modules), markdown stack prewarming, and reduced health-poll intervals.

### Features

* Load `~/.deepagents/.env` as global dotenv ([#1909](https://github.com/langchain-ai/deepagents/issues/1909)) ([5a21d0a](https://github.com/langchain-ai/deepagents/commit/5a21d0add1d2a885c6cd3cf36621d1da690c8db3))
* Skill invocation via `/skill:name` ([#2037](https://github.com/langchain-ai/deepagents/issues/2037)) ([cc8cce7](https://github.com/langchain-ai/deepagents/commit/cc8cce70e1d6c5f897a3cebe0388aa3d774de487))
  * `--skill` startup invocation ([#2477](https://github.com/langchain-ai/deepagents/issues/2477)) ([5f0f1d4](https://github.com/langchain-ai/deepagents/commit/5f0f1d4605de1f1545846e2dbcc29a24c6f138d7))
* Auto-update lifecycle, `/update` command, install script ux ([#2095](https://github.com/langchain-ai/deepagents/issues/2095)) ([fd92f6e](https://github.com/langchain-ai/deepagents/commit/fd92f6eaa87bb1a397f1b7fd216657f354c46e0f))
  * `/auto-update` to toggle auto-updates ([#2276](https://github.com/langchain-ai/deepagents/issues/2276)) ([ad70bde](https://github.com/langchain-ai/deepagents/commit/ad70bde070c58057ff98871fcacd95f03487051e))
* Themes ([#2134](https://github.com/langchain-ai/deepagents/issues/2134)) ([db67af0](https://github.com/langchain-ai/deepagents/commit/db67af07984b5d73711383cf717b5e3d96eac3b8))
  * Allow color overrides on built-in themes, default dark to false ([#2275](https://github.com/langchain-ai/deepagents/issues/2275)) ([8f71865](https://github.com/langchain-ai/deepagents/commit/8f718650ee1913f83e70aa7854e576edc816e694))
* Add `DEEPAGENTS_CLI_` env var prefix and fix dotenv load order ([#2303](https://github.com/langchain-ai/deepagents/issues/2303)) ([29647bb](https://github.com/langchain-ai/deepagents/commit/29647bb4cdd89a1de65f6adb518f708135d59e03))
* Add `ls_integration` metadata to langsmith traces ([#2272](https://github.com/langchain-ai/deepagents/issues/2272)) ([5dd8098](https://github.com/langchain-ai/deepagents/commit/5dd80983a57a5c48bb9c75c6760733882cb908c2))
* Add animated spinner to non-interactive verbose mode ([#2001](https://github.com/langchain-ai/deepagents/issues/2001)) ([153f465](https://github.com/langchain-ai/deepagents/commit/153f465937c09a792119a0e0c8656fa8df29d4e5))
* Add async backend support to local context middleware ([#2118](https://github.com/langchain-ai/deepagents/issues/2118)) ([a0d623c](https://github.com/langchain-ai/deepagents/commit/a0d623cf18d216b5de36cc7d40e7804d5cc4dfa3))
* Agent-friendly ux for scripted/headless workflows ([#2271](https://github.com/langchain-ai/deepagents/issues/2271)) ([386438f](https://github.com/langchain-ai/deepagents/commit/386438f62a458baa367468cf746c3a5387a217d4))
* AgentCore Code Interpreter sandbox provider ([#2120](https://github.com/langchain-ai/deepagents/issues/2120)) ([92556c7](https://github.com/langchain-ai/deepagents/commit/92556c767aaab601d3ea753baf7f52642e5769a4))
* Context-aware connecting banner for resume and local server ([#2092](https://github.com/langchain-ai/deepagents/issues/2092)) ([18b385b](https://github.com/langchain-ai/deepagents/commit/18b385be979d461797a76b6d7217e259fe260b2c))
* Default langsmith project to `'deepagents-cli'` ([#2277](https://github.com/langchain-ai/deepagents/issues/2277)) ([7178b87](https://github.com/langchain-ai/deepagents/commit/7178b8729a03ee1e5508fdf6c111488d18e77e8f))
* Enhance tool-call UI, add `Ctrl+U` shortcut for chat input ([#1757](https://github.com/langchain-ai/deepagents/issues/1757)) ([800c552](https://github.com/langchain-ai/deepagents/commit/800c55213aa4c6515759fb70d36af370feb86302))
* Persist token count in graph state across sessions ([#2323](https://github.com/langchain-ai/deepagents/issues/2323)) ([5be352d](https://github.com/langchain-ai/deepagents/commit/5be352d8bb4d266758f3e934f7affbe8d42b0149))
* Pop queued messages individually on `esc` instead of clearing all ([#2089](https://github.com/langchain-ai/deepagents/issues/2089)) ([c76d855](https://github.com/langchain-ai/deepagents/commit/c76d855e4c73a448af17ade8ad54a2071f2c6bfe))
* Render ask-user questions as markdown ([#2339](https://github.com/langchain-ai/deepagents/issues/2339)) ([5fbb14a](https://github.com/langchain-ai/deepagents/commit/5fbb14a4fd6f2800aec99add1d9f32b88c1d751a))
* Show platform-specific ripgrep install command in missing-tool warning ([#1997](https://github.com/langchain-ai/deepagents/issues/1997)) ([f000ce5](https://github.com/langchain-ai/deepagents/commit/f000ce58ebb6438397e8eb016b2a788a91fe5754))
* Support root/MDM installs ([#2346](https://github.com/langchain-ai/deepagents/issues/2346)) ([f618acc](https://github.com/langchain-ai/deepagents/commit/f618acc671ca2137df165d6f56e8a46f11c63abb))
* Surface unsupported input modalities in system prompt ([#2327](https://github.com/langchain-ai/deepagents/issues/2327)) ([95620e7](https://github.com/langchain-ai/deepagents/commit/95620e796aa6bf11f424d2f83693026fbee6be58))

### Bug Fixes

* Align headless todo guidance with non-interactive mode ([#2459](https://github.com/langchain-ai/deepagents/issues/2459)) ([281899b](https://github.com/langchain-ai/deepagents/commit/281899bf1932f959f512f94f4c2ac5d76ad32ecb))
* Disable markup parsing for blocked-link notifications ([#2170](https://github.com/langchain-ai/deepagents/issues/2170)) ([15867bf](https://github.com/langchain-ai/deepagents/commit/15867bfaeb43b5fca31677fd06b25022aec10c8b))
* Dismiss slash command autocomplete on space ([#2478](https://github.com/langchain-ai/deepagents/issues/2478)) ([02e46bc](https://github.com/langchain-ai/deepagents/commit/02e46bc4cc470d93f5ab3ca05252ef1ffe87c3e7))
* Eliminate autocomplete popup flicker ([#2020](https://github.com/langchain-ai/deepagents/issues/2020)) ([4b2db1e](https://github.com/langchain-ai/deepagents/commit/4b2db1eab412ecfb1343fae123777f14e935fcea))
* Eliminate trace fragmentation in non-interactive mode ([#2136](https://github.com/langchain-ai/deepagents/issues/2136)) ([9bddc52](https://github.com/langchain-ai/deepagents/commit/9bddc52b572672e0564bfc4a8675e106d98116af))
* Enforce approval toggle when launched with `-y` ([#2278](https://github.com/langchain-ai/deepagents/issues/2278)) ([28a32b7](https://github.com/langchain-ai/deepagents/commit/28a32b7dfb6fd6ddb9aa443a9d0d70c33790edbd))
* Escape exception text in rich markup error output ([#2307](https://github.com/langchain-ai/deepagents/issues/2307)) ([42bccca](https://github.com/langchain-ai/deepagents/commit/42bccca05f1733a04ee21da6bf9924140f6e1e65))
* Escape markup in toast notifications ([#2139](https://github.com/langchain-ai/deepagents/issues/2139)) ([90ccc28](https://github.com/langchain-ai/deepagents/commit/90ccc28ace79ae6e2d60d1ccc99f612fb1fcb30e))
* Exit app on `ctrl+d` when thread list is empty ([#2270](https://github.com/langchain-ai/deepagents/issues/2270)) ([e859077](https://github.com/langchain-ai/deepagents/commit/e859077ffa7ee0423bbc7b86d5a4238798ec9c9a))
* Guard against textual cursor/document desync crash ([#2494](https://github.com/langchain-ai/deepagents/issues/2494)) ([c14a748](https://github.com/langchain-ai/deepagents/commit/c14a7483ed92428ec2241dd8254cdab9c26a6a4f))
* Human-readable duration and consistent dim styling on teardown screen ([#1995](https://github.com/langchain-ai/deepagents/issues/1995)) ([901a0a4](https://github.com/langchain-ai/deepagents/commit/901a0a400fccfafbef069d6632acb2227f0aef41))
* Mark token count as approximate after interrupted generation ([#2353](https://github.com/langchain-ai/deepagents/issues/2353)) ([cb9a0c7](https://github.com/langchain-ai/deepagents/commit/cb9a0c7ad2f9421471b4276a33beabc30588b694))
* Resolve misleading "missing package" error when provider import fails ([#1960](https://github.com/langchain-ai/deepagents/issues/1960)) ([b90fbad](https://github.com/langchain-ai/deepagents/commit/b90fbad1335222a3b7ceb469f7f46fedec79ac61))
* Open trace in browser immediately when busy ([#2305](https://github.com/langchain-ai/deepagents/issues/2305)) ([b452032](https://github.com/langchain-ai/deepagents/commit/b452032438f83e0cca245dcabacf176cc1aa8857))
* Patch model identity in system prompt on `/model` swap ([#2024](https://github.com/langchain-ai/deepagents/issues/2024)) ([36aecbf](https://github.com/langchain-ai/deepagents/commit/36aecbf939cafb2d5914620a9e2ec890b02447a4))
* Harden MCP pre-flight health checks ([#2019](https://github.com/langchain-ai/deepagents/issues/2019)) ([2b27055](https://github.com/langchain-ai/deepagents/commit/2b270558fcd846f9f82ffa2bc3b56810b5851d7d))
* Pre-flight health checks for MCP servers ([#2008](https://github.com/langchain-ai/deepagents/issues/2008)) ([30d60e3](https://github.com/langchain-ai/deepagents/commit/30d60e3866bde51a71f607fcd5cbe9dd75c5da75))
* Prevent premature thinking state with parallel subtasks ([#1858](https://github.com/langchain-ai/deepagents/issues/1858)) ([189104c](https://github.com/langchain-ai/deepagents/commit/189104c622a5fa874ad1d4be1c18e012cafd605a))
* Prevent session stats loss on mid-turn exit ([#2238](https://github.com/langchain-ai/deepagents/issues/2238)) ([b1807aa](https://github.com/langchain-ai/deepagents/commit/b1807aab78175c53a2d6f835ae30171d0f802d1c))
* Rebind toggle tool output to `ctrl+o` to unblock `cmd+right` ([#2088](https://github.com/langchain-ai/deepagents/issues/2088)) ([b486fe5](https://github.com/langchain-ai/deepagents/commit/b486fe5df5aa0efd11cdcffdbd96f433c2b127b3))
* Remove duplicate server failure notification ([#2141](https://github.com/langchain-ai/deepagents/issues/2141)) ([c1cfe72](https://github.com/langchain-ai/deepagents/commit/c1cfe72055fd00685cc21a84540f21f4792033ab))
* Remove keybinding overrides that shadow textual built-ins ([#2084](https://github.com/langchain-ai/deepagents/issues/2084)) ([08fc5d0](https://github.com/langchain-ai/deepagents/commit/08fc5d088ca7be041ad680b87236a4f9e9c212cf))
* Resolve conflicting langsmith env var precedence ([#2455](https://github.com/langchain-ai/deepagents/issues/2455)) ([b6997d8](https://github.com/langchain-ai/deepagents/commit/b6997d830c8c2822acbaa0ad672bdec873af2b9d))
* Show server startup error instead of generic agent message ([#2397](https://github.com/langchain-ai/deepagents/issues/2397)) ([a3e1e93](https://github.com/langchain-ai/deepagents/commit/a3e1e9365178ad14d7233ae577bb87d714daff5c))
* Certain slash commands should not require server connection ([#1974](https://github.com/langchain-ai/deepagents/issues/1974)) ([32bd814](https://github.com/langchain-ai/deepagents/commit/32bd814b98a079778dad092a2e877d8eba22b0fb))
* Sort MCP tools deterministically for prompt-cache stability ([#2497](https://github.com/langchain-ai/deepagents/issues/2497)) ([39b43cf](https://github.com/langchain-ai/deepagents/commit/39b43cf8d0f44827f3b2faf32843c7db436539e8))
* Squash loading widget timer leak ([#2396](https://github.com/langchain-ai/deepagents/issues/2396)) ([01d3d86](https://github.com/langchain-ai/deepagents/commit/01d3d864098c4e96527082b2bd1755ac6591b098))
* Support pre-release versions in update checker ([#2164](https://github.com/langchain-ai/deepagents/issues/2164)) ([e18e9dc](https://github.com/langchain-ai/deepagents/commit/e18e9dcd0e6edc72c0a4a5b76ae752c4bc539752))
* Surface clear error for missing sandbox provider deps ([#1999](https://github.com/langchain-ai/deepagents/issues/1999)) ([939f56a](https://github.com/langchain-ai/deepagents/commit/939f56a19244714fc53158a2fd162e3098b3d56c))
* Use relative paths in langgraph config for Windows compat ([#2244](https://github.com/langchain-ai/deepagents/issues/2244)) ([d10dfbd](https://github.com/langchain-ai/deepagents/commit/d10dfbd7d8ff18653d0d25197c76c8b379f60bf0))
* Warn agent that local filesystem is inaccessible in sandbox mode ([#2274](https://github.com/langchain-ai/deepagents/issues/2274)) ([a3b61e5](https://github.com/langchain-ai/deepagents/commit/a3b61e5d57fcf240fec4afa7734a5c864bdcd886))
* Wire `enable_ask_user` flag to remove tool in non-interactive mode ([#2105](https://github.com/langchain-ai/deepagents/issues/2105)) ([2399747](https://github.com/langchain-ai/deepagents/commit/23997478c006728fc3923e1e8853626794319c32))

### Performance Improvements

* Sub 250ms first paint ([#2027](https://github.com/langchain-ai/deepagents/issues/2027)) ([e42e05c](https://github.com/langchain-ai/deepagents/commit/e42e05c12229ece450746d6f943d2483477518a5))
* Defer `/model` selector data loading off event loop ([#2259](https://github.com/langchain-ai/deepagents/issues/2259)) ([a32ce7f](https://github.com/langchain-ai/deepagents/commit/a32ce7ff6b2112cf48170d2279a1953eded61987))
* Defer heavy imports from startup path ([#2022](https://github.com/langchain-ai/deepagents/issues/2022)) ([b7f5a99](https://github.com/langchain-ai/deepagents/commit/b7f5a99ecda2e3b795764e74bc1e9cbc405b7e18))
* Defer pydantic and adapter imports out of startup hot path ([#2269](https://github.com/langchain-ai/deepagents/issues/2269)) ([0a410b4](https://github.com/langchain-ai/deepagents/commit/0a410b4aba729a71c78e7fd470c904d95fea171a))
* Prewarm markdown stack and cache skill body render ([#2236](https://github.com/langchain-ai/deepagents/issues/2236)) ([0a3ba47](https://github.com/langchain-ai/deepagents/commit/0a3ba47615bfdaba1a51dbff1c75c392dd87579e))
* Reduce health poll interval for local langgraph dev server ([#2283](https://github.com/langchain-ai/deepagents/issues/2283)) ([7f5c3de](https://github.com/langchain-ai/deepagents/commit/7f5c3de9cf5501abc24fbaf50152c4c625b04f1b))

## [0.0.34](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.33...deepagents-cli==0.0.34) (2026-03-17)

### Features

* External editor support via `ctrl+x` and `/editor` ([#1861](https://github.com/langchain-ai/deepagents/issues/1861)) ([bf5d088](https://github.com/langchain-ai/deepagents/commit/bf5d088d4b3cee6c7e44c3abe3736f9972897896))
* Defer HITL approval menu while user is typing ([#1833](https://github.com/langchain-ai/deepagents/issues/1833)) ([1d1572e](https://github.com/langchain-ai/deepagents/commit/1d1572e40cc9f87b97832cbe2b9152c281f8ec92))

### Bug Fixes

* Resolve config-defined providers during runtime model swaps ([#1941](https://github.com/langchain-ai/deepagents/issues/1941)) ([aebc660](https://github.com/langchain-ai/deepagents/commit/aebc660321895909f6b6eb71e72a99ca7754bcf1))

## [0.0.33](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.32...deepagents-cli==0.0.33) (2026-03-16)

### Highlights

* **Client-server architecture:** The CLI can now connect to a remote agent backend via `langgraph dev`, decoupling the UI from the runtime and unlocking server-side execution workflows.
* **Expanded model ecosystem:** Added LiteLLM and Baseten as built-in providers. The `/model` selector now shows live model status, supports an `enabled` flag to hide providers, and auto-discovers models for `class_path` (arbitrary) providers.
* **CLI ergonomics:** New `-y` and `-S` short flags for `--auto-approve` and `--shell-allow-list`. The `ask_user` tool is now enabled by default, and the welcome banner rotates helpful tips.
* **Performance:** Integrated `textual-speedups` for Rust-based layout primitives, switched the messages container to stream layout, and offloaded blocking path ops from the event loop — noticeably snappier on long conversations.
* **Stability:** 15 bug fixes covering markup injection, scrollbar flicker during streaming, reentrant model switching, and autocomplete race conditions.

### Features

* Client-server architecture via `langgraph dev` ([#1759](https://github.com/langchain-ai/deepagents/issues/1759)) ([f5407e6](https://github.com/langchain-ai/deepagents/commit/f5407e6300e6ee51d8c88f1f183dd960b30a5b56))
* Show model status in `/model` selector ([#1820](https://github.com/langchain-ai/deepagents/issues/1820)) ([92ce0cf](https://github.com/langchain-ai/deepagents/commit/92ce0cffde6f49131541b5985bc80058ae0ad46e))
* Enable `ask_user` tool by default ([#1830](https://github.com/langchain-ai/deepagents/issues/1830)) ([ed0c745](https://github.com/langchain-ai/deepagents/commit/ed0c745eef8d8fad40aa39b0dfd4de2ba9988fe5))
* Add `-y` and `-S` short flags for auto-approve and shell-allow-list ([#1919](https://github.com/langchain-ai/deepagents/issues/1919)) ([1036b16](https://github.com/langchain-ai/deepagents/commit/1036b16276d59d8be669d963901c91aeb8cc2236))
* Add `enabled` flag to hide providers from `/model` switcher ([#1897](https://github.com/langchain-ai/deepagents/issues/1897)) ([72a216c](https://github.com/langchain-ai/deepagents/commit/72a216c88f661f9e53eaf92aedce40fac7b76d3c))
* Add `litellm` optional dep ([#1818](https://github.com/langchain-ai/deepagents/issues/1818)) ([defa21b](https://github.com/langchain-ai/deepagents/commit/defa21bc6eea596bc67e70372e022d5b78049c0d))
* Add `sessions` as hidden keyword alias for `/threads` ([#1823](https://github.com/langchain-ai/deepagents/issues/1823)) ([ffa98cc](https://github.com/langchain-ai/deepagents/commit/ffa98ccf9ebe12eadb7f0e95f278dd9bd8eca240))
* Add Baseten as a built-in model provider/optional dep ([#1819](https://github.com/langchain-ai/deepagents/issues/1819)) ([e05ee66](https://github.com/langchain-ai/deepagents/commit/e05ee66b0d6c5cb996208d554686fc128f1094a2))
* Add rotating tips to welcome banner ([#1898](https://github.com/langchain-ai/deepagents/issues/1898)) ([d882ca8](https://github.com/langchain-ai/deepagents/commit/d882ca81e2c9a11768a07824fd5b5ce8579bdcd7))
* Add sandbox type to trace metadata ([#1845](https://github.com/langchain-ai/deepagents/issues/1845)) ([59ef941](https://github.com/langchain-ai/deepagents/commit/59ef94143fc0adabb5f70f308527d98aa1511e44))
* Show versions in non-interactive header ([#1881](https://github.com/langchain-ai/deepagents/issues/1881)) ([adacc3f](https://github.com/langchain-ai/deepagents/commit/adacc3fd0a0f040373b8ef39032978019b5c8e5f))
* Show editable install source path in help and banner ([#1916](https://github.com/langchain-ai/deepagents/issues/1916)) ([4ce1cee](https://github.com/langchain-ai/deepagents/commit/4ce1ceebdd2e4aa6db008061519d3df1e422c2db))

### Bug Fixes

* Replace per-chunk `scroll_end` with anchor to fix scrollbar flicker ([#1891](https://github.com/langchain-ai/deepagents/issues/1891)) ([a9be236](https://github.com/langchain-ai/deepagents/commit/a9be2368509f9f2c66537014ae2138253cf0dc39))
* Auto-discover models for `class_path` providers ([#1816](https://github.com/langchain-ai/deepagents/issues/1816)) ([177fe0f](https://github.com/langchain-ai/deepagents/commit/177fe0f663926d6879aa2177b7988d45cd1e4055))
* Correct model selector footer showing wrong profile after search ([#1805](https://github.com/langchain-ai/deepagents/issues/1805)) ([2f1d52f](https://github.com/langchain-ai/deepagents/commit/2f1d52f4ad65add210d6f840b1b78e62eba37195))
* Escape dynamic strings in rich markup to prevent markup injection ([#1888](https://github.com/langchain-ai/deepagents/issues/1888)) ([d349d10](https://github.com/langchain-ai/deepagents/commit/d349d1061647325fdb4ea6322254b24555abf751))
* Forward `DAYTONA_API_URL` to avoid deprecated `server_url` access ([#1844](https://github.com/langchain-ai/deepagents/issues/1844)) ([7d19ca8](https://github.com/langchain-ai/deepagents/commit/7d19ca8e6404a2ea98aac6a9d4cae7a2b529922c))
* Let unknown providers through credential check ([#1815](https://github.com/langchain-ai/deepagents/issues/1815)) ([89d39de](https://github.com/langchain-ai/deepagents/commit/89d39ded171bf300e9b17972f18063e9157f298f))
* Persist `models.recent` on every session start ([#1802](https://github.com/langchain-ai/deepagents/issues/1802)) ([32aa371](https://github.com/langchain-ai/deepagents/commit/32aa371a371208146cc093c4e5eb7a752a74b3c9))
* Prevent reentrant model switching ([#1824](https://github.com/langchain-ai/deepagents/issues/1824)) ([09d16a8](https://github.com/langchain-ai/deepagents/commit/09d16a8151466fd2d82976d44d7b4e957255bcd9))
* Remove double slash in skills path template ([#1808](https://github.com/langchain-ai/deepagents/issues/1808)) ([2bc9620](https://github.com/langchain-ai/deepagents/commit/2bc962075146548f2ef4c8851bd502df9d6a1fa5))
* Show checkmark for `class_path` providers in model selector ([#1899](https://github.com/langchain-ai/deepagents/issues/1899)) ([4adb712](https://github.com/langchain-ai/deepagents/commit/4adb712c8eacbcfbc06801c31bfd74fc0705bed3))
* Sort prefetched threads by user preference on initial render ([#1806](https://github.com/langchain-ai/deepagents/issues/1806)) ([6f71153](https://github.com/langchain-ai/deepagents/commit/6f711532976821a92e3396fe458ec7874b1237ef))
* Use `max-height` for `tool-info-scroll` to shrink-wrap content ([#1835](https://github.com/langchain-ai/deepagents/issues/1835)) ([a4e1908](https://github.com/langchain-ai/deepagents/commit/a4e1908527c3a30a6f967dd1d989739d540ddd2a))
* Use ASCII-safe glyphs in tool status restore path ([#1895](https://github.com/langchain-ai/deepagents/issues/1895)) ([2a9cbc8](https://github.com/langchain-ai/deepagents/commit/2a9cbc8540b2ab77362cbb00764a3533e234891f))
* Use counter to close history-recall autocomplete race ([#1901](https://github.com/langchain-ai/deepagents/issues/1901)) ([bfd08af](https://github.com/langchain-ai/deepagents/commit/bfd08afbc0eca844e565842dff50eddb067e4750))
* Use UUID7 for thread IDs instead of 8-char hex ([#1826](https://github.com/langchain-ai/deepagents/issues/1826)) ([821885b](https://github.com/langchain-ai/deepagents/commit/821885bab8ae2eef873a33fdaa6dc427a473d764))

### Performance Improvements

* Add `textual-speedups` for Rust-based layout primitives ([#1910](https://github.com/langchain-ai/deepagents/issues/1910)) ([45b58a1](https://github.com/langchain-ai/deepagents/commit/45b58a13e14abfb58ef3688cee51a5819d8ca52d))
* Use stream layout for messages container ([#1896](https://github.com/langchain-ai/deepagents/issues/1896)) ([b35401b](https://github.com/langchain-ai/deepagents/commit/b35401b885a7651db548f1826ddb24476ecde5b7))
* Offload blocking path ops from textual event loop ([#1894](https://github.com/langchain-ai/deepagents/issues/1894)) ([d3eedcc](https://github.com/langchain-ai/deepagents/commit/d3eedccc7edd21103ca1e586fc365d721f674099))
* Speed up `/model` selector with cache pre-warming and async saves ([#1813](https://github.com/langchain-ai/deepagents/issues/1813)) ([2aec75c](https://github.com/langchain-ai/deepagents/commit/2aec75c8dca6c0510671cf1288a924c4ca9bce1c))
* Speed up `/threads` modal startup ([#1811](https://github.com/langchain-ai/deepagents/issues/1811)) ([5758df1](https://github.com/langchain-ai/deepagents/commit/5758df1b663cd09726b8cd08d86897351142ed5e))

## [0.0.32](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.31...deepagents-cli==0.0.32) (2026-03-11)

### Features

* Add token breakdown to `/tokens` and simplify `/compact` messages ([#1782](https://github.com/langchain-ai/deepagents/issues/1782)) ([2f37bff](https://github.com/langchain-ai/deepagents/commit/2f37bffa9d7a9ced6945abe4ab6bc3409bfb97b1))
* `--json` flag for machine-readable output ([#1768](https://github.com/langchain-ai/deepagents/issues/1768)) ([6f62496](https://github.com/langchain-ai/deepagents/commit/6f62496bb699dfa6086ee1850b83f38d3b1242fa))

### Bug Fixes

* Work around VS Code 1.110 space key regression ([#1748](https://github.com/langchain-ai/deepagents/issues/1748)) ([f5fe431](https://github.com/langchain-ai/deepagents/commit/f5fe4315143bf5b636cf42fc98cbfe3d99918cfc))

## [0.0.31](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.30...deepagents-cli==0.0.31) (2026-03-09)

### Features

* Opt-in `ask_user` tool for interactive agent questions ([#1377](https://github.com/langchain-ai/deepagents/issues/1377)) ([de7068d](https://github.com/langchain-ai/deepagents/commit/de7068d21fd4b932c6e53f500b0ea3b02a04c0aa))
* Big thread improvements!
  * Rework `/thread` switcher with search, columns, delete, and sort toggle ([#1723](https://github.com/langchain-ai/deepagents/issues/1723)) ([8b21ddb](https://github.com/langchain-ai/deepagents/commit/8b21ddb2ff7f13d6b3ffcbf2fe605bfbadbc3d38))
  * Track and display working directory per thread ([#1735](https://github.com/langchain-ai/deepagents/issues/1735)) ([0e4f25d](https://github.com/langchain-ai/deepagents/commit/0e4f25dfbc3e15653bc3f8a6d32a0a61ead4ba82))
  * Add `-n` short flag for `threads list --limit` ([#1731](https://github.com/langchain-ai/deepagents/issues/1731)) ([8bbace9](https://github.com/langchain-ai/deepagents/commit/8bbace9facd1e33757521e835dcb291accd2fa91))
  * Add sort, branch filter, and verbose flags to threads list ([#1732](https://github.com/langchain-ai/deepagents/issues/1732)) ([11dc8e3](https://github.com/langchain-ai/deepagents/commit/11dc8e3397ef9e9dbe8b15578e9258544ed6b452))
* Tailor system prompt for non-interactive mode ([#1727](https://github.com/langchain-ai/deepagents/issues/1727)) ([871e5cf](https://github.com/langchain-ai/deepagents/commit/871e5cf76b1a7e7cf7175b4415bb8e2206da39ec))
* `/reload` command for in-session config refresh ([#1722](https://github.com/langchain-ai/deepagents/issues/1722)) ([381aee6](https://github.com/langchain-ai/deepagents/commit/381aee6d223fe3d866bedfe3a534916f419a4435))
* Rearrange HITL option order in approval menu ([#1726](https://github.com/langchain-ai/deepagents/issues/1726)) ([0ca6cb2](https://github.com/langchain-ai/deepagents/commit/0ca6cb237b6da538bad2b4bf292942c8db72ec1f))

### Bug Fixes

* Localize newline shortcut labels by platform ([#1721](https://github.com/langchain-ai/deepagents/issues/1721)) ([f35576b](https://github.com/langchain-ai/deepagents/commit/f35576bafac711d6c04f1f9dd40ec97a90e30060))
* Prevent `shift+enter` from sending `backslash+enter` ([#1728](https://github.com/langchain-ai/deepagents/issues/1728)) ([81dceb0](https://github.com/langchain-ai/deepagents/commit/81dceb043097a47702bb5a0227a8f12e9055bd05))
* Write files with langsmith sandbox ([#1714](https://github.com/langchain-ai/deepagents/issues/1714)) ([5933c9e](https://github.com/langchain-ai/deepagents/commit/5933c9e2995c422e43649c61981e086ac1eaf725))

## [0.0.30](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.29...deepagents-cli==0.0.30) (2026-03-07)

### Features

* `--acp` mode to run CLI agent as ACP server ([#1297](https://github.com/langchain-ai/deepagents/issues/1297)) ([c9ba00a](https://github.com/langchain-ai/deepagents/commit/c9ba00a56b7ee5e48b56b13f9f093bb8bf639700))
* Model detail footer + persist `--profile-override` on hot-swap ([#1700](https://github.com/langchain-ai/deepagents/issues/1700)) ([f2c8b54](https://github.com/langchain-ai/deepagents/commit/f2c8b54e9b4c541bf6f91139bfb9b6a2f20c8de0))
* Show message timestamp toast on click ([#1702](https://github.com/langchain-ai/deepagents/issues/1702)) ([4f403ec](https://github.com/langchain-ai/deepagents/commit/4f403ecb3332010062158ec30fd55f349654a533))

### Bug Fixes

* Expire `ctrl+c` quit window when toast disappears ([#1701](https://github.com/langchain-ai/deepagents/issues/1701)) ([38b5ea9](https://github.com/langchain-ai/deepagents/commit/38b5ea9484ab121c9b2919dd74469e82fce19b82))
* Preserve input text when escaping shell/command mode ([#1706](https://github.com/langchain-ai/deepagents/issues/1706)) ([3c00edb](https://github.com/langchain-ai/deepagents/commit/3c00edb93eddf74e87d58526a02be72577ed65b1))
* Right-align token count next to model name in status bar ([#1705](https://github.com/langchain-ai/deepagents/issues/1705)) ([311c919](https://github.com/langchain-ai/deepagents/commit/311c9191cf663540e1b62eb9452abecda5bc7b4f))

## [0.0.29](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.28...deepagents-cli==0.0.29) (2026-03-06)

### Features

* `--model-params` flag on `/model` command ([#1679](https://github.com/langchain-ai/deepagents/issues/1679)) ([9b6433d](https://github.com/langchain-ai/deepagents/commit/9b6433d557e6e8b3d39c10577595b0ef6d741c94))
* `--shell-allow-list all` ([#1695](https://github.com/langchain-ai/deepagents/issues/1695)) ([4aec7b3](https://github.com/langchain-ai/deepagents/commit/4aec7b35caa7723b8bbda189c9ca1d213e0a9a6d))
* Hook dispatch for external tool integration ([#1553](https://github.com/langchain-ai/deepagents/issues/1553)) ([cdb2230](https://github.com/langchain-ai/deepagents/commit/cdb2230f04ce7a2b7ef0837cbbc223dcbf04b78e))
* Detect deceptive unicode in tool args and URLs ([#1694](https://github.com/langchain-ai/deepagents/issues/1694)) ([d4c8544](https://github.com/langchain-ai/deepagents/commit/d4c8544bd6bf3b6df50b99f8a0c7208c20f86bd9))
* MCP tool loading with auto-discovery ([#801](https://github.com/langchain-ai/deepagents/issues/801)) ([df0908e](https://github.com/langchain-ai/deepagents/commit/df0908ebed4e17f0fd904d83e9d4ea38dfc1207d))
  * Surface mcp server/tool info in system prompt ([#1693](https://github.com/langchain-ai/deepagents/issues/1693)) ([068e075](https://github.com/langchain-ai/deepagents/commit/068e075ecd4a7f3e35219ae6b87707bd9dc3f785))

### Bug Fixes

* Anchor `ChatInput` below scrollable area ([#1671](https://github.com/langchain-ai/deepagents/issues/1671)) ([11105d9](https://github.com/langchain-ai/deepagents/commit/11105d93f593d802d5e120c095f16d771c674bef))
  * Remove dead chat-spacer widget and resize handler ([#1686](https://github.com/langchain-ai/deepagents/issues/1686)) ([b6ecec5](https://github.com/langchain-ai/deepagents/commit/b6ecec5bd14677a878c92a1b51e950f61fabf8d3))

## [0.0.28](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.27...deepagents-cli==0.0.28) (2026-03-05)

### Features

* Video support to multimodal inputs ([#1521](https://github.com/langchain-ai/deepagents/issues/1521)) ([f9b49b7](https://github.com/langchain-ai/deepagents/commit/f9b49b7341bd42b5278a03496743e4709689598e))
* NVIDIA API key support and default model ([#1577](https://github.com/langchain-ai/deepagents/issues/1577)) ([9ce2660](https://github.com/langchain-ai/deepagents/commit/9ce2660a67c3497cff18d27131fb7ef49e85b310))
* Fuzzy search for slash command autocomplete ([#1660](https://github.com/langchain-ai/deepagents/issues/1660)) ([5f6e9c0](https://github.com/langchain-ai/deepagents/commit/5f6e9c014e6a99783b3113184cc12f0179a902f0))
* Tab autocomplete in model selector ([#1669](https://github.com/langchain-ai/deepagents/issues/1669)) ([28bd0aa](https://github.com/langchain-ai/deepagents/commit/28bd0aaca737b8bb194ecb9f6612989b9aacec02))

### Bug Fixes

* Backspace at cursor position 0 exits mode even with text ([#1666](https://github.com/langchain-ai/deepagents/issues/1666)) ([dfa4c1f](https://github.com/langchain-ai/deepagents/commit/dfa4c1fedcecf2bb17d8ffef01cf50efe6c80fb0))
* Skip auto-approve toggle when modal screen is open ([#1668](https://github.com/langchain-ai/deepagents/issues/1668)) ([6597f0b](https://github.com/langchain-ai/deepagents/commit/6597f0b8da3c3bd701a42e228660d459cefe3f64))
* Truncate model name in status bar on narrow terminals ([#1665](https://github.com/langchain-ai/deepagents/issues/1665)) ([0e24a04](https://github.com/langchain-ai/deepagents/commit/0e24a04aa9e5894735522ce23295bb27fd2b8190))

## [0.0.27](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.26...deepagents-cli==0.0.27) (2026-03-04)

### Features

* Background PyPI update check ([#1648](https://github.com/langchain-ai/deepagents/issues/1648)) ([2e7a5e7](https://github.com/langchain-ai/deepagents/commit/2e7a5e7d97f64147ab2d000fae833fe681f1d6b2))
* Install script ([#1649](https://github.com/langchain-ai/deepagents/issues/1649)) ([68f6ef9](https://github.com/langchain-ai/deepagents/commit/68f6ef96e7d66b2c98d1371e91e5d25f107b80fe))
* Fuzzy search for model switcher ([#1266](https://github.com/langchain-ai/deepagents/issues/1266)) ([a6bbb18](https://github.com/langchain-ai/deepagents/commit/a6bbb182a2336ba748d93a06b9fcf27966321e20))
* Model usage stats display ([#1587](https://github.com/langchain-ai/deepagents/issues/1587)) ([a1208db](https://github.com/langchain-ai/deepagents/commit/a1208db096761eb54e0fe712a5aa922502575cb6))
* Substring matching in command history navigation ([#1301](https://github.com/langchain-ai/deepagents/issues/1301)) ([e276d5a](https://github.com/langchain-ai/deepagents/commit/e276d5a64bee9394f53ab993b01447023bcd4c7d))

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
* `--profile-override` CLI flag ([#1605](https://github.com/langchain-ai/deepagents/issues/1605)) ([1984099](https://github.com/langchain-ai/deepagents/commit/1984099ae9ac4b0c13dc08722abb9d56055da7b7))
* Model profile overrides in config ([#1603](https://github.com/langchain-ai/deepagents/issues/1603)) ([d3d6899](https://github.com/langchain-ai/deepagents/commit/d3d6899209b7cf97447da0eee642b3f55261ffbc))
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
* Per-command `timeout` override to `execute()` ([#1154](https://github.com/langchain-ai/deepagents/issues/1154)) ([49277d4](https://github.com/langchain-ai/deepagents/commit/49277d45a026c86b5bf176142dcb1dfc2c7643ae))

### Bug Fixes

* Escape `Rich` markup in shell command display ([#1413](https://github.com/langchain-ai/deepagents/issues/1413)) ([c330290](https://github.com/langchain-ai/deepagents/commit/c33029032a1e2072dab2d06e93953f2acaa6d400))
* Load root-level `AGENTS.md` into agent system prompt ([#1445](https://github.com/langchain-ai/deepagents/issues/1445)) ([047fa2c](https://github.com/langchain-ai/deepagents/commit/047fa2cadfb9f005410c21a6e1e3b3d59eadda7d))
* Prevent crash when quitting with queued messages ([#1421](https://github.com/langchain-ai/deepagents/issues/1421)) ([a3c9ae6](https://github.com/langchain-ai/deepagents/commit/a3c9ae681501cd3efca82573a8d20a0dc8c9b338))

## [0.0.23](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.22...deepagents-cli==0.0.23) (2026-02-18)

### Features

* Add drag-and-drop image attachment to chat input ([#1386](https://github.com/langchain-ai/deepagents/issues/1386)) ([cd3d89b](https://github.com/langchain-ai/deepagents/commit/cd3d89b4419b4c164915ff745afff99cb11b55a5))
* Skill deletion command ([#580](https://github.com/langchain-ai/deepagents/issues/580)) ([40a8d86](https://github.com/langchain-ai/deepagents/commit/40a8d866f952e0cf8d856e2fa360de771721b99a))
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
* `/threads` command switcher ([#1262](https://github.com/langchain-ai/deepagents/issues/1262)) ([45bf38d](https://github.com/langchain-ai/deepagents/commit/45bf38d7c5ca7ca05ec58c320494a692e419b632)), closes [#1111](https://github.com/langchain-ai/deepagents/issues/1111)
* Make thread link clickable when switching ([#1296](https://github.com/langchain-ai/deepagents/issues/1296)) ([9409520](https://github.com/langchain-ai/deepagents/commit/9409520d524c576c3b0b9686c96a1749ee9dcbbb)), closes [#1291](https://github.com/langchain-ai/deepagents/issues/1291)
* `/trace` command to open LangSmith thread, link in switcher ([#1291](https://github.com/langchain-ai/deepagents/issues/1291)) ([fbbd45b](https://github.com/langchain-ai/deepagents/commit/fbbd45b51be2cf09726a3cd0adfcb09cb2b1ff46))
* `/changelog`, `/feedback`, `/docs` ([#1261](https://github.com/langchain-ai/deepagents/issues/1261)) ([4561afb](https://github.com/langchain-ai/deepagents/commit/4561afbea17bb11f7fc02ae9f19db15229656280))
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
* Per-command `timeout` override to `execute` tool ([#1158](https://github.com/langchain-ai/deepagents/issues/1158)) ([cb390ef](https://github.com/langchain-ai/deepagents/commit/cb390ef7a89966760f08c5aceb2211220e8653b8))
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

* LangSmith sandbox integration ([#1077](https://github.com/langchain-ai/deepagents/issues/1077)) ([7d17be0](https://github.com/langchain-ai/deepagents/commit/7d17be00b59e586c55517eaca281342e1a6559ff))
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
