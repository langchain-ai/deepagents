<!-- markdownlint-disable MD024 -->

# Deep Agents Code Changelog

## [0.1.16](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.15...deepagents-code==0.1.16) (2026-06-13)

### Features

* ChatGPT OAuth sign-in for Codex models ([#3532](https://github.com/langchain-ai/deepagents/issues/3532)) ([202e0bd](https://github.com/langchain-ai/deepagents/commit/202e0bd3e5b8b874a69656815489308d75a77d05))
* Add Vercel Sandbox provider ([#3588](https://github.com/langchain-ai/deepagents/issues/3588)) ([e5e4748](https://github.com/langchain-ai/deepagents/commit/e5e4748cb6c66ddaa9444ab464990c1a5d10854d))

## [0.1.15](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.14...deepagents-code==0.1.15) (2026-06-12)

### Features

* Add Deep Agents client version metadata ([#3934](https://github.com/langchain-ai/deepagents/issues/3934)) ([058cec6](https://github.com/langchain-ai/deepagents/commit/058cec654645dd1a504408d712ab678edc48273d))

## [0.1.14](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.13...deepagents-code==0.1.14) (2026-06-12)

### Features

* Surface editable mode and core deps in `/version` ([#3898](https://github.com/langchain-ai/deepagents/issues/3898)) ([d4f5a12](https://github.com/langchain-ai/deepagents/commit/d4f5a12cc6d73dc19d24d5748242705d79fa65bf))

### Bug Fixes

* Don't move cursor on click that re-focuses terminal ([#3893](https://github.com/langchain-ai/deepagents/issues/3893)) ([b76788b](https://github.com/langchain-ai/deepagents/commit/b76788b7850e910fe2882cb1a62ffff6ff5085b0))
* Label `-r` resume as `"Resuming..."` in the status bar ([#3892](https://github.com/langchain-ai/deepagents/issues/3892)) ([70cd286](https://github.com/langchain-ai/deepagents/commit/70cd28643bc2ca7da5a61a7b7203c6de2266ca4e))
* Scope `@` file completion to current cwd ([#3874](https://github.com/langchain-ai/deepagents/issues/3874)) ([dd237ac](https://github.com/langchain-ai/deepagents/commit/dd237acd21a39e98e37bb9d98dfae09bc44d5457))
* Set dcode agent names in trace metadata ([#3901](https://github.com/langchain-ai/deepagents/issues/3901)) ([ac94dd6](https://github.com/langchain-ai/deepagents/commit/ac94dd6b139211187cad8685c1152f02c51ba086))
* Surface untracked files in `@` completion ([#3872](https://github.com/langchain-ai/deepagents/issues/3872)) ([2bdf6ea](https://github.com/langchain-ai/deepagents/commit/2bdf6ea329c6b4c8be5f0075528e00c7ce2c1e6a))
* Warn on misconfigured subagent files ([#3873](https://github.com/langchain-ai/deepagents/issues/3873)) ([f1614d6](https://github.com/langchain-ai/deepagents/commit/f1614d67827c3d4d3bd01a784f57f83a50ec3410))

### Performance Improvements

* Cache theme colors/charset, fix `O(n^2)` tool-call streaming ([#3881](https://github.com/langchain-ai/deepagents/issues/3881)) ([9d463f5](https://github.com/langchain-ai/deepagents/commit/9d463f52fb3a4d458477982963f3d278d5362b48))

## [0.1.13](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.12...deepagents-code==0.1.13) (2026-06-11)

### Features

* Pluggable third-party sandbox backends ([#3842](https://github.com/langchain-ai/deepagents/issues/3842)) ([2b635a7](https://github.com/langchain-ai/deepagents/commit/2b635a7e6e6b50ca8ce783c2ac96ed8643ae0224))
* Auto-install ripgrep on first run ([#3348](https://github.com/langchain-ai/deepagents/issues/3348)) ([fecf22b](https://github.com/langchain-ai/deepagents/commit/fecf22b0909e79ff7bdf180baf20abf5fdf1f390))
* `config` command and canonical config manifest ([#3763](https://github.com/langchain-ai/deepagents/issues/3763)) ([79899a3](https://github.com/langchain-ai/deepagents/commit/79899a306d01de6217a1dfcc013ae92c808a47a0))
* Confirm modal for `/install --package` ([#3840](https://github.com/langchain-ai/deepagents/issues/3840)) ([3d75026](https://github.com/langchain-ai/deepagents/commit/3d75026e2f241648fae78d9e1de2cbb4985f58ff))
* Copy focused input selection on `Ctrl+C` ([#3841](https://github.com/langchain-ai/deepagents/issues/3841)) ([99f782c](https://github.com/langchain-ai/deepagents/commit/99f782cf08336c200d02a24ae4edaa650af67ed2))
* `[retries]` config ([#3772](https://github.com/langchain-ai/deepagents/issues/3772)) ([9334d91](https://github.com/langchain-ai/deepagents/commit/9334d91ef94997e46d5373daca9c146fa9498763))
* Show connection state in the status bar ([#3710](https://github.com/langchain-ai/deepagents/issues/3710)) ([3e3e8fe](https://github.com/langchain-ai/deepagents/commit/3e3e8feb0e6e1b77be75a7756fbf32e5c9497c28))
* Surface LangSmith tracing projects in `LocalContextMiddleware` ([#3836](https://github.com/langchain-ai/deepagents/issues/3836)) ([676abec](https://github.com/langchain-ai/deepagents/commit/676abecf892ff537fcb1425ba5929cace3c5d503))

### Bug Fixes

* Add debug-log guidance for truncated startup errors ([#3849](https://github.com/langchain-ai/deepagents/issues/3849)) ([cd1ef30](https://github.com/langchain-ai/deepagents/commit/cd1ef303cf5d0e9d746c787eb09a4a89437e965a))
* Drop lock-key events so Caps Lock in iTerm2 doesn't type ([#3855](https://github.com/langchain-ai/deepagents/issues/3855)) ([110f1a7](https://github.com/langchain-ai/deepagents/commit/110f1a7a975743efda12e181cb3afc8404202254))
* Hand pointer over splash tracing project link ([#3858](https://github.com/langchain-ai/deepagents/issues/3858)) ([ea7dae5](https://github.com/langchain-ai/deepagents/commit/ea7dae58a37cfb1d2b96544eb7c941aad331b280))
* Keep terminal-default theme on Esc in `/theme` selector ([#3854](https://github.com/langchain-ai/deepagents/issues/3854)) ([c3bc67b](https://github.com/langchain-ai/deepagents/commit/c3bc67b0cdeda6f4dbcc6360ddd72b455aec4fe7))
* Preserve inherited `PYTHONPATH` for server subprocess ([#3833](https://github.com/langchain-ai/deepagents/issues/3833)) ([4689569](https://github.com/langchain-ai/deepagents/commit/4689569f94138987319cd9cbb45ce66a1f496934))
* Resolve interpreter PTC allowlist against the runtime tool registry ([#3845](https://github.com/langchain-ai/deepagents/issues/3845)) ([c59a27e](https://github.com/langchain-ai/deepagents/commit/c59a27ef2405b8e04c4351ce7ffa53d8d16d519c))
* Treat multi-line key-event pastes as one input ([#3856](https://github.com/langchain-ai/deepagents/issues/3856)) ([6bb15d4](https://github.com/langchain-ai/deepagents/commit/6bb15d4bd97bd16f47504f937c8458d1b53d9cc4))

## [0.1.12](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.11...deepagents-code==0.1.12) (2026-06-10)

### Features

* Add `get_current_thread_id` tool ([#3820](https://github.com/langchain-ai/deepagents/issues/3820)) ([18ce094](https://github.com/langchain-ai/deepagents/commit/18ce094b7036796c7e23ce1338f4da56dc2ef126))
* Guard managed onboarding-name memory block from edits ([#3822](https://github.com/langchain-ai/deepagents/issues/3822)) ([dc0a51f](https://github.com/langchain-ai/deepagents/commit/dc0a51fc81d01b62dc8043148a9fd87099fc9123))
* Offer restart after restart-capable install ([#3821](https://github.com/langchain-ai/deepagents/issues/3821)) ([e356efe](https://github.com/langchain-ai/deepagents/commit/e356efeff1c8bfb894119cebbb7d3c0853d327a2))
* Persist `/threads` directory-scope preference ([#3824](https://github.com/langchain-ai/deepagents/issues/3824)) ([51a29f4](https://github.com/langchain-ai/deepagents/commit/51a29f4a0c28c122acae296a0df03bee20474455))
* Surface gateway key mismatch on `PermissionDeniedError` ([#3813](https://github.com/langchain-ai/deepagents/issues/3813)) ([5bd1ef8](https://github.com/langchain-ai/deepagents/commit/5bd1ef8cb29db781ccbb37755db44ce0ab1d7bd2))

### Bug Fixes

* Surface cached MCP errors as failed tool messages ([#3829](https://github.com/langchain-ai/deepagents/issues/3829)) ([d83b428](https://github.com/langchain-ai/deepagents/commit/d83b428f7f97bc40b9219c02846c2d8b4ccac434))
* Handle MCP tool errors locally ([#3830](https://github.com/langchain-ai/deepagents/issues/3830)) ([97a7052](https://github.com/langchain-ai/deepagents/commit/97a7052b3eef7ec80bc90a4c3b606deeaf13cb9b))
* Make non-incognito `!` shell output visible to the model ([#3825](https://github.com/langchain-ai/deepagents/issues/3825)) ([2407bca](https://github.com/langchain-ai/deepagents/commit/2407bca66bac5d199a923a0e58872c813eb006bc))
* Expose `/restart` slash command ([#3809](https://github.com/langchain-ai/deepagents/issues/3809)) ([64505b8](https://github.com/langchain-ai/deepagents/commit/64505b848868b80213b6da87113cd63890c3d520))
* Agent retries `gh search` with invalid `mergedAt` field ([#3802](https://github.com/langchain-ai/deepagents/issues/3802)) ([0b683a9](https://github.com/langchain-ai/deepagents/commit/0b683a9435b95bce79d10451868eae54c6a3e88c))
* Clarify `/restart` message during server startup ([#3823](https://github.com/langchain-ai/deepagents/issues/3823)) ([49ded9e](https://github.com/langchain-ai/deepagents/commit/49ded9ef448dcd263bee07ac15f4904dd342bfa2))
* Cursor at end when popping queued message to input ([#3832](https://github.com/langchain-ai/deepagents/issues/3832)) ([aa522c9](https://github.com/langchain-ai/deepagents/commit/aa522c94fffb93dc578addb39b4879f779ac5899))
* Coalesce streamed markdown writes to keep input responsive ([#3819](https://github.com/langchain-ai/deepagents/issues/3819)) ([b45eeeb](https://github.com/langchain-ai/deepagents/commit/b45eeebaf7e6ae6e07b8f9ca521525c66fe14f63))
* Don't show `"No threads found"` while threads load ([#3815](https://github.com/langchain-ai/deepagents/issues/3815)) ([68e6426](https://github.com/langchain-ai/deepagents/commit/68e64263f590b3244b689664715f354fb3a389cd))
* Ignore stale cwd warmers in file autocomplete cache ([#3835](https://github.com/langchain-ai/deepagents/issues/3835)) ([9c8c2b2](https://github.com/langchain-ai/deepagents/commit/9c8c2b24853b7b3a7610cbcfc368a6464c0711e0))
* Offer cwd switch on resume ([#3810](https://github.com/langchain-ai/deepagents/issues/3810)) ([d417f53](https://github.com/langchain-ai/deepagents/commit/d417f530aafa8c77523487393dc3039cc822dc09))
* Recover from tool errors instead of aborting run ([#3804](https://github.com/langchain-ai/deepagents/issues/3804)) ([45691c5](https://github.com/langchain-ai/deepagents/commit/45691c556105e171c7210c7c10388dc73202a025))
* Show `/restart` feedback ([#3808](https://github.com/langchain-ai/deepagents/issues/3808)) ([c40bef9](https://github.com/langchain-ai/deepagents/commit/c40bef9b2aaace3909593503b28e8b548bc20014))
* Standardize search previews and restart status ([#3806](https://github.com/langchain-ai/deepagents/issues/3806)) ([71a6d50](https://github.com/langchain-ai/deepagents/commit/71a6d508b21b275c0c7c6bd2e0752adfba496ea9))
* Suppress stale update notices ([#3801](https://github.com/langchain-ai/deepagents/issues/3801)) ([db25241](https://github.com/langchain-ai/deepagents/commit/db252411d8ca945aca86429acb644540ec970bdb))
* Unblock `/remember` in server mode ([#3812](https://github.com/langchain-ai/deepagents/issues/3812)) ([12e00e1](https://github.com/langchain-ai/deepagents/commit/12e00e1a78e2d6be106147a0aa95a518be66849f))
* Unblock TUI input during `/restart` ([#3826](https://github.com/langchain-ai/deepagents/issues/3826)) ([366cd81](https://github.com/langchain-ai/deepagents/commit/366cd81269e2d3891a477848b774cf51af0cefc0))

## [0.1.11](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.10...deepagents-code==0.1.11) (2026-06-07)

### Bug Fixes

* Pause loading timer during approvals ([#3782](https://github.com/langchain-ai/deepagents/issues/3782)) ([f98fb0c](https://github.com/langchain-ai/deepagents/commit/f98fb0c80d08e408a018ea33a8aa7144180f4e93))
* Run auto-update before startup ([#3784](https://github.com/langchain-ai/deepagents/issues/3784)) ([c160ea3](https://github.com/langchain-ai/deepagents/commit/c160ea3eeda1d0ba707bb524cfd0ce087a854e08))
* Skip update prompts for editable installs ([#3781](https://github.com/langchain-ai/deepagents/issues/3781)) ([ae2874e](https://github.com/langchain-ai/deepagents/commit/ae2874e8ece96c04233c1a88a9da1bd7b9ee2bb2))

## [0.1.10](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.9...deepagents-code==0.1.10) (2026-06-05)

### Features

* Pair model API keys with their endpoints ([#3770](https://github.com/langchain-ai/deepagents/issues/3770)) ([cf98030](https://github.com/langchain-ai/deepagents/commit/cf9803072dc0fdc1d5850c9fd2fc4eb6893ed8c9))
* Word-level double-click selection ([#3740](https://github.com/langchain-ai/deepagents/issues/3740)) ([4bb4286](https://github.com/langchain-ai/deepagents/commit/4bb4286a26c9c9bc69a36f2714d9eb0e3e5e4d40))
* Blueprint bootstrapping for Runloop sandboxes ([#3556](https://github.com/langchain-ai/deepagents/issues/3556)) ([13dafd8](https://github.com/langchain-ai/deepagents/commit/13dafd8823c4b530c8e096012733ad74cd501b59))

### Bug Fixes

* Propagate runtime model switches to subagents ([#3771](https://github.com/langchain-ai/deepagents/issues/3771)) ([f577182](https://github.com/langchain-ai/deepagents/commit/f577182c84746e625b65c3c2fda95f8ca21164cf))
* Guard pasted-path probes against `OSError` ([#3745](https://github.com/langchain-ai/deepagents/issues/3745)) ([c9617d3](https://github.com/langchain-ai/deepagents/commit/c9617d3594ab1448c4f3ee2212cdc66cbf138b77))
* Keep startup import prewarm from crashing the TUI mid-upgrade ([#3756](https://github.com/langchain-ai/deepagents/issues/3756)) ([867a2e5](https://github.com/langchain-ai/deepagents/commit/867a2e5c341bd9dfa70b47c7fafc194ac51d7469))
* Move MCP trust state out of user config ([#3742](https://github.com/langchain-ai/deepagents/issues/3742)) ([a97f2fd](https://github.com/langchain-ai/deepagents/commit/a97f2fd394e6b0b943225a0195b0901188bd368c))

## [0.1.9](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.8...deepagents-code==0.1.9) (2026-06-03)

### Bug Fixes

* Add terminal progress preference ([#3728](https://github.com/langchain-ai/deepagents/issues/3728)) ([d9e4976](https://github.com/langchain-ai/deepagents/commit/d9e4976826ae2281e90e06facb5a70a785703029))

## [0.1.8](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.7...deepagents-code==0.1.8) (2026-06-02)

### Features

* List valid extras when `/install` has no argument ([#3695](https://github.com/langchain-ai/deepagents/issues/3695)) ([c7d529c](https://github.com/langchain-ai/deepagents/commit/c7d529ca0fc478dec9060ea04bcc8589f9b1cd3a))
* Add MCP error detail modal ([#3687](https://github.com/langchain-ai/deepagents/issues/3687)) ([4ae4754](https://github.com/langchain-ai/deepagents/commit/4ae475453ce0df6d6b057b7e163396aa27d55143))

### Bug Fixes

* Allow recovery commands when startup fails ([#3706](https://github.com/langchain-ai/deepagents/issues/3706)) ([727d022](https://github.com/langchain-ai/deepagents/commit/727d022cd1526836c3d1de997c1f036e870881f7))
* Preserve extras during install ([#3707](https://github.com/langchain-ai/deepagents/issues/3707)) ([e636ce9](https://github.com/langchain-ai/deepagents/commit/e636ce9e979fd1c30335ec340acdabbd0a5ae79e))
* Normalize empty file list tool output ([#3697](https://github.com/langchain-ai/deepagents/issues/3697)) ([b67aead](https://github.com/langchain-ai/deepagents/commit/b67aead2b86e04aaee8f2dbfba7b263e3e23597d))
* Point MCP re-enable guidance at `Ctrl+R` ([#3688](https://github.com/langchain-ai/deepagents/issues/3688)) ([15ca302](https://github.com/langchain-ai/deepagents/commit/15ca3029f18fa38c1592859febc2a6d0469bff2d))
* Preserve MCP token refresh when metadata discovery fails ([#3685](https://github.com/langchain-ai/deepagents/issues/3685)) ([afafeeb](https://github.com/langchain-ai/deepagents/commit/afafeeb471c4008d4eb4263ec478cf868833fe0b))
* Reduce OAuth login modal noise ([#3693](https://github.com/langchain-ai/deepagents/issues/3693)) ([0e8a780](https://github.com/langchain-ai/deepagents/commit/0e8a780e2dfea2e22ac44545a16279dbe30eb8ee))
* Repair MCP OAuth login redirect and stale client registration ([#3692](https://github.com/langchain-ai/deepagents/issues/3692)) ([f741293](https://github.com/langchain-ai/deepagents/commit/f741293524f7d47eb8a16a3cd4def336c3c3c13f))
* Search all models from `/model` filter ([#3690](https://github.com/langchain-ai/deepagents/issues/3690)) ([5fcb877](https://github.com/langchain-ai/deepagents/commit/5fcb877d094c4504f671bb7aeb52efa7bf3a5b48))
* Serialize `QueuedUserMessage` as user input ([#3708](https://github.com/langchain-ai/deepagents/issues/3708)) ([307d598](https://github.com/langchain-ai/deepagents/commit/307d59826da9b1ddcbcdab8dccef6d18ecf16d10))
* Serialize cold SDK imports ([#3712](https://github.com/langchain-ai/deepagents/issues/3712)) ([fb2adc0](https://github.com/langchain-ai/deepagents/commit/fb2adc0585e978b12646602ba922e252abf41f81))
* Pluralize singular MCP login splash text ([#3689](https://github.com/langchain-ai/deepagents/issues/3689)) ([492b0fc](https://github.com/langchain-ai/deepagents/commit/492b0fc9209e13cd7004a255ef67b31b7e78e95e))

## [0.1.7](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.6...deepagents-code==0.1.7) (2026-05-30)

### Features

* Add toggleable message timestamp footers ([#3662](https://github.com/langchain-ai/deepagents/issues/3662)) ([977e110](https://github.com/langchain-ai/deepagents/commit/977e11006cfbd78fbaba4e7bb2a13acf6b788652))

### Bug Fixes

* Fix zero tool MCP server rendering ([#3649](https://github.com/langchain-ai/deepagents/issues/3649)) ([7e7a567](https://github.com/langchain-ai/deepagents/commit/7e7a567556110ad927a78b45c3a3d4ac37b65e86))
* Centralize debug logging setup to package root ([#3650](https://github.com/langchain-ai/deepagents/issues/3650)) ([5145ed1](https://github.com/langchain-ai/deepagents/commit/5145ed1f8296f41d78c905c2ce899d2742f7dc9b))
* Char-truncate execute tool preview output ([#3627](https://github.com/langchain-ai/deepagents/issues/3627)) ([bb276e2](https://github.com/langchain-ai/deepagents/commit/bb276e2c41177b0dfe6ffd44fd37a293fbfdcb27))
* Handle stale slash-command `Enter` before completion popup renders ([#3647](https://github.com/langchain-ai/deepagents/issues/3647)) ([9a28742](https://github.com/langchain-ai/deepagents/commit/9a287424e86d5d52d0a328388c3fe453b160f597))
* Keep chat input focused when clicking a message ([#3655](https://github.com/langchain-ai/deepagents/issues/3655)) ([daf6571](https://github.com/langchain-ai/deepagents/commit/daf65716d7c999eadb2b7c37e412ec07b2c7aed3))
* Mention `Ctrl+R` in MCP reconnect toast ([#3622](https://github.com/langchain-ai/deepagents/issues/3622)) ([3b4b086](https://github.com/langchain-ai/deepagents/commit/3b4b0867665e58959073e660d85b74c700acaa1e))
* Prevent duplicate-id crash on MCP reconnect and clipboard `NoScreen` ([#3632](https://github.com/langchain-ai/deepagents/issues/3632)) ([6b9a3c0](https://github.com/langchain-ai/deepagents/commit/6b9a3c051586c26c542e958849e952d08a4b5a88))
* Reconstruct message counts for `DeltaChannel` threads from writes table ([#3668](https://github.com/langchain-ai/deepagents/issues/3668)) ([27e1940](https://github.com/langchain-ai/deepagents/commit/27e1940a924abfc999126cf46024003f453ba0c8))
* Render MCP tool errors and drop empty-string optional params ([#3624](https://github.com/langchain-ai/deepagents/issues/3624)) ([fdf3db4](https://github.com/langchain-ai/deepagents/commit/fdf3db464cd9f3de4e84c246547dd2971d26c726))
* Respect line width in tool output previews ([#3646](https://github.com/langchain-ai/deepagents/issues/3646)) ([ba1ad2d](https://github.com/langchain-ai/deepagents/commit/ba1ad2dbabd19b3821490537465a3bcd39c6fed6))
* Restore resumed thread model ([#3651](https://github.com/langchain-ai/deepagents/issues/3651)) ([550a8ab](https://github.com/langchain-ai/deepagents/commit/550a8abf3c595d738162a97f694b5d9527613323))
* Tool spinner, result formatting, and expand-hint fixes ([#3661](https://github.com/langchain-ai/deepagents/issues/3661)) ([54485a3](https://github.com/langchain-ai/deepagents/commit/54485a305854f46a6ce00ae4df51f3301c652a38))

## [0.1.6](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.5...deepagents-code==0.1.6) (2026-05-27)

### Features

* `/install` optional extras ([#3606](https://github.com/langchain-ai/deepagents/issues/3606)) ([7ffaa93](https://github.com/langchain-ai/deepagents/commit/7ffaa93dca6910cd454040d416ff7e0e8bcbcea5))
* Surface deferred MCP reconnect state in `/mcp` ([#3612](https://github.com/langchain-ai/deepagents/issues/3612)) ([d8205c2](https://github.com/langchain-ai/deepagents/commit/d8205c2a39d00e8b6f7f70afe7cc9bb92fee42d8))
* Surface MCP servers awaiting reconnect on splash banner ([#3615](https://github.com/langchain-ai/deepagents/issues/3615)) ([24c5258](https://github.com/langchain-ai/deepagents/commit/24c5258ae6664bc3d3875d8065038716f7c86161))

### Bug Fixes

* Cancel server-side runs before re-trying interrupted-state writes ([#3611](https://github.com/langchain-ai/deepagents/issues/3611)) ([7d46357](https://github.com/langchain-ai/deepagents/commit/7d46357c5446bbc6225f972fd66dc52af8dd0547))
* Editable-install guidance for adding extras ([#3610](https://github.com/langchain-ai/deepagents/issues/3610)) ([771e55f](https://github.com/langchain-ai/deepagents/commit/771e55f171b8087b876ecf767d2f23c86c2a27b9))
* Reuse persisted DCR loopback port across OAuth launches ([#3613](https://github.com/langchain-ai/deepagents/issues/3613)) ([f2f7471](https://github.com/langchain-ai/deepagents/commit/f2f747104945ac79b68e6524d6da886f7cfeb1b0))
* Polish MCP auth success UX ([#3614](https://github.com/langchain-ai/deepagents/issues/3614)) ([d225cb4](https://github.com/langchain-ai/deepagents/commit/d225cb41f41a0a9b2876aff2443eaa0ada24bf29))

## [0.1.5](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.4...deepagents-code==0.1.5) (2026-05-26)

### Bug Fixes

* Join aiosqlite worker thread after close ([#3585](https://github.com/langchain-ai/deepagents/issues/3585)) ([152cec0](https://github.com/langchain-ai/deepagents/commit/152cec04affed3508d4bfdffe7cae522b16d45e6))

## [0.1.4](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.3...deepagents-code==0.1.4) (2026-05-23)

### Features

* Add `--sandbox-snapshot-name` flag ([#3538](https://github.com/langchain-ai/deepagents/issues/3538)) ([b01392e](https://github.com/langchain-ai/deepagents/commit/b01392e7549798434f27f3784fa8c4e734053787))
* `dcode mcp config` and unify `--mcp-config` flag ([#3541](https://github.com/langchain-ai/deepagents/issues/3541)) ([f037b14](https://github.com/langchain-ai/deepagents/commit/f037b140f90a1ba3725b3ef23ab385b3cafe223b))
* Interpreter middleware via `langchain-quickjs` ([#3525](https://github.com/langchain-ai/deepagents/issues/3525)) ([f0ca89c](https://github.com/langchain-ai/deepagents/commit/f0ca89c962c22058194121526638bc2d29f546bd))

### Bug Fixes

* Chat input history navigation and newline scrolling ([#3560](https://github.com/langchain-ai/deepagents/issues/3560)) ([3b51cbd](https://github.com/langchain-ai/deepagents/commit/3b51cbdc8c50d9990477e18a47de6a58e9165bab))
* Distinguish LangSmith failure modes in `/trace` ([#3558](https://github.com/langchain-ai/deepagents/issues/3558)) ([4d158a0](https://github.com/langchain-ai/deepagents/commit/4d158a031aecad8862e02e332f127573003938ec))
* Recover initial session prompts from writes table ([#3535](https://github.com/langchain-ai/deepagents/issues/3535)) ([46b6f3f](https://github.com/langchain-ai/deepagents/commit/46b6f3f3e6ce880cd5ec9cf59622bb745d6ac2eb))
* Install script binary checks reference `dcode` ([#3546](https://github.com/langchain-ai/deepagents/issues/3546)) ([f8977a6](https://github.com/langchain-ai/deepagents/commit/f8977a63769e3f2037619f32596cb9bb7bd1020b))
* Show tool call previews during batched HITL approvals ([#3530](https://github.com/langchain-ai/deepagents/issues/3530)) ([84daa1a](https://github.com/langchain-ai/deepagents/commit/84daa1a2e27963a6d7694dc9278de83782b4a7b7))

## [0.1.3](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.2...deepagents-code==0.1.3) (2026-05-20)

### Features

* In-TUI MCP OAuth login with auto-refresh ([#3469](https://github.com/langchain-ai/deepagents/issues/3469)) ([20e38b8](https://github.com/langchain-ai/deepagents/commit/20e38b8ebd8d9aa4697334432f7832a0a07aea3a))
  * Float unauthorized MCP servers to top and prompt before reconnect ([#3493](https://github.com/langchain-ai/deepagents/issues/3493)) ([2d66580](https://github.com/langchain-ai/deepagents/commit/2d665804131961dfa7e2849248047deec818e4ef))
  * Disable MCP servers from TUI ([#3501](https://github.com/langchain-ai/deepagents/issues/3501)) ([5725de8](https://github.com/langchain-ai/deepagents/commit/5725de857722dbca768a95bc6d97af5b838a11a9))
* `/restart` hidden slash command ([#3514](https://github.com/langchain-ai/deepagents/issues/3514)) ([74bdd36](https://github.com/langchain-ai/deepagents/commit/74bdd3688948d8369cdd978590f5a822eabeb12c))

### Bug Fixes

* Persist `_context_tokens` via `after_model` middleware ([#3496](https://github.com/langchain-ai/deepagents/issues/3496)) ([e2bb284](https://github.com/langchain-ai/deepagents/commit/e2bb284e506e0e49a05169fc6de01bdf42350267))
* Refresh status bar model after recovering from failed startup ([#3511](https://github.com/langchain-ai/deepagents/issues/3511)) ([c96f822](https://github.com/langchain-ai/deepagents/commit/c96f822de187431404d093b852c4a855d3ab8d30))

## [0.1.2](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.1...deepagents-code==0.1.2) (2026-05-19)

### Features

* `/model` toggle for recommended-only list ([#3453](https://github.com/langchain-ai/deepagents/issues/3453)) ([c326b7e](https://github.com/langchain-ai/deepagents/commit/c326b7ec1b9940861175e0466ab4221f03e2bcba))
* `--timeout` flag for non-interactive ([#3351](https://github.com/langchain-ai/deepagents/issues/3351)) ([44e86ab](https://github.com/langchain-ai/deepagents/commit/44e86abbb1870f689dace8b1be6ed430d65e74c1))
* Browser loopback OAuth callback for MCP auth ([#3467](https://github.com/langchain-ai/deepagents/issues/3467)) ([d83aa07](https://github.com/langchain-ai/deepagents/commit/d83aa07c818af35800f81d062a147fa45a47ace7))
* MCP screen metadata ([#3349](https://github.com/langchain-ai/deepagents/issues/3349)) ([ce2f07e](https://github.com/langchain-ai/deepagents/commit/ce2f07e7211f22b3f44a1a232088b89a469a0a99))

### Bug Fixes

* Drop sections from `system_prompt.md` already supplied by SDK middleware ([#3448](https://github.com/langchain-ai/deepagents/issues/3448)) ([9dbf2c2](https://github.com/langchain-ai/deepagents/commit/9dbf2c2f19e941e012d0c93418ef09fb56f30d6a))
* Rename stale usage commands ([#3460](https://github.com/langchain-ai/deepagents/issues/3460)) ([da43b7f](https://github.com/langchain-ai/deepagents/commit/da43b7f9d913e6190ff03c496a269faf08bbf182))
* Suppress interrupt-cleanup state writes from traces ([#3465](https://github.com/langchain-ai/deepagents/issues/3465)) ([319b24e](https://github.com/langchain-ai/deepagents/commit/319b24e6f179eaf56f105a6db683901c82fe95be))

## [0.1.1](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.0...deepagents-code==0.1.1) (2026-05-16)

### Bug Fixes

* Correct LangSmith sandbox working directory ([#3415](https://github.com/langchain-ai/deepagents/issues/3415)) ([b0e8d83](https://github.com/langchain-ai/deepagents/commit/b0e8d83f97a2a698268173a839000c84e8368324))
* Guard `fetch_url` against SSRF ([#3411](https://github.com/langchain-ai/deepagents/issues/3411)) ([54d8521](https://github.com/langchain-ai/deepagents/commit/54d8521976940dfe147ead4b56565360241335be))

## [0.1.0](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.0.1...deepagents-code==0.1.0) (2026-05-12)

Hello world! Ported from `libs/cli`.

---

## Prior Releases

`deepagents-code` was forked from `deepagents-cli` at v0.1.0 (2026-05-12).
For history prior to the fork, see [the `deepagents-cli` changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/cli/CHANGELOG.md).
