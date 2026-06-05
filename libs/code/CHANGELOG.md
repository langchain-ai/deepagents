<!-- markdownlint-disable MD024 -->

# Deep Agents Code Changelog

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
