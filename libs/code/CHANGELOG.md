<!-- markdownlint-disable MD024 -->

# Deep Agents Code Changelog

## [0.1.37](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.36...deepagents-code==0.1.37) (2026-07-13)


### Features

* **code:** `/model` Ctrl+N toggle for names vs raw specs ([#4592](https://github.com/langchain-ai/deepagents/issues/4592)) ([518c322](https://github.com/langchain-ai/deepagents/commit/518c322e7dda840b5a740541afc895516a74925f))
* **code:** `/tools` slash command ([#4649](https://github.com/langchain-ai/deepagents/issues/4649)) ([b1600a8](https://github.com/langchain-ai/deepagents/commit/b1600a8da784899d9db5bdf93be07b4c7f53a46c))
* **code:** add `-s` alias for `--skill` ([#4620](https://github.com/langchain-ai/deepagents/issues/4620)) ([c9b7ac2](https://github.com/langchain-ai/deepagents/commit/c9b7ac20752e4270d286b4af3fcf710e7e5bee87))
* **code:** add configurable chat cursor style ([#4687](https://github.com/langchain-ai/deepagents/issues/4687)) ([a22484b](https://github.com/langchain-ai/deepagents/commit/a22484ba2e001fd62fbff0fe916f1b3b93889fc3))
* **code:** add Meta model provider ([#4650](https://github.com/langchain-ai/deepagents/issues/4650)) ([70829c5](https://github.com/langchain-ai/deepagents/commit/70829c5846b6bdde1cee51f0f4929e819ba1026b))
* **code:** expand environment variables in MCP config ([#4681](https://github.com/langchain-ai/deepagents/issues/4681)) ([4f5d7be](https://github.com/langchain-ai/deepagents/commit/4f5d7be1b4073dee346e5a974841721646fa4860))
* **code:** hide diff widget for credential files ([#4593](https://github.com/langchain-ai/deepagents/issues/4593)) ([4c49a24](https://github.com/langchain-ai/deepagents/commit/4c49a24c4c488f45859b45628cfa57719e52f596))
* **code:** in-app Debug Console ([#4564](https://github.com/langchain-ai/deepagents/issues/4564)) ([4f94a30](https://github.com/langchain-ai/deepagents/commit/4f94a30c11efb1819b08647d15b50b9b1e7bf043))
* **code:** offer abort in `/threads` cwd-switch prompt ([#4583](https://github.com/langchain-ai/deepagents/issues/4583)) ([aaeac99](https://github.com/langchain-ai/deepagents/commit/aaeac99d9f17b6a55d663c21a4d063444b6a1870))
* **code:** resume threads in-TUI with `/threads -r [ID]` ([#4609](https://github.com/langchain-ai/deepagents/issues/4609)) ([d442673](https://github.com/langchain-ai/deepagents/commit/d44267358e272553c8ece33d367514430be7c0da))
* **code:** set `prompt_cache_key` for OpenAI models ([#4632](https://github.com/langchain-ai/deepagents/issues/4632)) ([8cf57ac](https://github.com/langchain-ai/deepagents/commit/8cf57aca9f41c3c4ca6d9796dc2cb80d4b22fb6d))
* **code:** show `(debug enabled)` on splash when `DEEPAGENTS_CODE_DEBUG` is set ([#4584](https://github.com/langchain-ai/deepagents/issues/4584)) ([f10b877](https://github.com/langchain-ai/deepagents/commit/f10b877e5e48c3a1e15bd72d2ce06dfb16203231))
* **code:** support Fireworks `/routers` model ids ([#4591](https://github.com/langchain-ai/deepagents/issues/4591)) ([1c08d27](https://github.com/langchain-ai/deepagents/commit/1c08d2705f73f6870610849e00910b94458accf1))


### Bug Fixes

* **code:** avoid repeated startup auto-update stalls ([#4648](https://github.com/langchain-ai/deepagents/issues/4648)) ([12a9c9d](https://github.com/langchain-ai/deepagents/commit/12a9c9d6813a83e5eeff9feed0c4068d1fcc69c4))
* **code:** capture input typed before TUI startup ([#4684](https://github.com/langchain-ai/deepagents/issues/4684)) ([ef9a4a8](https://github.com/langchain-ai/deepagents/commit/ef9a4a8770f4daec70e640d6d2f23dd798d4ede2))
* **code:** detach owned `langgraph dev` server from terminal ([#4642](https://github.com/langchain-ai/deepagents/issues/4642)) ([d1f3afe](https://github.com/langchain-ai/deepagents/commit/d1f3afecdc015b5d1c9fb098d6347c7d7180e417))
* **code:** infer additional model providers ([#4675](https://github.com/langchain-ai/deepagents/issues/4675)) ([4ceed24](https://github.com/langchain-ai/deepagents/commit/4ceed24d3758801ed9df1024602e90488004cef1))
* **code:** infer Fireworks provider from qualified model IDs ([#4594](https://github.com/langchain-ai/deepagents/issues/4594)) ([4d2aa8a](https://github.com/langchain-ai/deepagents/commit/4d2aa8a9684609eee45cd93b0af5149d8dd09eea))
* **code:** preserve `Ctrl+D` deletion in non-empty input ([#4626](https://github.com/langchain-ai/deepagents/issues/4626)) ([306bd89](https://github.com/langchain-ai/deepagents/commit/306bd893f214237b6b4eede7c2ce0eaff66f5527))
* **code:** quit with `Ctrl+D` at end of prompt ([#4678](https://github.com/langchain-ai/deepagents/issues/4678)) ([2f8c111](https://github.com/langchain-ai/deepagents/commit/2f8c11176f5cc9350c3337b2dfcf9f19886d5d02))
* **code:** reap langgraph dev server when startup is cancelled ([#4629](https://github.com/langchain-ai/deepagents/issues/4629)) ([904ff05](https://github.com/langchain-ai/deepagents/commit/904ff05620a576100e05fd65810d38405c4942d4))
* **code:** reject `--auto-approve` in headless mode ([#4617](https://github.com/langchain-ai/deepagents/issues/4617)) ([997be16](https://github.com/langchain-ai/deepagents/commit/997be1643aff6d6900b8b8c95832cdf93be8c0d7))
* **code:** remove misleading agent names from help ([#4671](https://github.com/langchain-ai/deepagents/issues/4671)) ([ac15732](https://github.com/langchain-ai/deepagents/commit/ac1573281530c1564fe07f8a21608e4d4dec2c3b))
* **code:** route explicit `--stdin` + `--skill` to headless path ([#4611](https://github.com/langchain-ai/deepagents/issues/4611)) ([724e24a](https://github.com/langchain-ai/deepagents/commit/724e24a31546e1f31feca3bb2344717550682595))
* **code:** skip Esc prompt-restore once output generation begins ([#4582](https://github.com/langchain-ai/deepagents/issues/4582)) ([14f384f](https://github.com/langchain-ai/deepagents/commit/14f384fc0083c07a7f44f97543b40b74cf93c13f))
* **code:** support plain `exit` quit command ([#4543](https://github.com/langchain-ai/deepagents/issues/4543)) ([e6f10a1](https://github.com/langchain-ai/deepagents/commit/e6f10a149051128677c9f6f37663922054a7ac26))
* **code:** sync `ask_user` active-question highlight with focus ([#4599](https://github.com/langchain-ai/deepagents/issues/4599)) ([e4c29b5](https://github.com/langchain-ai/deepagents/commit/e4c29b5264413d23d8dc70cabae72feb5f3f3dcb))
* **code:** wrap MCP viewer navigation ([#4677](https://github.com/langchain-ai/deepagents/issues/4677)) ([cffc732](https://github.com/langchain-ai/deepagents/commit/cffc73213bfdcec26cb82c413fa7e27db4afc0ca))


### Performance Improvements

* **code:** load MCP servers concurrently during graph build ([#4659](https://github.com/langchain-ai/deepagents/issues/4659)) ([c5345cc](https://github.com/langchain-ai/deepagents/commit/c5345cc04cd810c6238e42dfbfa14497ae1b9020))

## [0.1.36](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.35...deepagents-code==0.1.36) (2026-07-09)

### Features

* Add `GPT-5.6` model family ([#4600](https://github.com/langchain-ai/deepagents/issues/4600)) ([4a806bc](https://github.com/langchain-ai/deepagents/commit/4a806bc703b69334cccd05bcd27d411602e82318))
* Add Grok 4.5 model ([#4596](https://github.com/langchain-ai/deepagents/issues/4596)) ([b0a209d](https://github.com/langchain-ai/deepagents/commit/b0a209da3a8c9f80e2e3d5e199340736c854c567))

### Bug Fixes

* Strip input before Ctrl+C copy-input fallback ([#4590](https://github.com/langchain-ai/deepagents/issues/4590)) ([505d55a](https://github.com/langchain-ai/deepagents/commit/505d55ad83e8569213175911ee6c8c39fbbf340c))

## [0.1.35](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.34...deepagents-code==0.1.35) (2026-07-09)

### Features

* Restore interrupted prompt to input on ESC ([#4544](https://github.com/langchain-ai/deepagents/issues/4544)) ([fccf037](https://github.com/langchain-ai/deepagents/commit/fccf03732140d39749e70e8fc6cb7d73124a1d70))
* Add `[startup].mode` default approval mode ([#4573](https://github.com/langchain-ai/deepagents/issues/4573)) ([7c5bf54](https://github.com/langchain-ai/deepagents/commit/7c5bf542c2f58fa46ecd708c66cd1817a0ebdc41))
* Offer restart after saving Tavily key via `/auth` ([#4560](https://github.com/langchain-ai/deepagents/issues/4560)) ([12df81a](https://github.com/langchain-ai/deepagents/commit/12df81ad00d814c570b4d24a2b06de9b18d51abd))
* Reload env from `/auth` modal via Ctrl+R ([#4566](https://github.com/langchain-ai/deepagents/issues/4566)) ([f07d638](https://github.com/langchain-ai/deepagents/commit/f07d6387b7dc1f0e880f77784905c2aa94143adb))
* Toast on saved `/auth` API key ([#4558](https://github.com/langchain-ai/deepagents/issues/4558)) ([ee3c264](https://github.com/langchain-ai/deepagents/commit/ee3c26415be84e369b829fefb21461c8ec210a5e))

### Bug Fixes

* Harden approval content rendering ([#4581](https://github.com/langchain-ai/deepagents/issues/4581)) ([38446fd](https://github.com/langchain-ai/deepagents/commit/38446fda649b891dd8604e788701eabe739c1dd0))
* Preserve transcript order during virtualization ([#4549](https://github.com/langchain-ai/deepagents/issues/4549)) ([f6ee70c](https://github.com/langchain-ai/deepagents/commit/f6ee70c00ac6ad7b1b180155018b794308a18361))
* Run stdio MCP server pre-flight check off the event loop ([#4434](https://github.com/langchain-ai/deepagents/issues/4434)) ([c9636e2](https://github.com/langchain-ai/deepagents/commit/c9636e22725a5bd32c1f8b9e739b02b4aa7c3dc7))
* Avoid duplicate "criteria ready" message on `/goal` revisions ([#4559](https://github.com/langchain-ai/deepagents/issues/4559)) ([1110497](https://github.com/langchain-ai/deepagents/commit/1110497e17fac6deceb9d1d6e530f9197bfc0d4a))
* Restore welcome banner tips ([#4528](https://github.com/langchain-ai/deepagents/issues/4528)) ([3f1e55e](https://github.com/langchain-ai/deepagents/commit/3f1e55eafbc1fde79cb75f631561cf35d4b8ff4c))
* Clarify managed `rg` install failures ([#4578](https://github.com/langchain-ai/deepagents/issues/4578)) ([434c84a](https://github.com/langchain-ai/deepagents/commit/434c84ae144b0319afe37d6f2dbadfd7f5fc9f70))
* Dedupe update/install log path output ([#4553](https://github.com/langchain-ai/deepagents/issues/4553)) ([1398fee](https://github.com/langchain-ai/deepagents/commit/1398feeca507cab9ec51a66604d46c747f7fcd2f))
* Keep notification center open for API-key entry ([#4568](https://github.com/langchain-ai/deepagents/issues/4568)) ([6e89417](https://github.com/langchain-ai/deepagents/commit/6e8941776cdcc35909da305f3bb39500c0f479cb))
* Queue `/mcp login` sent before the server connects ([#4533](https://github.com/langchain-ai/deepagents/issues/4533)) ([edac82c](https://github.com/langchain-ai/deepagents/commit/edac82c837de8e12b58b97ac25acc31f743d4c5d))
* Serialize MCP OAuth token refreshes to prevent reuse revocation ([#4565](https://github.com/langchain-ai/deepagents/issues/4565)) ([c37100d](https://github.com/langchain-ai/deepagents/commit/c37100d4763e527fce12fa4451794ca25ded7640))

## [0.1.34](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.33...deepagents-code==0.1.34) (2026-07-07)

### Bug Fixes

* **code:** show `(local)` tag under ANSI themes ([#4534](https://github.com/langchain-ai/deepagents/issues/4534)) ([699e439](https://github.com/langchain-ai/deepagents/commit/699e439699723e477a621f95eef87a7d76aac5f5))

## [0.1.33](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.32...deepagents-code==0.1.33) (2026-07-06)

### Features

* In-the-moment trust prompt for symlinked skills ([#4200](https://github.com/langchain-ai/deepagents/issues/4200)) ([a4431e4](https://github.com/langchain-ai/deepagents/commit/a4431e4339348431e91533cb2b177259ab94b083))
* Selective per-server project MCP trust ([#4507](https://github.com/langchain-ai/deepagents/issues/4507)) ([aaa22a9](https://github.com/langchain-ai/deepagents/commit/aaa22a9340cb3d2c8e9ce1a921957d0a9121da20))
* Add `dcode tools list` command ([#4461](https://github.com/langchain-ai/deepagents/issues/4461)) ([1402d0e](https://github.com/langchain-ai/deepagents/commit/1402d0e735a1580503f240457c77db76fcd779d7))

### Bug Fixes

* Strip media placeholders from model-facing message text ([#4462](https://github.com/langchain-ai/deepagents/issues/4462)) ([aa0ae36](https://github.com/langchain-ai/deepagents/commit/aa0ae36b00df6411a9200610a9075ef2dc28b1af))
* Re-apply theme preference on `/reload` ([#4514](https://github.com/langchain-ai/deepagents/issues/4514)) ([5d1c392](https://github.com/langchain-ai/deepagents/commit/5d1c3928f7ca0ea97f28fd4d34ae1a4ee885e888))

## [0.1.32](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.31...deepagents-code==0.1.32) (2026-07-06)

### Features

* Simplify welcome banner to compact box ([#4482](https://github.com/langchain-ai/deepagents/issues/4482)) ([b7f46e9](https://github.com/langchain-ai/deepagents/commit/b7f46e931807f86476c9c2d661dcd4e9623c10a2))
* Add LangSmith base URL to `/auth` ([#4228](https://github.com/langchain-ai/deepagents/issues/4228)) ([88d167f](https://github.com/langchain-ai/deepagents/commit/88d167f9ceec1cfadd7c3feac6f63c0da7c893f1))
* `tool.use` and `tool.result` hook events ([#3954](https://github.com/langchain-ai/deepagents/issues/3954)) ([ba1979d](https://github.com/langchain-ai/deepagents/commit/ba1979d64c3ed5e96ebc896197bc8f97f6f84bac))
* Fall back to folder name for subagents ([#4504](https://github.com/langchain-ai/deepagents/issues/4504)) ([9db3db5](https://github.com/langchain-ai/deepagents/commit/9db3db5f31a93fb8598a5dc5bedf4fe61fe92c94))
* Report tracing gateway in `dcode doctor` ([#4466](https://github.com/langchain-ai/deepagents/issues/4466)) ([a912427](https://github.com/langchain-ai/deepagents/commit/a9124275504bc1af5f76dfdd9a6c558c98eea92d))
* Gate paste auto-collapse behind `display.collapse_pastes` ([#4473](https://github.com/langchain-ai/deepagents/issues/4473)) ([ff5dd56](https://github.com/langchain-ai/deepagents/commit/ff5dd564a3a590e7d2d521b20df495b5e1512e12))

### Bug Fixes

* Keep footer branch visible and ellipsized instead of hiding when narrow ([#4506](https://github.com/langchain-ai/deepagents/issues/4506)) ([ccf30c3](https://github.com/langchain-ai/deepagents/commit/ccf30c342e5ead677aafd196d8f1ddd2bdf95196))
* Remove MCP OAuth success page message shift ([#4463](https://github.com/langchain-ai/deepagents/issues/4463)) ([69bb06c](https://github.com/langchain-ai/deepagents/commit/69bb06c0680fa6d37b917deb6b70927c3eff9c09))
* Prevent `UnicodeEncodeError` crash in non-interactive mode on legacy Windows consoles ([#4478](https://github.com/langchain-ai/deepagents/issues/4478)) ([b1b16cd](https://github.com/langchain-ai/deepagents/commit/b1b16cd114d4ee4f077cf5824a0176cd94ebd851))

### Performance Improvements

* Make `threads list` faster on large session databases ([#4005](https://github.com/langchain-ai/deepagents/issues/4005)) ([85ca01a](https://github.com/langchain-ai/deepagents/commit/85ca01a43b002c2bdb4205f48fac5fa6f66a3276))

## [0.1.31](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.30...deepagents-code==0.1.31) (2026-07-03)

### Features

* Collapse large pastes into compact placeholders ([#4447](https://github.com/langchain-ai/deepagents/issues/4447)) ([9ae927d](https://github.com/langchain-ai/deepagents/commit/9ae927d73ebf30ba50bbb7ec0c1076a0937e5cb7))
* Show model name instead of spec in switcher ([#4460](https://github.com/langchain-ai/deepagents/issues/4460)) ([0059620](https://github.com/langchain-ai/deepagents/commit/005962088ad31b16745c399e4ab2e725e97f7858))
* Offer abort during resume ([#4188](https://github.com/langchain-ai/deepagents/issues/4188)) ([a4c25cd](https://github.com/langchain-ai/deepagents/commit/a4c25cd74de9902fd0ea9440b5dda2d4248bb3d2))
* Persistent banner when installation is stale ([#4459](https://github.com/langchain-ai/deepagents/issues/4459)) ([b74c185](https://github.com/langchain-ai/deepagents/commit/b74c18591a8ce9a49a5fee43b07e602ca9fbf68a))
* Show `"Took <duration>"` after `execute` finishes ([#4301](https://github.com/langchain-ai/deepagents/issues/4301)) ([a5240eb](https://github.com/langchain-ai/deepagents/commit/a5240ebe3654e72fa5beb4070224f52f944768c8))
* Unify `config show`/`list` around effective values ([#4174](https://github.com/langchain-ai/deepagents/issues/4174)) ([ccd9d21](https://github.com/langchain-ai/deepagents/commit/ccd9d216e77855a28b88850beb525f47ce8b686a))

### Bug Fixes

* Make execute command expandable in code TUI transcript ([#4428](https://github.com/langchain-ai/deepagents/issues/4428)) ([d999181](https://github.com/langchain-ai/deepagents/commit/d999181e8424498681f1e442c71380bf6b82b0aa))
* Allow suppressing LangSmith key override warning ([#4436](https://github.com/langchain-ai/deepagents/issues/4436)) ([ddcae5e](https://github.com/langchain-ai/deepagents/commit/ddcae5e0bd9aebc946be1d9fb3f5e35eeb690fa6))
* Ensure unique message widget IDs on history load ([#4454](https://github.com/langchain-ai/deepagents/issues/4454)) ([de2f7d2](https://github.com/langchain-ai/deepagents/commit/de2f7d2835485df905772e1c96e33d2c239d8e22))
* Resolve `/threads` header link on mount, not after load ([#4453](https://github.com/langchain-ai/deepagents/issues/4453)) ([4cba728](https://github.com/langchain-ai/deepagents/commit/4cba72893fc4aae5b429d8c573c1a513d97ddf8a))
* Restore caller's LangSmith API key in shell subprocess env ([#4458](https://github.com/langchain-ai/deepagents/issues/4458)) ([9293b19](https://github.com/langchain-ai/deepagents/commit/9293b190170a872b73d0d94e66479bcc5f7962c7))
* Route Anthropic effort through output config ([#4446](https://github.com/langchain-ai/deepagents/issues/4446)) ([1e8ed81](https://github.com/langchain-ai/deepagents/commit/1e8ed81940a862dde44f704826b61650336367e0))
* Show loading state during model switch ([#4209](https://github.com/langchain-ai/deepagents/issues/4209)) ([7cc4e9f](https://github.com/langchain-ai/deepagents/commit/7cc4e9f9bda5a8ae42bb072b82b634ad8c39ade1))

## [0.1.30](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.29...deepagents-code==0.1.30) (2026-07-02)

### Features

* Add reasoning effort selector ([#4403](https://github.com/langchain-ai/deepagents/issues/4403)) ([6ee0ac4](https://github.com/langchain-ai/deepagents/commit/6ee0ac4cca998f84e157187d565084e2a1912626))
* Collapse completed tool calls into group summaries ([#4373](https://github.com/langchain-ai/deepagents/issues/4373)) ([3735829](https://github.com/langchain-ai/deepagents/commit/3735829a0c14fb1537daf930140c66b647c8222c))
* Auto-detect MCP OAuth from 401 challenge ([#4364](https://github.com/langchain-ai/deepagents/issues/4364)) ([9763ffc](https://github.com/langchain-ai/deepagents/commit/9763ffceab8800526ae9fe10decdb2f32d8e4707))
* Toast when opening a clicked URL ([#4368](https://github.com/langchain-ai/deepagents/issues/4368)) ([434f29e](https://github.com/langchain-ai/deepagents/commit/434f29e5cb3d94988e2a4d80b78452bee4b10812))

### Bug Fixes

* Allow `/remember` with args when no conversation history ([#4418](https://github.com/langchain-ai/deepagents/issues/4418)) ([5df9c83](https://github.com/langchain-ai/deepagents/commit/5df9c83b58f2529561848ae9f1364961e1c80641))
* Persist rubric model after server restart ([#4419](https://github.com/langchain-ai/deepagents/issues/4419)) ([5605a68](https://github.com/langchain-ai/deepagents/commit/5605a681c9fec888d8c6dfbde2decd4692c8cef0))
* Persist resume model state privately ([#4400](https://github.com/langchain-ai/deepagents/issues/4400)) ([bbd0f0d](https://github.com/langchain-ai/deepagents/commit/bbd0f0d1ddc9d22d399a819d7b4363f7997d119e))
* Quiet routine ripgrep installer output ([#4417](https://github.com/langchain-ai/deepagents/issues/4417)) ([a52c18d](https://github.com/langchain-ai/deepagents/commit/a52c18d3ef8fe462d1289dca8798c51450e0368a))
* Honest MCP OAuth callback close message ([#4410](https://github.com/langchain-ai/deepagents/issues/4410)) ([ef637f4](https://github.com/langchain-ai/deepagents/commit/ef637f4a3f38ee155ad7d31e64a5cdd7bfaee62c))

## [0.1.29](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.28...deepagents-code==0.1.29) (2026-07-01)

### Features

* Add rubric iteration controls ([#4405](https://github.com/langchain-ai/deepagents/issues/4405)) ([d6692a7](https://github.com/langchain-ai/deepagents/commit/d6692a7c713490f170b17510d613e02ee37574ab))
* Add Claude Sonnet 5 to model picker ([#4386](https://github.com/langchain-ai/deepagents/issues/4386)) ([64758dd](https://github.com/langchain-ai/deepagents/commit/64758dddf8b46c5b68c345fb1eac1fb7bcbf7f7e))

### Bug Fixes

* Full-width chat messages, hide scrollbar, flush input bg ([#4374](https://github.com/langchain-ai/deepagents/issues/4374)) ([1f8e8dc](https://github.com/langchain-ai/deepagents/commit/1f8e8dc942ac54c4014ac48f9345ee0201be0b1b))
* Resolve editable SDK version metadata ([#4394](https://github.com/langchain-ai/deepagents/issues/4394)) ([3239bf4](https://github.com/langchain-ai/deepagents/commit/3239bf4edf2be3037bb18315463ac6a0c3537e5c))
* Unblock MCP force reconnect modal ([#4396](https://github.com/langchain-ai/deepagents/issues/4396)) ([8b7eab0](https://github.com/langchain-ai/deepagents/commit/8b7eab023db686f577d16f4f9ac872205baf136f))

## [0.1.28](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.27...deepagents-code==0.1.28) (2026-06-29)

### Features

* Add Fireworks session settings ([#4360](https://github.com/langchain-ai/deepagents/issues/4360)) ([90ebb1d](https://github.com/langchain-ai/deepagents/commit/90ebb1d68cb5942f07847f6dce48c14c7caef992))
* `RubricMiddleware` implementation (`/goal`/`--goal` and `/rubric`/`--rubric`)
  * Add rubric-backed goal workflow ([#4365](https://github.com/langchain-ai/deepagents/issues/4365)) ([8fca61d](https://github.com/langchain-ai/deepagents/commit/8fca61dc036888a4414ff5a4536b957c63dab4a5))
  * Non-interactive rubric grading flags ([#4305](https://github.com/langchain-ai/deepagents/issues/4305)) ([1bcb112](https://github.com/langchain-ai/deepagents/commit/1bcb112ee78138eca9cb400c206ce198322aae32))
* Adopt coding-agent-v1 trace metadata ([#4367](https://github.com/langchain-ai/deepagents/issues/4367)) ([cb39747](https://github.com/langchain-ai/deepagents/commit/cb3974748024fcec2e2ee0ccc0a9b23c880733a2))
* Quit shortcut on completed update modal ([#4312](https://github.com/langchain-ai/deepagents/issues/4312)) ([5e6eae9](https://github.com/langchain-ai/deepagents/commit/5e6eae9f094a5167b6bf01f2d84cf06ca5cd80c3))

### Bug Fixes

* Redact LangSmith trace secrets by default ([#4356](https://github.com/langchain-ai/deepagents/issues/4356)) ([5e01fec](https://github.com/langchain-ai/deepagents/commit/5e01fec72d8b179a3b075b07268162d2eaebfe84))
* Honor Baseten base URL env precedence ([#4328](https://github.com/langchain-ai/deepagents/issues/4328)) ([8f20d74](https://github.com/langchain-ai/deepagents/commit/8f20d74892112c3b88aaca63d8c04a355fd6726f))
* Highlight just-installed provider on `/auth` reopen ([#4311](https://github.com/langchain-ai/deepagents/issues/4311)) ([85e47b5](https://github.com/langchain-ai/deepagents/commit/85e47b532b5caf92a52f1a070a95bdb39bee6d3d))
* Clear transient update launch status ([#4355](https://github.com/langchain-ai/deepagents/issues/4355)) ([b870b18](https://github.com/langchain-ai/deepagents/commit/b870b18750f65442009028a85a2ac18f1b06640c))
* Clearer MCP config JSON parse errors ([#4327](https://github.com/langchain-ai/deepagents/issues/4327)) ([9cee602](https://github.com/langchain-ai/deepagents/commit/9cee60274d38d4ad8a2d7a7e9d3ae59aebce5261))
* Record shell output as user context ([#4353](https://github.com/langchain-ai/deepagents/issues/4353)) ([0d504a5](https://github.com/langchain-ai/deepagents/commit/0d504a5df8454e871597932ce1cd3e95b10ab00e))
* Suppress expected MCP reauth logs ([#4359](https://github.com/langchain-ai/deepagents/issues/4359)) ([15ee384](https://github.com/langchain-ai/deepagents/commit/15ee384117ccd955f7481346dd92b7f4b007a60d))

### Performance Improvements

* Background refresh for `@` file completion cache ([#3911](https://github.com/langchain-ai/deepagents/issues/3911)) ([aa22d6b](https://github.com/langchain-ai/deepagents/commit/aa22d6b6d5a49e4d6c3aa2d4932df75d14d65f78))
* Speed up shutdown after `Ctrl+C`/`Ctrl+D` ([#4351](https://github.com/langchain-ai/deepagents/issues/4351)) ([db441ed](https://github.com/langchain-ai/deepagents/commit/db441ed306a8e03fb68f2ae6a2c08523e248976d))

## [0.1.27](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.26...deepagents-code==0.1.27) (2026-06-26)

### Features

* Show last update check time in `dcode doctor` ([#4307](https://github.com/langchain-ai/deepagents/issues/4307)) ([b669f37](https://github.com/langchain-ai/deepagents/commit/b669f37fb08998cfbf983da76f57822215614e7b))

### Bug Fixes

* `dcode doctor` shows `not configured` for unset tracing ([#4318](https://github.com/langchain-ai/deepagents/issues/4318)) ([e323d0c](https://github.com/langchain-ai/deepagents/commit/e323d0c7d91f3b11e03a016c14cf52008dc66b55))
* Drop duplicate token-request `client_id` under Basic auth ([#4323](https://github.com/langchain-ai/deepagents/issues/4323)) ([426dfad](https://github.com/langchain-ai/deepagents/commit/426dfad3ea1e453914cb87efa8fc70fb85a9efcb))
* Include skill invocations in input history ([#4211](https://github.com/langchain-ai/deepagents/issues/4211)) ([7b8d0b2](https://github.com/langchain-ai/deepagents/commit/7b8d0b2ec184e0d9a0cbe858a0d9a7128791969d))
* Offload `create_model` in server graph factory to unblock Codex ([#4324](https://github.com/langchain-ai/deepagents/issues/4324)) ([064ea0c](https://github.com/langchain-ai/deepagents/commit/064ea0c6851353d0b0bc347a0758149765688945))

## [0.1.26](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.25...deepagents-code==0.1.26) (2026-06-26)

### Bug Fixes

* Pin app version when installing extras ([#4313](https://github.com/langchain-ai/deepagents/issues/4313)) ([c20c8e2](https://github.com/langchain-ai/deepagents/commit/c20c8e2fc138f72f1444107d2a936305a591807b))

## [0.1.25](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.24...deepagents-code==0.1.25) (2026-06-26)

### Bug Fixes

* Bind ephemeral port instead of squatting `langgraph dev`'s 2024 ([#4264](https://github.com/langchain-ai/deepagents/issues/4264)) ([11e5359](https://github.com/langchain-ai/deepagents/commit/11e5359851f0291783661b4311ad5e4436a36fb7))
* Block dotenv shell startup hooks ([#4288](https://github.com/langchain-ai/deepagents/issues/4288)) ([686d6f3](https://github.com/langchain-ai/deepagents/commit/686d6f3a1da8b6393efb4c0cf87b3eb35e0cca50))
* Defer server graph construction ([#4300](https://github.com/langchain-ai/deepagents/issues/4300)) ([220dfc0](https://github.com/langchain-ai/deepagents/commit/220dfc0e6b03f9ccb499c6b850c586b9d57cc077))
  * Avoid blocking MCP imports during graph readiness ([#4302](https://github.com/langchain-ai/deepagents/issues/4302)) ([7533ca8](https://github.com/langchain-ai/deepagents/commit/7533ca89f3afc9863ba5e1ecee2d4c5974dea320))
* Gate `delete` file operations ([#4299](https://github.com/langchain-ai/deepagents/issues/4299)) ([92a8681](https://github.com/langchain-ai/deepagents/commit/92a86819adfefbc6ccfd01a861191ba292eca754))
* Handle recursive `fetch_url` conversion ([#4257](https://github.com/langchain-ai/deepagents/issues/4257)) ([f240a40](https://github.com/langchain-ai/deepagents/commit/f240a40dc05d812c38e9926c1d81ba38deb86e3f))
* Report editable SDK runtime version ([#4304](https://github.com/langchain-ai/deepagents/issues/4304)) ([4439e91](https://github.com/langchain-ai/deepagents/commit/4439e912da4bfa6f1e38e14b5a03d2bfe9367d3b))
* Show months instead of "0y ago" for 360-364 day old timestamps ([#4267](https://github.com/langchain-ai/deepagents/issues/4267)) ([820b331](https://github.com/langchain-ai/deepagents/commit/820b331552cb7ce4695ddca3c9b8343a3144392b))
* Surface `/auth`-stored credentials in `config show`/`get` ([#4258](https://github.com/langchain-ai/deepagents/issues/4258)) ([c7c8788](https://github.com/langchain-ai/deepagents/commit/c7c8788ecf0068914298a6055a5f3fd31c36bd44))
* Switch input mode without flashing the mode trigger ([#4243](https://github.com/langchain-ai/deepagents/issues/4243)) ([fc5d9cb](https://github.com/langchain-ai/deepagents/commit/fc5d9cb8fb978ec95f98407692d4809ea1e86577))

## [0.1.24](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.23...deepagents-code==0.1.24) (2026-06-25)

### Features

* Enable `js_eval` by default ([#4245](https://github.com/langchain-ai/deepagents/issues/4245)) ([2e04ff3](https://github.com/langchain-ai/deepagents/commit/2e04ff397e60389c9a19c4a9b528e15602ad8338))
* Dynamic subagents UI ([#4221](https://github.com/langchain-ai/deepagents/issues/4221)) ([10bcba2](https://github.com/langchain-ai/deepagents/commit/10bcba25600e51aba135f170b34aa6315c0f53d6))
* Gate onboarding integrations modal behind opt-in flag ([#4227](https://github.com/langchain-ai/deepagents/issues/4227)) ([6c930c5](https://github.com/langchain-ai/deepagents/commit/6c930c5e4502f572be554acc896c5fb6d061e0e5))

### Bug Fixes

* Eager managed ripgrep install via `dcode tools install` ([#4199](https://github.com/langchain-ai/deepagents/issues/4199)) ([cf536f3](https://github.com/langchain-ai/deepagents/commit/cf536f339958d6726fa41f896c4a3e42df644c9f))
* Interrupt remote runs on chat cancellation ([#4234](https://github.com/langchain-ai/deepagents/issues/4234)) ([37c5fa2](https://github.com/langchain-ai/deepagents/commit/37c5fa23e621616836694bc59c1b0c38def81604))
* Sync approval toggles during active runs ([#4239](https://github.com/langchain-ai/deepagents/issues/4239)) ([4600365](https://github.com/langchain-ai/deepagents/commit/4600365ea0b60c3e9113ecf59b5336be37d03428))
* Clear stale live approval mode keys ([#4242](https://github.com/langchain-ai/deepagents/issues/4242)) ([f11a769](https://github.com/langchain-ai/deepagents/commit/f11a76962c9d536a38e27ac05b32feca364b2424))

## [0.1.23](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.22...deepagents-code==0.1.23) (2026-06-25)

### Features

* Collect Tavily key during onboarding ([#4233](https://github.com/langchain-ai/deepagents/issues/4233)) ([e321cba](https://github.com/langchain-ai/deepagents/commit/e321cba5701313e974f979069186680f1a39587c))
* Surface `/auth` in `/trace not-configured` hint ([#4206](https://github.com/langchain-ai/deepagents/issues/4206)) ([392e410](https://github.com/langchain-ai/deepagents/commit/392e410e48639a56fda5e35b6f7f85a5215cce9c))
* Add Opus 4.8 to recommended models ([#4204](https://github.com/langchain-ai/deepagents/issues/4204)) ([8faf2b0](https://github.com/langchain-ai/deepagents/commit/8faf2b025f8429481a41a3e65544d26614e53589))

### Bug Fixes

* `--reinstall` on `/install` so upgrades rebuild a clean env ([#4196](https://github.com/langchain-ai/deepagents/issues/4196)) ([5e152ac](https://github.com/langchain-ai/deepagents/commit/5e152ac0256d64376f96b293f8844bc8acc993ec))
* Suppress auto-update migration notice on fresh installs ([#4224](https://github.com/langchain-ai/deepagents/issues/4224)) ([eb8ff80](https://github.com/langchain-ai/deepagents/commit/eb8ff809a1b3130a261083c66a193e532095c6db))
* Bake release commit into `dcode doctor` ([#4225](https://github.com/langchain-ai/deepagents/issues/4225)) ([6dc0246](https://github.com/langchain-ai/deepagents/commit/6dc0246f5c6e9170e05191e1c82fad6975ac945c))
* Drop redundant version from "already up to date" message ([#4223](https://github.com/langchain-ai/deepagents/issues/4223)) ([5d080df](https://github.com/langchain-ai/deepagents/commit/5d080df2776d380214bf20ae31266b9628e4e5b0))
* Note subscription plans unusable for Anthropic in `/auth` ([#4207](https://github.com/langchain-ai/deepagents/issues/4207)) ([28cd19d](https://github.com/langchain-ai/deepagents/commit/28cd19db08bacf981ab2c91195638e66072816fe))

## [0.1.22](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.21...deepagents-code==0.1.22) (2026-06-24)

### Features

* Add LangSmith tracing config to `/auth` ([#4193](https://github.com/langchain-ai/deepagents/issues/4193)) ([8e62957](https://github.com/langchain-ai/deepagents/commit/8e6295791093c2ddfec7e6ac57f0df7f12048447))
* Auto-retry credentials-blocked startup after `/auth` ([#4176](https://github.com/langchain-ai/deepagents/issues/4176)) ([d10ba3d](https://github.com/langchain-ai/deepagents/commit/d10ba3dc162e47cc71df56b8e3d0000b6e0ed847))
* Improve onboarding Installed Integrations screen ([#4195](https://github.com/langchain-ai/deepagents/issues/4195)) ([0827bf1](https://github.com/langchain-ai/deepagents/commit/0827bf1b635b825121740ac8946dd6d98c815c7d))

### Bug Fixes

* Exclude managed bin dir from agent picker ([#4190](https://github.com/langchain-ai/deepagents/issues/4190)) ([d869d1e](https://github.com/langchain-ai/deepagents/commit/d869d1e1fa8558510fb48e71f78fb74c69ca4840))
* Warn on `/trace` when thread has no messages ([#4162](https://github.com/langchain-ai/deepagents/issues/4162)) ([c338fc9](https://github.com/langchain-ai/deepagents/commit/c338fc914fcd383beb4b97c13f53dfb2684f8c90))
* Hide "Recent" section during onboarding model selection ([#4198](https://github.com/langchain-ai/deepagents/issues/4198)) ([af882e4](https://github.com/langchain-ai/deepagents/commit/af882e4f6c859878e350d8a37bb75b7bc01bb453))
* Keep auth modal interactive after install-on-select ([#4187](https://github.com/langchain-ai/deepagents/issues/4187)) ([afbc56a](https://github.com/langchain-ai/deepagents/commit/afbc56a1d9dbe8a89015f4c2f90bb87547d5c1d4))
* Generic "missing credentials" in model switcher ([#4182](https://github.com/langchain-ai/deepagents/issues/4182)) ([456ce5c](https://github.com/langchain-ai/deepagents/commit/456ce5c2f5487a754d1fea7046e32c03b7e27a17))
* Hide chat input action buttons in same frame as empty draft ([#4178](https://github.com/langchain-ai/deepagents/issues/4178)) ([f94d417](https://github.com/langchain-ai/deepagents/commit/f94d417d5ad9928db967777fe1d7bc2c37684fb9))
* Preserve uv tool context when installing extras ([#4201](https://github.com/langchain-ai/deepagents/issues/4201)) ([fcc616c](https://github.com/langchain-ai/deepagents/commit/fcc616cf9bc2bbc7b2dc2574ae8649e2f270dc53))
* Unpin uv self-updates and warn when a stale dcode shadows PATH ([#4185](https://github.com/langchain-ai/deepagents/issues/4185)) ([8ca0a18](https://github.com/langchain-ai/deepagents/commit/8ca0a185a15f800267cc057178a348fb5063df1e))

## [0.1.21](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.20...deepagents-code==0.1.21) (2026-06-23)

### Features

* `dcode doctor` diagnostics command ([#4148](https://github.com/langchain-ai/deepagents/issues/4148)) ([8179731](https://github.com/langchain-ai/deepagents/commit/81797312c7d857e7d94d03c9c695cd3c8d88799a))
* Add structured TUI display for `js_eval` ([#4151](https://github.com/langchain-ai/deepagents/issues/4151)) ([91c0dae](https://github.com/langchain-ai/deepagents/commit/91c0dae3fe0253f02a5926fcd3c6f796cd8d11fe))
* Allow dependency updates without requiring release ([#4157](https://github.com/langchain-ai/deepagents/issues/4157)) ([7beb97a](https://github.com/langchain-ai/deepagents/commit/7beb97a2b02e2fd238baf3b6f05d43a4accf3f42))
* Clear chat input via `esc+esc`, add `[ X ]/[ COPY ]` buttons ([#4000](https://github.com/langchain-ai/deepagents/issues/4000)) ([c20546f](https://github.com/langchain-ai/deepagents/commit/c20546feac7876786e6816776d1ccfa5fcd4b2c8))
* Confirm "Launched" after auto-update restart ([#4098](https://github.com/langchain-ai/deepagents/issues/4098)) ([df8db8a](https://github.com/langchain-ai/deepagents/commit/df8db8af6a7cbfc2ab535020b951d73759da73dd))
* Surface tracing in `doctor` and `config show` ([#4163](https://github.com/langchain-ai/deepagents/issues/4163)) ([2bb3e44](https://github.com/langchain-ai/deepagents/commit/2bb3e44243553a5f2954a0f3ec42364563842a87))

### Bug Fixes

* Handle LangSmith project-not-found and default tracing project ([#4153](https://github.com/langchain-ai/deepagents/issues/4153)) ([e303ce9](https://github.com/langchain-ai/deepagents/commit/e303ce986a3595f0cf458e796d857f7c8f5f8b5c))
* Make `/timestamps` toggle instant via per-footer class ([#4095](https://github.com/langchain-ai/deepagents/issues/4095)) ([7ae32b0](https://github.com/langchain-ai/deepagents/commit/7ae32b0a606cc200d4311e11036a65f17e8282b3))
* Refocus `/mcp` filter input after in-place refresh ([#4080](https://github.com/langchain-ai/deepagents/issues/4080)) ([d79cd74](https://github.com/langchain-ai/deepagents/commit/d79cd74cb8a44c300c3bbad712fe77e709f9221a))
* Report same-version dependency updates ([#4146](https://github.com/langchain-ai/deepagents/issues/4146)) ([156e118](https://github.com/langchain-ai/deepagents/commit/156e1185242a19746f8c268904637c73f07b9a10))
* Show "Loading..." in `/threads` agent dropdown while loading ([#4101](https://github.com/langchain-ai/deepagents/issues/4101)) ([c2d949e](https://github.com/langchain-ai/deepagents/commit/c2d949e8765fbbbdb81e5a70125932842358099f))
* Skip tool interrupts once auto-approve is set ([#4092](https://github.com/langchain-ai/deepagents/issues/4092)) ([9e21c34](https://github.com/langchain-ai/deepagents/commit/9e21c346a6eb8ad25b9cc671f24527b07732e2b7))
* Word-delete backspace parity in ask-user text area ([#4079](https://github.com/langchain-ai/deepagents/issues/4079)) ([ed3c499](https://github.com/langchain-ai/deepagents/commit/ed3c499354467bc5e8476e5c7cdf0cd5f8b6aec1))

## [0.1.20](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.19...deepagents-code==0.1.20) (2026-06-17)

### Features

* Manage Tavily web-search API key in `/auth` ([#4062](https://github.com/langchain-ai/deepagents/issues/4062)) ([90cc099](https://github.com/langchain-ai/deepagents/commit/90cc099b0998a9aa55df01e3bc12ad1597c65365))
* Surface uninstalled known providers in `/auth` menu ([#4059](https://github.com/langchain-ai/deepagents/issues/4059)) ([73db271](https://github.com/langchain-ai/deepagents/commit/73db2719bce5ad3c933ffee643df4a62cfd60e17))
* Show provider in usage stats table ([#4049](https://github.com/langchain-ai/deepagents/issues/4049)) ([309b054](https://github.com/langchain-ai/deepagents/commit/309b054fdd040adb5086408a3bcb2550ba73439e))
* Quiet install script's full dependency list ([#4058](https://github.com/langchain-ai/deepagents/issues/4058)) ([de2c9fd](https://github.com/langchain-ai/deepagents/commit/de2c9fd8c7437fdcdfe738599f83808c5423c55c))

### Bug Fixes

* Read `LANGSMITH_PROJECT` for `tracing.langsmith_project` + show default ([#4054](https://github.com/langchain-ai/deepagents/issues/4054)) ([fec1551](https://github.com/langchain-ai/deepagents/commit/fec1551e901a4fe88bd74ee4833a91ef93b1e93f))
* Surface recommended models missing from installed provider profiles ([#4057](https://github.com/langchain-ai/deepagents/issues/4057)) ([56e0d31](https://github.com/langchain-ai/deepagents/commit/56e0d31b6c507d5a1e95c3f2058a8d3b9488a4db))
* Clarify post-update restart semantics ([#4046](https://github.com/langchain-ai/deepagents/issues/4046)) ([6318b81](https://github.com/langchain-ai/deepagents/commit/6318b817399f7dd2f2e6dc07987dca46f1da9886))
* Hide `[SYSTEM]` interrupt notices from `/threads` prompt ([#3988](https://github.com/langchain-ai/deepagents/issues/3988)) ([68fa0cb](https://github.com/langchain-ai/deepagents/commit/68fa0cb4d48ff3f3e552bbf16f8316433c7b6f34))
* Keep install-required `/model` rows dimmed after navigation ([#4048](https://github.com/langchain-ai/deepagents/issues/4048)) ([c499634](https://github.com/langchain-ai/deepagents/commit/c499634c530dcab9b18e3126b4573ff0d36c3efa))
* Rescaffold server workspace on `/restart` when config is missing ([#4050](https://github.com/langchain-ai/deepagents/issues/4050)) ([de51b0e](https://github.com/langchain-ai/deepagents/commit/de51b0e082293f5f795d96d3bf717c82efb9337e))
* Remove transient `Restarting server...` message after restart ([#4047](https://github.com/langchain-ai/deepagents/issues/4047)) ([87cc504](https://github.com/langchain-ai/deepagents/commit/87cc50428bc96ef11dfb11cee4020bff322a7216))
* Suppress model-switch defer toast on bare reconnect ([#4060](https://github.com/langchain-ai/deepagents/issues/4060)) ([60b1052](https://github.com/langchain-ai/deepagents/commit/60b10520d24c87b9c8c9d85b94cffae9e3c67c43))

## [0.1.19](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.18...deepagents-code==0.1.19) (2026-06-17)

### Features

* Add latest frontier models to recommended list ([#4045](https://github.com/langchain-ai/deepagents/issues/4045)) ([1145356](https://github.com/langchain-ai/deepagents/commit/114535637eaee383c64eac5282bc9ac57007c83f))
* Dual-write agent traces to extra LangSmith projects ([#3998](https://github.com/langchain-ai/deepagents/issues/3998)) ([1b94cf3](https://github.com/langchain-ai/deepagents/commit/1b94cf322949e0cafd6f2bacf343fb0044ec5a8c))
* Prompt to install provider when selecting an uninstalled model ([#3981](https://github.com/langchain-ai/deepagents/issues/3981)) ([619207c](https://github.com/langchain-ai/deepagents/commit/619207c8d4d4308a9a3ab9d2d039506feeaf8a67))
* Note minimum OpenAI key permissions in `/auth` ([#4040](https://github.com/langchain-ai/deepagents/issues/4040)) ([14be63d](https://github.com/langchain-ai/deepagents/commit/14be63d476ed8dece75e20fff05016122c258be8))

### Bug Fixes

* Exclude prompt prefix from `UserMessage` selection ([#4002](https://github.com/langchain-ai/deepagents/issues/4002)) ([8ee6ba6](https://github.com/langchain-ai/deepagents/commit/8ee6ba6e64895d3d36a5957627221ed7667bc1d4))

## [0.1.18](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.17...deepagents-code==0.1.18) (2026-06-16)

### Features

* Add provider-specific auth guidance ([#4004](https://github.com/langchain-ai/deepagents/issues/4004)) ([6a39247](https://github.com/langchain-ai/deepagents/commit/6a392471ba30089b1a63135494873d55209bb081))

## [0.1.17](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.16...deepagents-code==0.1.17) (2026-06-16)

### Features

* Make auto-update opt-out by default ([#3994](https://github.com/langchain-ai/deepagents/issues/3994)) ([7ff6e22](https://github.com/langchain-ai/deepagents/commit/7ff6e2224d7ba8462c073333da937c35831f9b34))
* Warn on `--interpreter-tools` without `--interpreter` ([#3976](https://github.com/langchain-ai/deepagents/issues/3976)) ([e68e720](https://github.com/langchain-ai/deepagents/commit/e68e720e2c3caa4e4e7c5e937faf718db7ad2e98))
* Add pre-release update support ([#3960](https://github.com/langchain-ai/deepagents/issues/3960)) ([341a5cc](https://github.com/langchain-ai/deepagents/commit/341a5cc55a30d808dbb366b396923c7801c431d2))
* Add agent filter dropdown to `/threads` picker ([#3677](https://github.com/langchain-ai/deepagents/issues/3677)) ([f8193df](https://github.com/langchain-ai/deepagents/commit/f8193df41d7c47626d94252039217f4079d117cd))
* Non-interactive `deepagents auth` subcommands ([#3910](https://github.com/langchain-ai/deepagents/issues/3910)) ([11a71bb](https://github.com/langchain-ai/deepagents/commit/11a71bbf00c4e6c1f989e70fb3472a65c8d39662))
* Prompt before updating an out-of-date dcode install ([#3995](https://github.com/langchain-ai/deepagents/issues/3995)) ([a5ec6dd](https://github.com/langchain-ai/deepagents/commit/a5ec6dd0fec66ad321c1cc0f5e2f990c9026e03f))

### Bug Fixes

* Guard against misconfigured LangSmith tracing ([#3993](https://github.com/langchain-ai/deepagents/issues/3993)) ([81acc2e](https://github.com/langchain-ai/deepagents/commit/81acc2ecdada55a98e512f96b0caf8cb0f2c6d29))
* Guard restart-prompt import against in-place self-upgrade ([#3980](https://github.com/langchain-ai/deepagents/issues/3980)) ([7af13f6](https://github.com/langchain-ai/deepagents/commit/7af13f6ec2440c2e3eb63bad2ff31e809f772804))
* Skip startup auto-update when already updated in-session ([#3915](https://github.com/langchain-ai/deepagents/issues/3915)) ([3ba3471](https://github.com/langchain-ai/deepagents/commit/3ba347138ac61a474bad58365fdeaa0d00cc1a42))

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
