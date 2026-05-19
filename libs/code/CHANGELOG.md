# Deep Agents Code Changelog

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

`deepagents-code` was forked from `deepagents-cli` at v0.1.0 (2026-05-12).
For history prior to the fork, see [the `deepagents-cli` changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/cli/CHANGELOG.md).
