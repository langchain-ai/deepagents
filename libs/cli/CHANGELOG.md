# Changelog

## [0.0.18](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.17...deepagents-cli==0.0.18) (2026-02-05)


### Features

* **cli:** add langsmith sandbox integration ([#1077](https://github.com/langchain-ai/deepagents/issues/1077)) ([7d17be0](https://github.com/langchain-ai/deepagents/commit/7d17be00b59e586c55517eaca281342e1a6559ff))
* **cli:** resume thread enhancements ([#1065](https://github.com/langchain-ai/deepagents/issues/1065)) ([e6663b0](https://github.com/langchain-ai/deepagents/commit/e6663b0b314582583afd32cb906a6d502cd8f16b))
* **cli:** support  .`agents/skills` dir alias ([#1059](https://github.com/langchain-ai/deepagents/issues/1059)) ([ec1db17](https://github.com/langchain-ai/deepagents/commit/ec1db172c12bc8b8f85bb03138e442353d4b1013))


### Bug Fixes

* **cli:** `Ctrl+E` for tool output toggle ([#1100](https://github.com/langchain-ai/deepagents/issues/1100)) ([9fa9d72](https://github.com/langchain-ai/deepagents/commit/9fa9d727dbf6b8996a61f2f764675dbc2e23c1b6))
* **cli:** consolidate tool output expand/collapse hint placement ([#1102](https://github.com/langchain-ai/deepagents/issues/1102)) ([70db34b](https://github.com/langchain-ai/deepagents/commit/70db34b5f15a7e81ff586dd0adb2bdfd9ac5d4e9))
* **cli:** delete `/exit` ([#1052](https://github.com/langchain-ai/deepagents/issues/1052)) ([8331b77](https://github.com/langchain-ai/deepagents/commit/8331b7790fcf0474e109c3c29f810f4ced0f1745)), closes [#836](https://github.com/langchain-ai/deepagents/issues/836) [#651](https://github.com/langchain-ai/deepagents/issues/651)
* **cli:** installed default prompt not updated following upgrade ([#1082](https://github.com/langchain-ai/deepagents/issues/1082)) ([bffd956](https://github.com/langchain-ai/deepagents/commit/bffd95610730c668406c485ad941835a5307c226))
* **cli:** replace silent exception handling with proper logging ([#708](https://github.com/langchain-ai/deepagents/issues/708)) ([20faf7a](https://github.com/langchain-ai/deepagents/commit/20faf7ac244d97e688f1cc4121d480ed212fe97c))
* **cli:** show full shell command in error output ([#1097](https://github.com/langchain-ai/deepagents/issues/1097)) ([23bb1d8](https://github.com/langchain-ai/deepagents/commit/23bb1d8af85eec8739aea17c3bb3616afb22072a)), closes [#1080](https://github.com/langchain-ai/deepagents/issues/1080)
* **cli:** support `-h`/`--help` flags ([#1106](https://github.com/langchain-ai/deepagents/issues/1106)) ([26bebf5](https://github.com/langchain-ai/deepagents/commit/26bebf592ab56ffdc5eeff55bb7c2e542ef8f706))

## [0.0.17](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.16...deepagents-cli==0.0.17) (2026-02-03)


### Features

* **cli:** add expandable shell command display in HITL approval ([#976](https://github.com/langchain-ai/deepagents/issues/976)) ([fb8a007](https://github.com/langchain-ai/deepagents/commit/fb8a007123d18025beb1a011f2050e1085dcf69b))
* **cli:** model identity ([#770](https://github.com/langchain-ai/deepagents/issues/770)) ([e54a0ee](https://github.com/langchain-ai/deepagents/commit/e54a0ee43c7dfc7fd14c3f43d37cc0ee5e85c5a8))
* **cli:** sandbox provider interface ([#900](https://github.com/langchain-ai/deepagents/issues/900)) ([d431cfd](https://github.com/langchain-ai/deepagents/commit/d431cfd4a56713434e84f4fa1cdf4a160b43db95))

## [0.0.16](https://github.com/langchain-ai/deepagents/compare/deepagents-cli==0.0.15...deepagents-cli==0.0.16) (2026-02-02)

### Features

* **cli:** add configurable timeout to `ShellMiddleware` ([#961](https://github.com/langchain-ai/deepagents/issues/961)) ([bc5e417](https://github.com/langchain-ai/deepagents/commit/bc5e4178a76d795922beab93b87e90ccaf99fba6))
* **cli:** add timeout formatting to enhance `shell` command display ([#987](https://github.com/langchain-ai/deepagents/issues/987)) ([cbbfd49](https://github.com/langchain-ai/deepagents/commit/cbbfd49011c9cf93741a024f6efeceeca830820e))
* **cli:** display thread ID at splash ([#988](https://github.com/langchain-ai/deepagents/issues/988)) ([e61b9e8](https://github.com/langchain-ai/deepagents/commit/e61b9e8e7af417bf5f636180631dbd47a5bb31bb))

### Bug Fixes

* **cli:** improve clipboard copy/paste on macOS ([#960](https://github.com/langchain-ai/deepagents/issues/960)) ([3e1c604](https://github.com/langchain-ai/deepagents/commit/3e1c604474bd98ce1e0ac802df6fb049dd049682))
* **cli:** make `pyperclip` hard dep ([#985](https://github.com/langchain-ai/deepagents/issues/985)) ([0f5d4ad](https://github.com/langchain-ai/deepagents/commit/0f5d4ad9e63d415c9b80cd15fa0f89fc2f91357b)), closes [#960](https://github.com/langchain-ai/deepagents/issues/960)
* **cli:** revert, improve clipboard copy/paste on macOS ([#964](https://github.com/langchain-ai/deepagents/issues/964)) ([4991992](https://github.com/langchain-ai/deepagents/commit/4991992a5a60fd9588e2110b46440337affc80da))
* **cli:** update timeout message for long-running commands in `ShellMiddleware` ([#986](https://github.com/langchain-ai/deepagents/issues/986)) ([dcbe128](https://github.com/langchain-ai/deepagents/commit/dcbe12805a3650e63da89df0774dd7e0181dbaa6))

---

## Pre-release History

Versions prior to 0.0.16 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=deepagents-cli) for details on previous versions.
