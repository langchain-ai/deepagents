<!-- markdownlint-disable MD024 -->

# Deep Agents Changelog

## [0.7.0](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.12...deepagents==0.7.0) (2026-07-02)


### ⚠ BREAKING CHANGES

* **sdk:** Agents can now overwrite existing files with `write_file` instead of receiving a file-exists error. Workflows, prompts, tests, or guardrails that relied on that error to force `edit_file` usage or to protect existing content must add explicit permission/HITL checks or use `edit_file` when preserving existing content matters.
* **sdk:** Agents now see a destructive `delete` filesystem tool when the backend supports it. Existing agents may choose to remove files or directories recursively, and existing filesystem permission rules that allow writes can now authorize deletion unless a narrower deny or interrupt rule blocks the target subtree.

### Features

* **code,sdk:** add rubric iteration controls ([#4405](https://github.com/langchain-ai/deepagents/issues/4405)) ([d6692a7](https://github.com/langchain-ai/deepagents/commit/d6692a7c713490f170b17510d613e02ee37574ab))
* **sdk:** add recursive filesystem delete support ([f2a21ec](https://github.com/langchain-ai/deepagents/commit/f2a21ec1606c2560bd2fd5765409a05c2a44edee))
* **sdk:** allow `write_file` to overwrite existing files ([2506fcc](https://github.com/langchain-ai/deepagents/commit/2506fccfac0e09b656d5e2fefe23162f1c762331))
* **sdk:** allow users to override default middleware by name ([#4251](https://github.com/langchain-ai/deepagents/issues/4251)) ([90c8472](https://github.com/langchain-ai/deepagents/commit/90c8472ae226394a186bf2075459645c2056db7d))
* **sdk:** reduce round trips when offloading large tool results with `sandbox.execute` ([#4230](https://github.com/langchain-ai/deepagents/issues/4230)) ([02f5bd7](https://github.com/langchain-ai/deepagents/commit/02f5bd7bc02f2baaf5a1eefce7d686ba62493647))


### Bug Fixes

* **sdk:** emit terminal rubric iteration status ([#4406](https://github.com/langchain-ai/deepagents/issues/4406)) ([a51c8d2](https://github.com/langchain-ai/deepagents/commit/a51c8d2b1723d143439a1466f5a49a52c442bdfc))
* **sdk:** return sentinel for empty file lists ([#3709](https://github.com/langchain-ai/deepagents/issues/3709)) ([efafd1e](https://github.com/langchain-ai/deepagents/commit/efafd1e070aeae8901ee3123c7fcbf1815a33c4f))

## [0.6.12](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.11...deepagents==0.6.12) (2026-06-25)

This release adds the `deepagents[aws]` extra, which installs `langchain-aws` so Bedrock users get the automatic prompt caching integration added in [#4108](https://github.com/langchain-ai/deepagents/issues/4108).

### Features

* Add Bedrock prompt caching middleware ([#4108](https://github.com/langchain-ai/deepagents/issues/4108)) ([a398382](https://github.com/langchain-ai/deepagents/commit/a398382c85ce518ea1b7f365e49df905cdcd498b))

### Bug Fixes

* Preserve media references in summarization archives ([#3990](https://github.com/langchain-ai/deepagents/issues/3990)) ([2d6fb53](https://github.com/langchain-ai/deepagents/commit/2d6fb53cd8c94680a42fa9d7041509ff78050616))

## [0.6.11](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.10...deepagents==0.6.11) (2026-06-18)

### Bug Fixes

* Route `BaseSandbox async` helpers through aexecute ([#3996](https://github.com/langchain-ai/deepagents/issues/3996)) ([52dcf1a](https://github.com/langchain-ai/deepagents/commit/52dcf1a42cb00dd614b336038e4398472f78859d))

## [0.6.10](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.9...deepagents==0.6.10) (2026-06-13)

### Bug Fixes

* Compare provider in `model_matches_spec` ([#3943](https://github.com/langchain-ai/deepagents/issues/3943)) ([34244b6](https://github.com/langchain-ai/deepagents/commit/34244b6aebde4c237758dacedccdc3e22f3ca8e5))

## [0.6.9](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.8...deepagents==0.6.9) (2026-06-12)

### Features

* Support configurable subagent response format ([#3882](https://github.com/langchain-ai/deepagents/issues/3882)) ([b0e4d7a](https://github.com/langchain-ai/deepagents/commit/b0e4d7aa8d0ad58c65bddef3835e539b3d5b2a99))

### Bug Fixes

* Don't swallow `TypeError` from custom `token_counter` ([#3927](https://github.com/langchain-ai/deepagents/issues/3927)) ([a6ec9d0](https://github.com/langchain-ai/deepagents/commit/a6ec9d0e797a8b8a7d53d7debb197bb25e80be41))
* Handle `Overwrite`-wrapped messages in tool result interception ([#3905](https://github.com/langchain-ai/deepagents/issues/3905)) ([a043c5a](https://github.com/langchain-ai/deepagents/commit/a043c5a6717374c405a3ed740cfdafdd196f5f21))
* Keep private state out of subagent propagation ([#3542](https://github.com/langchain-ai/deepagents/issues/3542)) ([7ff9553](https://github.com/langchain-ai/deepagents/commit/7ff9553fc057a682ba503ebfe9a870adb51ab848))
* Make sync glob timeout bound wall-clock time ([#3866](https://github.com/langchain-ai/deepagents/issues/3866)) ([cba6caf](https://github.com/langchain-ai/deepagents/commit/cba6caf8f708133381506ecc9e217e7e3ca1c7f2))
* Normalize read slices after windowing ([#3888](https://github.com/langchain-ai/deepagents/issues/3888)) ([33d900c](https://github.com/langchain-ai/deepagents/commit/33d900c98b686aa7e7782d0be3019010ced5f03a))
* Stream Python grep fallback ([#3886](https://github.com/langchain-ai/deepagents/issues/3886)) ([3673d95](https://github.com/langchain-ai/deepagents/commit/3673d95d41aa00f03fbfe2083e46afa485800f3a))

### Performance Improvements

* Count tokens once per model call in summarization middleware ([#3877](https://github.com/langchain-ai/deepagents/issues/3877)) ([6558c8c](https://github.com/langchain-ai/deepagents/commit/6558c8ca3d8e10e3309f95c807e740914af3175f))
* Cache filesystem system prompts ([#3889](https://github.com/langchain-ai/deepagents/issues/3889)) ([2f432ba](https://github.com/langchain-ai/deepagents/commit/2f432ba63687d7c8488d5d41d0551d02b4ae00b4))
* Cache grep glob matchers ([#3887](https://github.com/langchain-ai/deepagents/issues/3887)) ([eae3cf1](https://github.com/langchain-ai/deepagents/commit/eae3cf148de3f8c8ee9ffd8af6e5a5e06cce9d6f))

## [0.6.8](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.7...deepagents==0.6.8) (2026-06-03)

### Features

* Add interrupt mode to filesystem permissions ([#3505](https://github.com/langchain-ai/deepagents/issues/3505)) ([a090162](https://github.com/langchain-ai/deepagents/commit/a090162ea9fd950c54fac874b8a165ca770fd2dc))
* Surface subagents via inherited `lc_agent_name` projection ([e0a1ed2](https://github.com/langchain-ai/deepagents/commit/e0a1ed24e6b44c31d0aac3358aeee0d6cb66b2c4))

### Bug Fixes

* Pass through summarization factory prompt knobs ([#3559](https://github.com/langchain-ai/deepagents/issues/3559)) ([a663cad](https://github.com/langchain-ai/deepagents/commit/a663cad8858a78a5d063af0c51bb789bfb1aba2b))
* Align `glob` path default with `grep` ([#3666](https://github.com/langchain-ai/deepagents/issues/3666)) ([ece8e75](https://github.com/langchain-ai/deepagents/commit/ece8e752059f493254753194bcf59befe54b3556))
* Guard empty binary reads with empty-content warning ([#3675](https://github.com/langchain-ai/deepagents/issues/3675)) ([2c2cec8](https://github.com/langchain-ai/deepagents/commit/2c2cec87476f06e32c7b09cedd2c44e0c713f147))
* Surface missing path errors in `FilesystemBackend.ls` ([#3574](https://github.com/langchain-ai/deepagents/issues/3574)) ([4c28760](https://github.com/langchain-ai/deepagents/commit/4c28760abbbe43117f2118793f11ff9d4f71761e))
* Timeout python grep fallback ([#1937](https://github.com/langchain-ai/deepagents/issues/1937)) ([665a18e](https://github.com/langchain-ai/deepagents/commit/665a18e7fe7f291319af6780328384cc658af899))

### Warnings

* We are dropping public exports of `SubagentRunStream`, `AsyncSubagentRunStream`, and `SubagentTransformer`. These are internal classes for the beta `stream_events(version="v3")` streaming API that should not have been publicly exported in the first place.

## [0.6.7](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.6...deepagents==0.6.7) (2026-05-30)

### Bug Fixes

* Propagate goto and graph in Commands returned by tools ([#3391](https://github.com/langchain-ai/deepagents/issues/3391)) ([d92aef6](https://github.com/langchain-ai/deepagents/commit/d92aef68f70a5e5277c43a50581e7895e682c417))
* Handle base64 reads with unknown file extensions ([#3663](https://github.com/langchain-ai/deepagents/issues/3663)) ([9857a08](https://github.com/langchain-ai/deepagents/commit/9857a08b6144b2f001d8f4bf03ac2f254c6b2da4))
* Export `DeepAgentState` ([#3653](https://github.com/langchain-ai/deepagents/issues/3653)) ([14a9047](https://github.com/langchain-ai/deepagents/commit/14a904757c2d9d797945b6ea1bc9529f1f8cf369))

## [0.6.6](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.5...deepagents==0.6.6) (2026-05-28)

### Features

* Allow passing `state_schema` in `create_deep_agent` ([#3642](https://github.com/langchain-ai/deepagents/issues/3642)) ([37839bd](https://github.com/langchain-ai/deepagents/commit/37839bd7d67fba8c11ff0ccaaa8ac92b39609450))

## [0.6.5](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.4...deepagents==0.6.5) (2026-05-28)

### Features

* `RubricMiddleware` for self-evaluated agent iteration ([#3529](https://github.com/langchain-ai/deepagents/issues/3529)) ([5b8d44d](https://github.com/langchain-ai/deepagents/commit/5b8d44d65c6ec43084687019cf00a37d730ac2fa))
* Log when grep falls back from ripgrep ([#3593](https://github.com/langchain-ai/deepagents/issues/3593)) ([379b1ff](https://github.com/langchain-ai/deepagents/commit/379b1ffdc3ccf5da72d9f7264d531be108f0b36d))

### Bug Fixes

* Use `file_path` kwarg in `read_file` examples ([#3630](https://github.com/langchain-ai/deepagents/issues/3630)) ([97946ee](https://github.com/langchain-ai/deepagents/commit/97946ee09eb167c63d8c07f8bb116f40cfc9603f))
* `read_file` pagination skipping lines after wrapping ([#3641](https://github.com/langchain-ai/deepagents/issues/3641)) ([390551d](https://github.com/langchain-ai/deepagents/commit/390551d61d57ce68c2a80ed78c07eaa8e985908b))
* Handle `None` state in messages delta reducer ([#3636](https://github.com/langchain-ai/deepagents/issues/3636)) ([5a6d920](https://github.com/langchain-ai/deepagents/commit/5a6d920d9dec2199cbe743062a5cc1ff8f298567))
* Return grep errors for sandbox exec failures ([#3637](https://github.com/langchain-ai/deepagents/issues/3637)) ([f87d61f](https://github.com/langchain-ai/deepagents/commit/f87d61f01a23fcfb994e88b7f9324d404e66faef))

## [0.6.4](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.3...deepagents==0.6.4) (2026-05-26)

### Bug Fixes

* Grep crashing when files vanish mid-walk ([#3592](https://github.com/langchain-ai/deepagents/issues/3592)) ([0b8301b](https://github.com/langchain-ai/deepagents/commit/0b8301b2067f7dbbc83ebcf05c52e91413260fd1))
* Stable `HumanMessage` IDs across resumed threads ([#3591](https://github.com/langchain-ai/deepagents/issues/3591)) ([82c3194](https://github.com/langchain-ai/deepagents/commit/82c31947f9dc938ffc71e1cea96d162a39aec3a1))

## [0.6.3](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.2...deepagents==0.6.3) (2026-05-20)

### Bug Fixes

* Anchor ripgrep glob to search root ([#3454](https://github.com/langchain-ai/deepagents/issues/3454)) ([e50fa3f](https://github.com/langchain-ai/deepagents/commit/e50fa3f00ab1b1a84bbaed74bf7e89118b7c2d82))
* Assign UUIDs to ID-less messages in _messages_delta_reducer ([#3513](https://github.com/langchain-ai/deepagents/issues/3513)) ([6d959ad](https://github.com/langchain-ai/deepagents/commit/6d959ade30655eae3967c9809994434e0bbd1148))
* Clarify skill source labels in system prompt ([#3464](https://github.com/langchain-ai/deepagents/issues/3464)) ([fc6a24f](https://github.com/langchain-ai/deepagents/commit/fc6a24f18829cf3f36089945226edfa50d52ab9e))
* Strip HTML comments from memory content before system prompt injection ([#3462](https://github.com/langchain-ai/deepagents/issues/3462)) ([bfbb8bc](https://github.com/langchain-ai/deepagents/commit/bfbb8bc5575ebd1ba9aa29430f6d2f86c24b7d3c))

## [0.6.2](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.1...deepagents==0.6.2) (2026-05-18)

### Features

* Add `system_prompt` override slot to memory, skills, and summarization ([#3451](https://github.com/langchain-ai/deepagents/issues/3451)) ([7583f4a](https://github.com/langchain-ai/deepagents/commit/7583f4aff6a6044bc987ee7980d322bb5c791428))

### Bug Fixes

* Patch invalid tool calls ([#3386](https://github.com/langchain-ai/deepagents/issues/3386)) ([c916d1b](https://github.com/langchain-ai/deepagents/commit/c916d1b2e3a81dcd4fb2e595d6b971923c18fa31))
* Align `MemoryMiddleware` prompt with investigate-first agent behavior ([#2461](https://github.com/langchain-ai/deepagents/issues/2461)) ([d53c8f4](https://github.com/langchain-ai/deepagents/commit/d53c8f4f481288a3aa04a5e62362beba6ed7e57a))
* Subagents: update prompt and make fetching of last message more robust ([#3406](https://github.com/langchain-ai/deepagents/issues/3406)) ([4421bec](https://github.com/langchain-ai/deepagents/commit/4421bec94ffbe1f3a3bf44088ebcf8ab8c24a736))
* Summarization: truncate trailing ToolMessages to keep context within `keep` limit ([#3405](https://github.com/langchain-ai/deepagents/issues/3405)) ([bee514f](https://github.com/langchain-ai/deepagents/commit/bee514fd24862b6b22a5993eb8b6cfc69e42dd80))
* Surface OS errors in sandbox ls/read/edit/glob ([#3359](https://github.com/langchain-ai/deepagents/issues/3359)) ([7598bd9](https://github.com/langchain-ai/deepagents/commit/7598bd93f72b609a46da64f7c458c42ac07a0f3a))

## [0.6.1](https://github.com/langchain-ai/deepagents/compare/deepagents==0.6.0...deepagents==0.6.1) (2026-05-12)

### Bug Fixes

* Import profile re-exports from leaf modules ([#3377](https://github.com/langchain-ai/deepagents/issues/3377)) ([ca99391](https://github.com/langchain-ai/deepagents/commit/ca99391668ea1510932f8e9097e8ed3c0caadf73))

## [0.6.0](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.9...deepagents==0.6.0) (2026-05-12)

### Features

* **[`CodeInterpreterMiddleware`](https://pypi.org/project/langchain-quickjs)**: (experimental) `deepagents` now supports code execution and programmatic tool calling through a scoped QuickJS runtime. Install via the optional dependency: `deepagents[quickjs]`.
* Support for `version="v3"` in `stream_events` / `astream_events`. Refer to the [event streaming](https://docs.langchain.com/oss/python/deepagents/event-streaming) guide for details.

## [0.5.9](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.8...deepagents==0.5.9) (2026-05-10)

### Bug Fixes

* Import profile symbols directly from `harness_profiles` ([#3291](https://github.com/langchain-ai/deepagents/issues/3291)) ([503453c](https://github.com/langchain-ai/deepagents/commit/503453c06f7e0545914789a07ddba6ca6b0c8ec5))

## [0.5.8](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.7...deepagents==0.5.8) (2026-05-08)

### Bug Fixes

* Avoid deprecated-use warnings in `CompositeBackend` path mutation ([#3244](https://github.com/langchain-ai/deepagents/issues/3244)) ([64d45f6](https://github.com/langchain-ai/deepagents/commit/64d45f67c86edb4df2ced0e7b82f1a8fd158ec8c))

## [0.5.7](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.6...deepagents==0.5.7) (2026-05-05)

### Bug Fixes

* Auto-added GP subagent inherits parent permissions ([#3131](https://github.com/langchain-ai/deepagents/issues/3131)) ([0d55b3b](https://github.com/langchain-ai/deepagents/commit/0d55b3ba8b974d06b1e0f52893f33e44496bff8b))
* Default OpenRouter routing to ignore Azure upstream ([#3157](https://github.com/langchain-ai/deepagents/issues/3157)) ([01a9113](https://github.com/langchain-ai/deepagents/commit/01a911379d368fab8280cd827c38776800abe7b8))

## [0.5.6](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.5...deepagents==0.5.6) (2026-05-01)

### Bug Fixes

* Add write preflight and native read to langsmith sandbox ([#2695](https://github.com/langchain-ai/deepagents/issues/2695)) ([741221c](https://github.com/langchain-ai/deepagents/commit/741221c9d8b65a535816e318ee24d3c19a4bde80))
* Propagate `CompiledSubAgent` name into `lc_agent_name` metadata ([#3045](https://github.com/langchain-ai/deepagents/issues/3045)) ([f671e6b](https://github.com/langchain-ai/deepagents/commit/f671e6b18aa49700a535f7b48441662b67dafef9))

## [0.5.5](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.4...deepagents==0.5.5) (2026-04-30)

### Bug Fixes

* Harden `FilesystemBackend` against symlink loops ([#3035](https://github.com/langchain-ai/deepagents/issues/3035)) ([abd02f9](https://github.com/langchain-ai/deepagents/commit/abd02f99ef12030bdfe429fdc3ad80a2785bea61))
* Surface EOF-newline mismatch in `edit_file` ([#3031](https://github.com/langchain-ai/deepagents/issues/3031)) ([d30686e](https://github.com/langchain-ai/deepagents/commit/d30686ec82d36a0e9430f7c512c34835aba2c079))
* Prevent stdin hang by passing `DEVNULL` ([#2427](https://github.com/langchain-ai/deepagents/issues/2427)) ([5bf5fae](https://github.com/langchain-ai/deepagents/commit/5bf5fae8d93beba90628f2f71e3e79817a36ac9e))

## [0.5.4](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.3...deepagents==0.5.4) (2026-04-29)

### Highlights

Until this release, `deepagents` shipped a single set of prompts, tools, and middleware tuned to work across *all* model families. Today's headline change is **harness profiles** — a declarative override layer for the parts of the harness that vary per model (system-prompt prefix/suffix, tool inclusion and naming, middleware selection, subagent configuration, skills). `ProviderProfile` shapes how `resolve_model` builds the client; `HarnessProfile` / `HarnessProfileConfig` shape how `create_deep_agent` assembles the agent. Both are keyed by provider (`"openai"`) or full `provider:model` spec (`"openai:gpt-5.4"`), and registrations are additive — your call site doesn't change when you swap models.

We currently ship built-ins for OpenAI and Anthropic out of the box, drawn directly from each vendor's published prompting guides (Codex's `apply_patch` / `shell_command` conventions, Claude's tool-result reflection and active-investigation patterns, etc.). On a curated tau2-bench subset, applying these profiles produced a 10–20 point lift over the default harness (GPT-5.3 Codex: 33% -> 53%; Claude Opus 4.7: 43% -> 53%). Third-party plugins can register via the `deepagents.provider_profiles` and `deepagents.harness_profiles` entry points.

See the [profiles documentation](https://docs.langchain.com/oss/python/deepagents/profiles) for the full field surface, merge semantics, and plugin packaging.

### Features

* Profiles API ([#2892](https://github.com/langchain-ai/deepagents/issues/2892)) ([7365ad1](https://github.com/langchain-ai/deepagents/commit/7365ad1600064eec616c5de970320104189ddf80))
* `ls_agent_type` configurable tag on subagent runs ([#2788](https://github.com/langchain-ai/deepagents/issues/2788)) ([3bcc51a](https://github.com/langchain-ai/deepagents/commit/3bcc51a95da80094cfc8bc4bcaf25dc1e2ad8f44))

### Bug Fixes

* Normalize Windows backslash paths before `PurePosixPath` processing ([#1859](https://github.com/langchain-ai/deepagents/issues/1859)) ([e1c1d50](https://github.com/langchain-ai/deepagents/commit/e1c1d5024729f5205eaa42bf6a9bc1c93a30d043))
* Preserve CRLF line endings in sandbox edit ([#2899](https://github.com/langchain-ai/deepagents/issues/2899)) ([291aebe](https://github.com/langchain-ai/deepagents/commit/291aebe21f8a53604a2bf47daa120761dace2536))
* Support read-your-writes in `StateBackend` ([#2991](https://github.com/langchain-ai/deepagents/issues/2991)) ([0924869](https://github.com/langchain-ai/deepagents/commit/0924869bc3d946577e7c3cbc79a86e4aaf522edd))
* Treat boundary-truncated UTF-8 in `read()` prefix check as text ([#2980](https://github.com/langchain-ai/deepagents/issues/2980)) ([c36ebc7](https://github.com/langchain-ai/deepagents/commit/c36ebc7be5840e9008279992741c67a8377ffc01))
* Use `configurable` directly instead of tracing context for subagent tagging ([#2845](https://github.com/langchain-ai/deepagents/issues/2845)) ([bd6ec6b](https://github.com/langchain-ai/deepagents/commit/bd6ec6bcebcdcc26f6b79e2c55611074b0e01631))

### Performance Improvements

* Add cache breakpoint to `MemoryMiddleware` ([#2713](https://github.com/langchain-ai/deepagents/issues/2713)) ([1699f3a](https://github.com/langchain-ai/deepagents/commit/1699f3aea710985087b16318bb8e6f6e80e02a1b))

## [0.5.3](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.2...deepagents==0.5.3) (2026-04-14)

### Features

* **sdk:** add static structured output to subagent response ([#2437](https://github.com/langchain-ai/deepagents/issues/2437)) ([6e57731](https://github.com/langchain-ai/deepagents/commit/6e57731fc6d908ac1ebe131e782696a4776147e9))
* **sdk:** deprecate `model=None` in `create_deep_agent` ([#2677](https://github.com/langchain-ai/deepagents/issues/2677)) ([149df41](https://github.com/langchain-ai/deepagents/commit/149df415d17f3cf3b7eb0bd1e78460112bfa9b04))

### Bug Fixes

* **sdk:** skill loading should default to 1000 lines ([#2721](https://github.com/langchain-ai/deepagents/issues/2721)) ([badc4d3](https://github.com/langchain-ai/deepagents/commit/badc4d3921ae0ede4305f44f85fa7266df9465e7))

## [0.5.2](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.1...deepagents==0.5.2) (2026-04-10)

### Features

* Permissions system for filesystem access control ([#2633](https://github.com/langchain-ai/deepagents/issues/2633)) ([41dc759](https://github.com/langchain-ai/deepagents/commit/41dc7597deb3fc036f1f850e68edc3c0870f27da))
  * Scope permissions to routes for composite backends with sandbox default ([#2659](https://github.com/langchain-ai/deepagents/issues/2659)) ([6dd6122](https://github.com/langchain-ai/deepagents/commit/6dd612237a7ee707726c4cafc4b691704e4cdb37))
  * Raise `ValueError` for permission paths without leading slash and path traversal ([#2665](https://github.com/langchain-ai/deepagents/issues/2665)) ([723d27d](https://github.com/langchain-ai/deepagents/commit/723d27dcdce03cc9ffaa757c70533f0134a43a44))
* Implement `upload_files` for `StateBackend` ([#2661](https://github.com/langchain-ai/deepagents/issues/2661)) ([5798345](https://github.com/langchain-ai/deepagents/commit/579834513a4ba1a024a52fc4edf918f526eab5f2))

### Bug Fixes

* Catch `PermissionError` in `FilesystemBackend` ripgrep ([#2571](https://github.com/langchain-ai/deepagents/issues/2571)) ([3d5d673](https://github.com/langchain-ai/deepagents/commit/3d5d67349c8e88e33af98137db9634742f018cb0))

## [0.5.1](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.0...deepagents==0.5.1) (2026-04-07)

### Features

* **sdk:** `BASE_AGENT_PROMPT` tweaks ([#2541](https://github.com/langchain-ai/deepagents/issues/2541)) ([812eef1](https://github.com/langchain-ai/deepagents/commit/812eef185ffda7bc9e6f11425eb5eddc3d3b32e8))
* **sdk:** add `artifacts_root` to `CompositeBackend` and middleware ([#2490](https://github.com/langchain-ai/deepagents/issues/2490)) ([753ee56](https://github.com/langchain-ai/deepagents/commit/753ee567f1cc4d544dc2afea7b414564fd07d37d))

### Bug Fixes

* **sdk:** updates for multimodal ([#2514](https://github.com/langchain-ai/deepagents/issues/2514)) ([a2edf3e](https://github.com/langchain-ai/deepagents/commit/a2edf3ed80e17a87027c41a46283387031ebd3e5))

---

## Prior Releases

Versions prior to 0.5.1 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=deepagents) for details on previous versions.
