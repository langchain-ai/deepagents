# Changelog

## [0.5.2](https://github.com/chebe24/deepagents/compare/deepagents==0.5.1...deepagents==0.5.2) (2026-04-09)


### Features

* **sdk,cli:** add openrouter SDK attribution ([#2205](https://github.com/chebe24/deepagents/issues/2205)) ([2798e51](https://github.com/chebe24/deepagents/commit/2798e51fd90128ffd1a2064383db17c699805395))
* **sdk,cli:** add package version metadata to traces ([#2129](https://github.com/chebe24/deepagents/issues/2129)) ([e4a44b4](https://github.com/chebe24/deepagents/commit/e4a44b467dee1e284ebe741a0e568f2dc613e068))
* **sdk:** `BASE_AGENT_PROMPT` tweaks ([#2541](https://github.com/chebe24/deepagents/issues/2541)) ([812eef1](https://github.com/chebe24/deepagents/commit/812eef185ffda7bc9e6f11425eb5eddc3d3b32e8))
* **sdk:** add `artifacts_root` to `CompositeBackend` and middleware ([#2490](https://github.com/chebe24/deepagents/issues/2490)) ([753ee56](https://github.com/chebe24/deepagents/commit/753ee567f1cc4d544dc2afea7b414564fd07d37d))
* **sdk:** add `subagent_model` param to `create_deep_agent` ([#1369](https://github.com/chebe24/deepagents/issues/1369)) ([40f9620](https://github.com/chebe24/deepagents/commit/40f96208e64aec1feaa0e9b8a78ea53e0af04719))
* **sdk:** add async subagent middleware for remote LangGraph servers ([#1758](https://github.com/chebe24/deepagents/issues/1758)) ([0c5d501](https://github.com/chebe24/deepagents/commit/0c5d501066e7e9cb74737740d9b3c1dfc74751a6))
* **sdk:** Add LangSmith integration metadata to deepagents ([#1837](https://github.com/chebe24/deepagents/issues/1837)) ([0e572ff](https://github.com/chebe24/deepagents/commit/0e572ff4ac296c52ea0ec450a1410671b0287cd4))
* **sdk:** add MemoryAgentBench evaluation suite ([#1807](https://github.com/chebe24/deepagents/issues/1807)) ([6a0763d](https://github.com/chebe24/deepagents/commit/6a0763d301b6d66813f7168abc300243e0a4d1ea))
* **sdk:** add system prompt snapshot test with sync and async subagents ([#2072](https://github.com/chebe24/deepagents/issues/2072)) ([98874f5](https://github.com/chebe24/deepagents/commit/98874f577f80524b47f6031f761870e1547c0645))
* **sdk:** add tau2 airline eval harness ([#1987](https://github.com/chebe24/deepagents/issues/1987)) ([610b19f](https://github.com/chebe24/deepagents/commit/610b19fa79d39e32f28464462235406430f65ceb))
* **sdk:** add tau2 LICENSE ([#2025](https://github.com/chebe24/deepagents/issues/2025)) ([650cedb](https://github.com/chebe24/deepagents/commit/650cedb21d638a77559f7ac034c8524178460f28))
* **sdk:** add timestamp tracking to `AsyncSubAgentJob` ([#2071](https://github.com/chebe24/deepagents/issues/2071)) ([93aa3cf](https://github.com/chebe24/deepagents/commit/93aa3cf16dedeec7d97834d5ea6dcd65cd4c6256))
* **sdk:** evict large HumanMessages ([#2183](https://github.com/chebe24/deepagents/issues/2183)) ([4f72c34](https://github.com/chebe24/deepagents/commit/4f72c34221aea0516738be566de0fffc2c7386ac))
* **sdk:** port `LangSmithSandbox` from CLI to SDK ([#1983](https://github.com/chebe24/deepagents/issues/1983)) ([dfff6e7](https://github.com/chebe24/deepagents/commit/dfff6e7d2f91c269290ab677a9bd7454bc2f4989))
* **sdk:** rename backend methods ([#1907](https://github.com/chebe24/deepagents/issues/1907)) ([7665066](https://github.com/chebe24/deepagents/commit/7665066c093c2583c4cd3bf83515b0906277306f))
* **sdk:** update backend return types for ls, glob, grep ([#1870](https://github.com/chebe24/deepagents/issues/1870)) ([23cf264](https://github.com/chebe24/deepagents/commit/23cf264d0d9784cfce8f8720b125c6ed05a2e2c3))
* **sdk:** update FileData and backend.read return type to support multimodal data ([#1869](https://github.com/chebe24/deepagents/issues/1869)) ([6fb4ede](https://github.com/chebe24/deepagents/commit/6fb4ede894fe5333ea11598468e924442b3b48a7))


### Bug Fixes

* plumb through generics for `create_deep_agent` ([#2383](https://github.com/chebe24/deepagents/issues/2383)) ([6c28e22](https://github.com/chebe24/deepagents/commit/6c28e22cc5768a82b27102adcea4dbb4b5ae6213))
* remove legacy subagents API ([#2443](https://github.com/chebe24/deepagents/issues/2443)) ([56bbfd3](https://github.com/chebe24/deepagents/commit/56bbfd3a7c9fc993f7ea6c54d5c54fca234c2273))
* **sdk,cli:** align error messages and clean up recent refactors ([#2171](https://github.com/chebe24/deepagents/issues/2171)) ([e2db737](https://github.com/chebe24/deepagents/commit/e2db73779926e1effd6d227e14b137b93393461e))
* **sdk:** `last_updated_at` field doesn't account for task status changes ([#2370](https://github.com/chebe24/deepagents/issues/2370)) ([8e1a2d6](https://github.com/chebe24/deepagents/commit/8e1a2d671eda3903d9ebacca4a5e4aa7378646c2))
* **sdk:** accept all langsmith tracing env vars in evals conftest ([#1847](https://github.com/chebe24/deepagents/issues/1847)) ([17c73da](https://github.com/chebe24/deepagents/commit/17c73da0a1808d40a4b82b776cc3d05f7e91b13b))
* **sdk:** add new line after HEREDOC for edit inline sandbox ([#2340](https://github.com/chebe24/deepagents/issues/2340)) ([beb4dbb](https://github.com/chebe24/deepagents/commit/beb4dbb6c75700c6969f2c3bfe8f7bf6046ef3bb))
* **sdk:** bump recursion limit to 10_000 in deepagents and set agent name ([#2194](https://github.com/chebe24/deepagents/issues/2194)) ([7dbc251](https://github.com/chebe24/deepagents/commit/7dbc25180e05261fba4d5ec752d3052fbc826e4c))
* **sdk:** catch `UnicodeDecodeError` in `FilesystemBackend.read` ([#2319](https://github.com/chebe24/deepagents/issues/2319)) ([9783fe4](https://github.com/chebe24/deepagents/commit/9783fe43b71ad0fbe30593a263a27f38c6bf8f55))
* **sdk:** catch PermissionError in FilesystemBackend ripgrep ([#2571](https://github.com/chebe24/deepagents/issues/2571)) ([3d5d673](https://github.com/chebe24/deepagents/commit/3d5d67349c8e88e33af98137db9634742f018cb0))
* **sdk:** fix offloading for state backend ([#2266](https://github.com/chebe24/deepagents/issues/2266)) ([92f5507](https://github.com/chebe24/deepagents/commit/92f55075d978c27a6aa658968a1999754af973fe))
* **sdk:** fix TypeError in async sub-agents ([#2376](https://github.com/chebe24/deepagents/issues/2376)) ([8be4a2e](https://github.com/chebe24/deepagents/commit/8be4a2ee3878a3e15c15d56fd64ba8db248a6328))
* **sdk:** improvements in default sandbox.write and sandbox.read implementations ([#2321](https://github.com/chebe24/deepagents/issues/2321)) ([4a37a46](https://github.com/chebe24/deepagents/commit/4a37a469ffd8a2d3b515c4de72674673f433082b))
* **sdk:** inherit parent `interrupt_on` for subagents ([#2334](https://github.com/chebe24/deepagents/issues/2334)) ([acad9bb](https://github.com/chebe24/deepagents/commit/acad9bb6ceebee7dc74bd75dcffa49faa8697658))
* **sdk:** make created_at and modified_at NotRequired fields on FileData ([#2248](https://github.com/chebe24/deepagents/issues/2248)) ([9862b5a](https://github.com/chebe24/deepagents/commit/9862b5ad0e94fa52fde6971ac9c2a0a1875ee451))
* **sdk:** match example_agent_descriptions closing tag in task tool prompt ([#2462](https://github.com/chebe24/deepagents/issues/2462)) ([c36a41c](https://github.com/chebe24/deepagents/commit/c36a41ccacd46d54806943821ebf16c606a5bc9f))
* **sdk:** normalize CRLF line endings in `FilesystemBackend.edit()` ([#2256](https://github.com/chebe24/deepagents/issues/2256)) ([69bd21e](https://github.com/chebe24/deepagents/commit/69bd21e2b14fa4bfd88d858702aff0e64237d35d))
* **sdk:** overwrite `write` on LangSmithSandbox ([#2097](https://github.com/chebe24/deepagents/issues/2097)) ([c492a76](https://github.com/chebe24/deepagents/commit/c492a7697e390ae760945d06ca4e824c4f47cf92))
* **sdk:** restore deprecated protocol return types ([#2342](https://github.com/chebe24/deepagents/issues/2342)) ([845cdf5](https://github.com/chebe24/deepagents/commit/845cdf52d7f15ab2b260de95f8448c04264c1efc))
* **sdk:** route subagent model resolution to `resolve_model` ([#2208](https://github.com/chebe24/deepagents/issues/2208)) ([6720e4c](https://github.com/chebe24/deepagents/commit/6720e4c520295ac1cb24c19cf3d315d2d05ba257))
* **sdk:** simplify memory agent bench evals ([#1873](https://github.com/chebe24/deepagents/issues/1873)) ([a5d577d](https://github.com/chebe24/deepagents/commit/a5d577d4871290272878137e1ffabc904402c10a))
* **sdk:** strip leading slash from glob patterns before matching ([#1846](https://github.com/chebe24/deepagents/issues/1846)) ([073893c](https://github.com/chebe24/deepagents/commit/073893c3ea82a1e7047eb939cf147091cb4c50c1))
* **sdk:** update middleware order ([#2070](https://github.com/chebe24/deepagents/issues/2070)) ([def526b](https://github.com/chebe24/deepagents/commit/def526b9621ee3c9e10b723acceb532341395554))
* **sdk:** update recursion limit ([#2273](https://github.com/chebe24/deepagents/issues/2273)) ([fd91a30](https://github.com/chebe24/deepagents/commit/fd91a30bf394d4d0ebf8deed9e7835dfc05c9891))
* **sdk:** updates for multimodal ([#2514](https://github.com/chebe24/deepagents/issues/2514)) ([a2edf3e](https://github.com/chebe24/deepagents/commit/a2edf3ed80e17a87027c41a46283387031ebd3e5))
* **sdk:** use file transfer instead of command strings for sandbox write/edit ([#2117](https://github.com/chebe24/deepagents/issues/2117)) ([6c2d559](https://github.com/chebe24/deepagents/commit/6c2d559b574a5a5d9de3adc36d6cf02d6cf93d9d)), closes [#1402](https://github.com/chebe24/deepagents/issues/1402)
* **sdk:** use Sequence instead of list for subagents parameter ([#1905](https://github.com/chebe24/deepagents/issues/1905)) ([b6f015e](https://github.com/chebe24/deepagents/commit/b6f015e6178d1bf7bf470d049dbbf544e3a0ac94))

## [0.5.1](https://github.com/langchain-ai/deepagents/compare/deepagents==0.5.0...deepagents==0.5.1) (2026-04-07)

### Features

* **sdk:** `BASE_AGENT_PROMPT` tweaks ([#2541](https://github.com/langchain-ai/deepagents/issues/2541)) ([812eef1](https://github.com/langchain-ai/deepagents/commit/812eef185ffda7bc9e6f11425eb5eddc3d3b32e8))
* **sdk:** add `artifacts_root` to `CompositeBackend` and middleware ([#2490](https://github.com/langchain-ai/deepagents/issues/2490)) ([753ee56](https://github.com/langchain-ai/deepagents/commit/753ee567f1cc4d544dc2afea7b414564fd07d37d))

### Bug Fixes

* **sdk:** updates for multimodal ([#2514](https://github.com/langchain-ai/deepagents/issues/2514)) ([a2edf3e](https://github.com/langchain-ai/deepagents/commit/a2edf3ed80e17a87027c41a46283387031ebd3e5))

---

# Prior Releases

Versions prior to 0.5.1 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=deepagents) for details on previous versions.
