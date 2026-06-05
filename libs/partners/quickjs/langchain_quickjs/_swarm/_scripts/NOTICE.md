# Vendored swarm scripts

The `.ts` files in this directory are vendored from the **swarm** skill in
[`langchain-ai/langchain-skills`](https://github.com/langchain-ai/langchain-skills/tree/main/config/skills/swarm)
(`config/skills/swarm/scripts`).

They implement the handle-based table API (`create` / `run` / `rows`) that the
swarm interpreter extension loads as the `swarm` module. At runtime the scripts
call the top-level host functions `__swarmTask`, `__swarmGlob`,
`__swarmReadFile`, `__swarmWriteFile`, and `__swarmEditFile`, which the
extension registers (see `../_extension.py`). They were adapted from the
upstream `tools.*` (PTC-namespace) calls to these named host functions.

The sibling `../_prompt.py` is likewise vendored — `SWARM_SYSTEM_PROMPT` is
the body of the upstream skill's `SKILL.md`, with the import specifier
adapted from `@/skills/swarm` to `swarm`.

To update: re-copy from the upstream skill directory, re-apply the
`tools.*` → `__swarm*` adaptation, and re-run the test suite.
