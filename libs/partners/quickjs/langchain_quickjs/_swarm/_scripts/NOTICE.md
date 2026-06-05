# Vendored swarm scripts

The `.ts` files in this directory are vendored from the **swarm** skill in
[`langchain-ai/langchain-skills`](https://github.com/langchain-ai/langchain-skills/tree/main/config/skills/swarm)
(`config/skills/swarm/scripts`).

They implement the handle-based table API (`create` / `run` / `rows`) that the
swarm interpreter extension loads as the `swarm` module. At runtime the scripts
read `swarmTask`, `glob`, `readFile`, `writeFile`, and `editFile` off
`globalThis.tools`; the extension registers those as host functions
(see `_swarm_extension.py`).

To update: re-copy from the upstream skill directory and re-run the test suite.
