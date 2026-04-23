# Known model-string locations in `langchain-ai/deepagents`

Snapshot of every place an LLM model identifier is hardcoded, keyed by the 6 categories defined in `SKILL.md`. Line numbers reflect `main` at the time of capture and may drift — always re-run `scripts/find_model_refs.py` before editing.

Paths are repo-root-relative.

## Table of contents

- [Category 1 — Hardcoded runtime defaults](#category-1--hardcoded-runtime-defaults) (high risk)
- [Category 2 — Example configs and their READMEs](#category-2--example-configs-and-their-readmes)
- [Category 3 — Docstring examples in SDK/CLI source](#category-3--docstring-examples-in-sdkcli-source)
- [Category 4 — Eval docs and CI matrices](#category-4--eval-docs-and-ci-matrices)
- [Category 5 — Test fixtures (leave alone by default)](#category-5--test-fixtures-leave-alone-by-default)
- [Category 6 — Provider-detection heuristics](#category-6--provider-detection-heuristics)

---

## Category 1 — Hardcoded runtime defaults

These strings are returned by production code paths. Changing them affects real users.

- `libs/cli/deepagents_cli/config.py:1835` — default return `"openai:gpt-5.2"` when `OPENAI_API_KEY` is set and no `recent_model` is recorded.
- `libs/acp/deepagents_acp/server.py:987` — hardcoded `model="openai:gpt-5.2"` in the `build_agent` example used by the ACP server entrypoint.

## Category 2 — Example configs and their READMEs

Kept consistent per-example. The two `deploy-*` examples currently disagree on model family.

- `examples/deploy-content-writer/deepagents.toml:3` — `model = "openai:gpt-4.1"`.
- `examples/deploy-content-writer/README.md:11` — `` `OPENAI_API_KEY` | GPT-4.1 model access ``.
- `examples/deploy-gtm-agent/deepagents.toml:4` — `model = "openai:gpt-5.4-nano"`.
- `examples/deploy-gtm-agent/subagents/market-researcher/deepagents.toml:4` — `model = "openai:gpt-5.4-mini"`.
- `examples/deploy-gtm-agent/README.md:11` — `` `OPENAI_API_KEY` | Model access (gpt-5.4-nano) ``.
- `libs/acp/examples/demo_agent.py:111-113` — `openai:gpt-5.4-pro`, `openai:gpt-5.4`, `openai:gpt-5.3-codex`.

## Category 3 — Docstring examples in SDK/CLI source

Non-behavioral, but user-facing. Prefer ONE "current recommended" string per provider.

Top-level:

- `README.md:66` — `init_chat_model("openai:gpt-4o")` customization snippet.
- `libs/acp/README.md:133` — `{"value": "openai:gpt-4-turbo", "name": "GPT-4 Turbo"}`.

SDK (`libs/deepagents/deepagents/`):

- `_models.py:25` — docstring: `Model string (e.g. "openai:gpt-5.4")`.
- `_models.py:94` — docstring: matching example with `"openai:gpt-5"`.
- `_models.py:100` — docstring: `e.g., openai:gpt-5`.
- `graph.py:257` — docstring: `e.g., openai:gpt-5`.
- `profiles/_harness_profiles.py:131` — docstring: `e.g. "openai:o3-pro"`.
- `middleware/subagents.py:49` — docstring: `e.g., 'openai:gpt-4o'`.
- `middleware/subagents.py:122` — docstring example dict: `"model": "openai:gpt-4o"`.
- `middleware/subagents.py:486` — docstring example: `"openai:gpt-4o"` (SubAgentMiddleware example).
- `middleware/subagents.py:495` — docstring example: `"model": "openai:gpt-4o"`.
- `middleware/summarization.py:32` — module docstring example: `model="gpt-4o-mini"`.
- `middleware/summarization.py:263` — class docstring example: `model="gpt-4o-mini"`.
- `middleware/summarization.py:1166` — docstring example: `model = "openai:gpt-5.4"`.
- `middleware/summarization.py:1187` — docstring example: `model = "openai:gpt-5.4"`.
- `middleware/summarization.py:1233` — docstring example: `SummarizationMiddleware(model="gpt-4o-mini", ...)`.

CLI (`libs/cli/deepagents_cli/`):

- `_cli_context.py:23` — docstring: `e.g. 'openai:gpt-4o'`.
- `configurable_model.py:167` — docstring: `e.g. "openai:gpt-5"`.
- `main.py:623` — argparse help text: `e.g., claude-sonnet-4-6, gpt-5.2`.
- `ui.py:106` — printed help text: `e.g., gpt-4o`.
- `offload.py:233` — docstring: `e.g. "openai:gpt-4"`.
- `config.py:2174` — docstring: `'openai:gpt-4o'`.
- `config.py:2195-2196` — docstring examples: `create_model("openai:gpt-4o")`, `create_model("gpt-4o")`.
- `model_config.py:93` — docstring: `e.g., 'claude-sonnet-4-5', 'gpt-4o'`.

Evals source:

- `libs/evals/deepagents_evals/radar.py:471` — `toy_data()` fixture `ModelResult(model="openai:gpt-5.4", ...)`.

## Category 4 — Eval docs and CI matrices

Must stay in sync with each other.

- `libs/evals/MODEL_GROUPS.md`:
  - lines 35-44 (group 1): 10 OpenAI entries — `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `o3`, `o4-mini`, `gpt-5.1-codex`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.4`, `gpt-5.4-mini`.
  - lines 57-60: subset — `gpt-4.1`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.4`.
  - line 86: `openai:gpt-5.4`.
  - line 92: `openai:gpt-5.4-mini`.
  - lines 165-174: same 10-entry list as group 1.
  - lines 228-237: same 10-entry list as group 1.
- `.github/scripts/models.py` lines 351, 355, 359, 372, 376, 380, 384, 397, 410, 425 — matrix of OpenAI models mirrored from `MODEL_GROUPS.md`.
- `.github/workflows/evals.yml` lines 90-99, 106, 158-160 — workflow matrix and description text.
- `.github/workflows/harbor.yml` lines 96-105, 112 — workflow matrix and description text.

Rule of thumb: any edit to `MODEL_GROUPS.md` should also touch `.github/scripts/models.py` and the two workflow files.

## Category 5 — Test fixtures (leave alone by default)

~150+ occurrences. The majority are arbitrary fixture strings used to exercise parsing/switching logic; changing them risks breaking tests for the wrong reason. Concentrated in:

- `libs/deepagents/tests/unit_tests/test_models.py`
- `libs/deepagents/tests/unit_tests/middleware/test_summarization_factory.py`
- `libs/deepagents/tests/unit_tests/middleware/test_subagent_middleware_init.py`
- `libs/deepagents/tests/integration_tests/test_subagent_middleware.py`
- `libs/cli/tests/unit_tests/` — `test_bundler.py`, `test_config.py` (including provider-detection cases for `o1-preview`, `o3-mini`, `o4-mini`, `gpt-5.2`), `test_model_selector.py`, `test_agent.py`, `test_textual_adapter.py`, `test_offload.py`, `test_model_switch.py`, `test_autocomplete.py`, `test_main.py`, `test_main_args.py`, `test_non_interactive.py`, `test_app.py`, `test_configurable_model.py`, `test_args.py`, `test_theme.py`, `test_model_config.py`, `test_session_stats.py`, `test_reload.py`.
- `libs/cli/tests/unit_tests/deploy/test_config.py` — includes `openai:gpt-5.3-codex` as a config fixture.
- `libs/evals/tests/evals/tau2_airline/test_tau2_airline.py:75` — `USER_SIM_MODEL = "gpt-4.1-mini"` (simulator model; treat as configuration, not a test assertion).
- `libs/evals/tests/evals/memory_agent_bench/data_utils.py:124` — `tiktoken.encoding_for_model("gpt-4o-mini")` (tokenizer lookup, not a model call; only change if the tokenizer mapping needs to change).
- `libs/evals/tests/evals/fixtures/summarization_seed_messages.json:103` — embedded copy of the `summarization.py` module docstring. If you edit the real docstring, regenerate this fixture separately; do NOT hand-edit it.

## Category 6 — Provider-detection heuristics

- `libs/cli/deepagents_cli/config.py:1788` — prefix tuple `("gpt-", "o1", "o3", "o4", "chatgpt")` used to auto-detect that a bare model name belongs to OpenAI. Only extend this when adding a new OpenAI family (e.g. a future `gpt-6` or `o5`), and add a matching case in `libs/cli/tests/unit_tests/test_config.py` (see the existing `("o1-preview", "openai")`, `("o3-mini", "openai")`, `("o4-mini", "openai")` parametrization near line 2016).
