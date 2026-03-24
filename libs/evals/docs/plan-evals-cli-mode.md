# Plan: add `--agent-mode cli` support to evals

## Context

Harbor already supports both SDK and CLI agent modes via `DeepAgentsWrapper._use_cli_agent`.
Evals currently hardcode `create_deep_agent()` (SDK only) in every test file.
Goal: run the same eval suite against the CLI agent to measure SDK vs CLI parity.

## Key differences between the two factory functions

| | `create_deep_agent` (SDK) | `create_cli_agent` (CLI) |
|---|---|---|
| Package | `deepagents` | `deepagents_cli.agent` |
| Returns | `CompiledStateGraph` | `tuple[Pregel, CompositeBackend]` |
| Required args | `model` | `model`, `assistant_id` |
| Extra kwargs | `tools`, `system_prompt`, `memory`, `skills`, `subagents`, `interrupt_on`, `checkpointer`, `middleware`, `backend`, `store` | `tools`, `system_prompt`, `sandbox`, `auto_approve`, `enable_memory`, `enable_skills`, `enable_shell`, `checkpointer` |
| Kwargs with no CLI equivalent | `memory`, `skills`, `subagents`, `interrupt_on`, `middleware`, `backend`, `store` | — |

## Compatibility analysis: which evals can run in CLI mode?

Tests pass these kwargs to `create_deep_agent`:

| Kwarg used | Test files | CLI-compatible? |
|---|---|---|
| `model` only | `test_file_operations`, `test_todos`, `test_followup_quality` | Yes |
| `+ tools` | `test_tool_selection`, `test_tool_usage_*`, `test_subagents` (one test) | Yes (`tools` exists on both) |
| `+ system_prompt` | `test_system_prompt`, `test_file_operations` (1 test), `test_summarization`, `external_benchmarks`, `tau2_airline` | Yes |
| `+ checkpointer` | `test_hitl`, `test_summarization`, `external_benchmarks`, `memory_agent_bench` | Yes |
| `+ memory` | `test_memory`, `test_memory_multiturn` | **No** — `create_cli_agent` has `enable_memory` bool, not path-based `memory` list |
| `+ skills` | `test_skills` | **No** — CLI has `enable_skills` bool, not skill paths |
| `+ subagents` | `test_subagents`, `test_hitl` (1 test) | **No** — CLI has `async_subagents` (different type) |
| `+ interrupt_on` | `test_hitl` | **No** — CLI uses `auto_approve` bool instead |
| `+ middleware` | `test_summarization` | **No** — not exposed on CLI |
| `+ backend` / `+ store` | `test_memory` (1 test) | **No** — CLI takes `sandbox` (different protocol) |

**~60% of evals are CLI-compatible out of the box** (the `model`-only, `tools`, `system_prompt`, `checkpointer` variants). The rest need either:

- Skip in CLI mode (phase 1)
- Kwarg translation layer (phase 2)

## Implementation plan

### Phase 1: basic CLI mode with compatible evals

#### 1. `conftest.py` — add `--agent-mode` option + agent factory

```python
# New pytest option
parser.addoption(
    "--agent-mode",
    choices=["sdk", "cli"],
    default="sdk",
    help="Agent implementation to evaluate.",
)

# New fixture
@pytest.fixture
def agent_mode(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--agent-mode")
```

Add an `agent_factory` fixture that wraps both creation paths:

```python
@pytest.fixture
def agent_factory(model: BaseChatModel, model_name: str, agent_mode: str):
    """Return a callable matching create_deep_agent's signature.

    In SDK mode, delegates directly. In CLI mode, translates compatible
    kwargs and returns just the Pregel (discards CompositeBackend).
    """
    if agent_mode == "sdk":
        return create_deep_agent

    def _create_cli(**kwargs):
        # translate compatible kwargs
        from deepagents_cli.agent import create_cli_agent
        cli_kwargs = {
            "model": kwargs.pop("model"),
            "assistant_id": str(uuid.uuid4()),
            "auto_approve": True,
            "enable_memory": False,
            "enable_skills": False,
            "enable_shell": False,
        }
        for k in ("tools", "system_prompt", "checkpointer"):
            if k in kwargs:
                cli_kwargs[k] = kwargs.pop(k)

        # Remaining kwargs incompatible — warn or skip
        if kwargs:
            pytest.skip(f"CLI mode: unsupported kwargs {set(kwargs)}")

        agent, _backend = create_cli_agent(**cli_kwargs)
        return agent

    return _create_cli
```

Key decisions:

- `auto_approve=True` — headless; no HITL prompts
- `enable_memory/skills/shell=False` — match SDK's minimal defaults
- `assistant_id` — uuid per test to isolate state
- Lazy import of `deepagents_cli.agent` — avoids import cost when running SDK mode

#### 2. Test file migration

Two options (recommend **option A** for least churn):

**Option A: incremental — mark CLI-compatible tests**

Add a `pytest.mark.cli_compatible` marker. Only tests with this marker run in CLI mode; others auto-skip. This avoids touching every test file immediately.

```python
# conftest.py
def pytest_collection_modifyitems(config, items):
    if config.getoption("--agent-mode") == "cli":
        for item in items:
            if not item.get_closest_marker("cli_compatible"):
                item.add_marker(pytest.mark.skip("not CLI-compatible"))
```

Then progressively add `@pytest.mark.cli_compatible` to test files/functions that only use compatible kwargs.

**Option B: full refactor — replace all `create_deep_agent()` calls with fixture**

Change every test from:

```python
agent = create_deep_agent(model=model, tools=TOOLS)
```

to:

```python
agent = agent_factory(model=model, tools=TOOLS)
```

This touches ~15 files and ~70 call sites. Cleaner long-term but bigger PR.

#### 3. `langsmith_experiment_metadata` — add agent mode

```python
@pytest.fixture(scope="session")
def langsmith_experiment_metadata(request):
    ...
    return {
        "model": model_name,
        "agent_mode": request.config.getoption("--agent-mode"),
        "date": ...,
        "deepagents_version": ...,
    }
```

#### 4. `.github/workflows/evals.yml` — add `agent_mode` input

```yaml
inputs:
  agent_mode:
    description: "Agent implementation to evaluate"
    required: false
    default: "sdk"
    type: choice
    options:
      - sdk
      - cli
```

Pass through:

```yaml
env:
  PYTEST_ADDOPTS: "--model ${{ matrix.model }} --agent-mode ${{ inputs.agent_mode }}"
```

#### 5. `Makefile` — convenience target

```makefile
evals-cli:
	PYTEST_ADDOPTS="--agent-mode cli" $(MAKE) evals
```

### Phase 2: expand CLI compatibility (future)

#### 6. Kwarg translation for `memory`, `skills`

Map SDK `memory=["/path/AGENTS.md"]` to CLI's memory system. Requires understanding how CLI memory paths differ from SDK's path-based injection. May need `create_cli_agent` API changes.

#### 7. HITL translation

Map `interrupt_on={...}` to CLI's approval flow. The CLI uses `auto_approve` as a global toggle, not per-tool config. Might need a CLI-side feature addition.

#### 8. Subagent translation

Map `subagents=[...]` to `async_subagents=[AsyncSubAgent(...)]`. Types differ; needs adapter.

## Files to modify

| File | Change |
|---|---|
| `libs/evals/tests/evals/conftest.py` | `--agent-mode` option, `agent_factory` fixture, metadata, collection hook |
| `.github/workflows/evals.yml` | `agent_mode` input + `PYTEST_ADDOPTS` |
| `libs/evals/Makefile` | `evals-cli` target |
| `libs/evals/tests/evals/test_file_operations.py` | Add `cli_compatible` marker (option A) or swap to `agent_factory` (option B) |
| Same for: `test_todos`, `test_followup_quality`, `test_tool_selection`, `test_tool_usage_*`, `test_system_prompt`, `test_summarization`, `external_benchmarks`, `tau2_airline` | Same |

## Risks

- **Behavioral drift**: CLI agent has a richer system prompt (`get_system_prompt()`) and more middleware. Evals tuned for SDK's minimal agent may fail on CLI due to extra tool calls, verbosity, or different tool names.
- **Performance**: CLI agent imports are heavy; lazy import mitigates but each test still pays the cost.
- **Flakiness**: CLI middleware (memory, skills) is disabled but `enable_shell=False` may affect tool availability — need to verify CLI still has file read/write tools without shell.
- **Maintenance**: Two code paths in conftest means bugs can hide in one mode.

## Recommendation

Start with **Phase 1 + Option A** (marker-based). Smallest PR, no test file rewrites, immediate value for the compatible subset (~60% of evals). Validate CLI vs SDK scores divergence before investing in full kwarg translation.
