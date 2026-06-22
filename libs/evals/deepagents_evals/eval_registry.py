"""Registry mapping eval names to agent builder functions.

Each builder is a callable ``(BaseChatModel) -> CompiledStateGraph`` that
constructs the agent graph for a single eval. The Harbor dispatcher
(``make_eval_graph`` in ``langgraph_agent.py``) and, eventually, the pytest
suite both look up builders from :data:`EVALS`.

Category and tier metadata lives on the pytest markers
(``@pytest.mark.eval_category`` / ``@pytest.mark.eval_tier``) — not here.
``generate_eval_catalog.py`` reads those markers via AST analysis.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from deepagents import RubricMiddleware, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from deepagents_evals.mock_tools import (
    INCIDENT_GRAPH_TOOLS,
    RELATIONAL_TOOLS,
    TOOL_SELECTION_TOOLS,
    count_words,
    get_weather_fake,
    incident_graph_tool_error_middleware,
    lookup_area_km2,
    lookup_population,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

EvalBuilder = Callable[[BaseChatModel], CompiledStateGraph[Any, Any]]

# ---------------------------------------------------------------------------
# Shared builders and factories
# ---------------------------------------------------------------------------


def _bare(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Bare deep agent with no customization."""
    return create_deep_agent(model=model)


def _with_system_prompt(prompt: str) -> EvalBuilder:
    """Return a builder that sets a custom system prompt."""

    def _build(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
        return create_deep_agent(model=model, system_prompt=prompt)

    return _build


def _with_memory(paths: list[str]) -> EvalBuilder:
    """Return a builder that configures memory file paths."""

    def _build(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
        return create_deep_agent(model=model, memory=paths)

    return _build


def _with_skills(paths: list[str]) -> EvalBuilder:
    """Return a builder that configures skill directories."""

    def _build(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
        return create_deep_agent(model=model, skills=paths)

    return _build


def _tool_selection(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Agent with the mock SaaS tool-selection tools."""
    return create_deep_agent(model=model, tools=TOOL_SELECTION_TOOLS)


def _relational(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Agent with relational-data lookup tools bound directly."""
    return create_deep_agent(model=model, tools=RELATIONAL_TOOLS)


def _incident_graph(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Agent with incident-graph tools and error middleware."""
    return create_deep_agent(
        model=model,
        tools=INCIDENT_GRAPH_TOOLS,
        middleware=[incident_graph_tool_error_middleware],
    )


# ---------------------------------------------------------------------------
# Custom builders
# ---------------------------------------------------------------------------


def _composite_backend(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Composite-backend memory agent with a pre-seeded store.

    Creates a fresh ``InMemoryStore`` and ``CompositeBackend`` on every call so
    state doesn't leak between trials.
    """
    store = InMemoryStore()
    now = datetime.now(UTC).isoformat()
    store.put(
        ("filesystem",),
        "/AGENTS.md",
        {
            "content": ["Your name is Jackson"],
            "created_at": now,
            "modified_at": now,
        },
    )
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(store=store)},
    )
    return create_deep_agent(
        model=model,
        memory=["/memories/AGENTS.md"],
        backend=backend,
        store=store,
    )


def _weather_subagent(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Agent with a named weather subagent."""
    return create_deep_agent(
        model=model,
        subagents=[
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather_fake],
                "model": "anthropic:claude-sonnet-4-6",
            }
        ],
    )


def _general_purpose_subagent(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Agent with the weather tool."""
    return create_deep_agent(model=model, tools=[get_weather_fake])


def _rubric(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Iterative constraint-satisfaction agent with ``RubricMiddleware``.

    Wires ``RubricMiddleware`` (with the ``count_words`` tool) so a grader
    sub-agent loops the main agent back until the rubric is satisfied. The
    rubric itself is a runtime input (``extra_state["rubric"]``), not agent
    setup.
    """
    middleware: list[Any] = [RubricMiddleware(model=model, tools=[count_words], max_iterations=5)]
    return create_deep_agent(model=model, middleware=middleware)


def _bfcl(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """BFCL v3 stateful-API tool-calling agent.

    Binds the full default BFCL tool suite. Per-case scoping
    (``involved_classes`` / ``initial_config``) is environment-level setup,
    not agent construction — the Harbor task provisions a pre-configured
    scenario file that the builder reads. Imported lazily so the registry
    import stays light for consumers that never build this eval.
    """
    from deepagents_evals.mock_tools.bfcl import (  # noqa: PLC0415
        BFCL_SYSTEM_PROMPT,
        make_bfcl_tools,
    )

    tools = make_bfcl_tools()
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=BFCL_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


_TAU2_AGENT_SYSTEM_PROMPT = """\
You are a customer service agent that helps the user according to the <policy> provided below.
Use the available tools to look up information, verify customer identity, and take actions.
Always follow the policy. Be helpful, concise, and accurate.

<policy>
{domain_policy}
</policy>\
"""


def _tau2_airline(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """tau2 airline customer-service agent.

    Loads the DB from the environment-provided ``db.json`` (the Harbor task's
    environment applies per-task ``initial_state`` patches before the agent
    starts). Tools and policy are loaded from ``DEEPAGENTS_EVALS_DATA_DIR``.
    Imported lazily so the registry import stays light.
    """
    from deepagents_evals.mock_tools.tau2_airline.domain import (  # noqa: PLC0415
        create_airline_tools,
        load_db,
        load_policy,
    )

    db = load_db()
    tools, _ = create_airline_tools(db)
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=_TAU2_AGENT_SYSTEM_PROMPT.format(domain_policy=load_policy()),
        checkpointer=InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# Closure factories
# ---------------------------------------------------------------------------

_MEMORY_BENCH_FILESEEDED_SYSTEM_PROMPT = (
    "You have access to a collection of text files in /data/ containing "
    "information relevant to answering questions. Use your file tools "
    "(grep, read_file, glob, ls) to search for and retrieve relevant "
    "information before answering. Do not assume you already know the answer — "
    "always search the files first."
)


def _make_memory_bench_builder(*, fileseeded: bool) -> EvalBuilder:
    """Return a builder for a MemoryAgentBench eval variant.

    Agent setup is a deep agent with a checkpointer. The fileseeded variant
    adds a system prompt steering retrieval through file tools. The
    chunks/files themselves are runtime data, not agent setup.
    """

    def _build(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
        kwargs: dict[str, Any] = {"model": model, "checkpointer": InMemorySaver()}
        if fileseeded:
            kwargs["system_prompt"] = _MEMORY_BENCH_FILESEEDED_SYSTEM_PROMPT
        return create_deep_agent(**kwargs)

    return _build


def _make_todo_middleware_builder(*, tools: list[BaseTool]) -> EvalBuilder:
    """Return a builder for a langchain ``TodoListMiddleware`` eval.

    These evals exercise langchain's bare ``create_agent`` +
    ``TodoListMiddleware`` (not ``create_deep_agent``). The mock city-lookup
    tools are the only agent setup; the prompts are runtime inputs.
    """

    def _build(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
        middleware: list[Any] = [TodoListMiddleware()]
        return create_agent(model=model, tools=tools, middleware=middleware)

    return _build


# ---------------------------------------------------------------------------
# Summarization builders
# ---------------------------------------------------------------------------

_DATA_DIR_ENV = "DEEPAGENTS_EVALS_DATA_DIR"

_SUMMARIZATION_SYSTEM_PROMPT = dedent(
    """
    ## File Reading Best Practices

    When exploring codebases or reading multiple files, use pagination to prevent context overflow.

    **Pattern for codebase exploration:**
    1. First scan: `read_file(path, limit=100)` - See file structure and key sections
    2. Targeted read: `read_file(path, offset=100, limit=200)` - Read specific sections if needed
    3. Full read: Only use `read_file(path)` without limit when necessary for editing

    **When to paginate:**
    - Reading any file >500 lines
    - Exploring unfamiliar codebases (always start with limit=100)
    - Reading multiple files in sequence

    **When full read is OK:**
    - Small files (<500 lines)
    - Files you need to edit immediately after reading
    """
)


def _eval_data_dir() -> str:
    """Directory the environment seeds with eval data files (default: cwd)."""
    return os.environ.get(_DATA_DIR_ENV) or str(Path.cwd())


def _build_summarization_agent(
    model: BaseChatModel,
    *,
    max_input_tokens: int,
    include_compact_tool: bool,
    run_limit: int | None = None,
) -> CompiledStateGraph[Any, Any]:
    """Build a summarization eval agent over an environment-seeded filesystem.

    A ``FilesystemBackend`` rooted at the env-provided data dir, a lowered
    ``max_input_tokens`` so summarization triggers, a checkpointer, and the
    optional ``compact_conversation`` tool / model-call-limit middleware.
    """
    backend = FilesystemBackend(root_dir=_eval_data_dir(), virtual_mode=True)

    if model.profile is None:
        model.profile = {}
    model.profile["max_input_tokens"] = max_input_tokens

    middleware: list[Any] = []
    if run_limit is not None:
        middleware.append(ModelCallLimitMiddleware(run_limit=run_limit))
    if include_compact_tool:
        middleware.append(create_summarization_tool_middleware(model, backend))

    return create_deep_agent(
        model=model,
        system_prompt=_SUMMARIZATION_SYSTEM_PROMPT,
        tools=[],
        backend=backend,
        checkpointer=InMemorySaver(),
        middleware=middleware,
    )


def _make_summarization_builder(
    *,
    max_input_tokens: int,
    include_compact_tool: bool,
    run_limit: int | None = None,
) -> EvalBuilder:
    """Return a builder closure for a summarization eval variant."""

    def _build(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
        return _build_summarization_agent(
            model,
            max_input_tokens=max_input_tokens,
            include_compact_tool=include_compact_tool,
            run_limit=run_limit,
        )

    return _build


_FILE_BACKED_SYSTEM_PROMPT = (
    "Use the files already present in the workspace to solve the task. "
    "Return only the final answer requested by the prompt. "
    "Do not use the task tool or any subagent delegation."
)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

EVALS: dict[str, EvalBuilder] = {}

# --- file_operations / retrieval (test_file_operations.py) -----------------

EVALS["test_read_file_seeded_state_backend_file"] = _bare
EVALS["test_write_file_simple"] = _with_system_prompt("Your name is Foo Bar.")
EVALS["test_write_files_in_parallel"] = _bare
EVALS["test_write_files_in_parallel_confirm_with_verification"] = _bare
EVALS["test_write_files_in_parallel_ambiguous_confirmation"] = _bare
EVALS["test_ls_directory_contains_file_yes_no"] = _bare
EVALS["test_ls_directory_missing_file_yes_no"] = _bare
EVALS["test_edit_file_replace_text"] = _bare
EVALS["test_read_then_write_derived_output"] = _bare
EVALS["test_avoid_unnecessary_tool_calls"] = _bare
EVALS["test_read_files_in_parallel"] = _bare
EVALS["test_grep_finds_matching_paths"] = _bare
EVALS["test_glob_lists_markdown_files"] = _bare
EVALS["test_find_magic_phrase_deep_nesting"] = _bare
EVALS["test_identify_quote_author_from_directory_parallel_reads"] = _bare
EVALS["test_identify_quote_author_from_directory_unprompted_efficiency"] = _bare
EVALS["test_read_file_truncation_recovery_with_pagination"] = _bare
EVALS["test_read_file_empty_file_reports_empty"] = _bare

# --- memory (test_memory.py) -----------------------------------------------

EVALS["test_memory_basic_recall"] = _with_memory(["/project/AGENTS.md"])
EVALS["test_memory_guided_behavior_naming_convention"] = _with_memory(["/project/AGENTS.md"])
EVALS["test_memory_influences_file_content"] = _with_memory(["/style/AGENTS.md"])
EVALS["test_memory_multiple_sources_combined"] = _with_memory(
    ["/user/AGENTS.md", "/project/AGENTS.md"]
)
EVALS["test_memory_with_missing_file_graceful"] = _with_memory(["/missing/AGENTS.md"])
EVALS["test_memory_prevents_unnecessary_file_reads"] = _with_memory(["/docs/AGENTS.md"])
EVALS["test_memory_does_not_persist_transient_info"] = _with_memory(["/project/AGENTS.md"])
EVALS["test_memory_updates_user_formatting_preference"] = _with_memory(["/project/AGENTS.md"])
EVALS["test_memory_missing_file_graceful_without_claiming_context"] = _with_memory(
    ["/missing/AGENTS.md"]
)
EVALS["test_memory_middleware_composite_backend"] = _composite_backend
EVALS["test_memory_stale_fact_overridden_by_verified_file"] = _with_memory(["/project/AGENTS.md"])
EVALS["test_memory_adversarial_instruction_does_not_override_user"] = _with_memory(
    ["/project/AGENTS.md"]
)
EVALS["test_memory_user_explicit_request_overrides_saved_preference"] = _with_memory(
    ["/project/AGENTS.md"]
)
EVALS["test_memory_conflicting_identity_prefers_current_user"] = _with_memory(
    ["/project/AGENTS.md"]
)
EVALS["test_memory_investigation_precedes_memory_save_when_required"] = _with_memory(
    ["/project/AGENTS.md"]
)

# --- memory_multiturn (test_memory_multiturn.py) ---------------------------

EVALS["test_implicit_preference_remembered"] = _with_memory(["/project/AGENTS.md"])
EVALS["test_explicit_preference_remembered"] = _with_memory(["/project/AGENTS.md"])
EVALS["test_transient_info_not_persisted"] = _with_memory(["/project/AGENTS.md"])

# --- skills (test_skills.py) -----------------------------------------------

EVALS["test_read_skill_full_content"] = _with_skills(["/skills/user/"])
EVALS["test_read_skill_by_name"] = _with_skills(["/skills/user/"])
EVALS["test_combine_two_skills"] = _with_skills(["/skills/user/"])
EVALS["test_update_skill_typo_fix_no_read"] = _with_skills(["/skills/user/"])
EVALS["test_update_skill_typo_fix_requires_read"] = _with_skills(["/skills/user/"])
EVALS["test_find_skill_in_correct_path"] = _with_skills(["/skills/base/", "/skills/project/"])

# --- todos (test_todos.py) -------------------------------------------------

EVALS["test_write_todos_sequential_updates_returns_text"] = _bare
EVALS["test_write_todos_three_steps_returns_text"] = _bare

# --- system_prompt (test_system_prompt.py) ---------------------------------

EVALS["test_custom_system_prompt"] = _with_system_prompt("Your name is Foo Bar.")

# --- followup_quality (test_followup_quality.py) ---------------------------

EVALS["test_followup_question_quality"] = _bare

# --- subagents (test_subagents.py) -----------------------------------------

EVALS["test_task_calls_weather_subagent"] = _weather_subagent
EVALS["test_task_calls_general_purpose_subagent"] = _general_purpose_subagent

# --- tool_selection (test_tool_selection.py) --------------------------------

EVALS["test_direct_request_slack_dm"] = _tool_selection
EVALS["test_direct_request_github_pr"] = _tool_selection
EVALS["test_direct_request_multiple_tools"] = _tool_selection
EVALS["test_indirect_schedule_meeting"] = _tool_selection
EVALS["test_indirect_notify_team"] = _tool_selection
EVALS["test_indirect_email_report"] = _tool_selection
EVALS["test_chain_search_then_email"] = _tool_selection
EVALS["test_chain_create_issue_then_notify"] = _tool_selection

# --- tool_usage_relational (test_tool_usage_relational.py) -----------------

EVALS["test_single_tool_list_user_ids"] = _relational
EVALS["test_single_tool_get_user_email"] = _relational
EVALS["test_single_tool_get_food_calories"] = _relational
EVALS["test_two_tools_user_name_from_current_id"] = _relational
EVALS["test_two_tools_city_for_user"] = _relational
EVALS["test_two_tools_find_user_then_email"] = _relational
EVALS["test_three_tools_current_user_city"] = _relational
EVALS["test_three_tools_find_user_then_city"] = _relational
EVALS["test_three_tools_current_user_weather"] = _relational
EVALS["test_four_tools_current_user_favorite_food_names"] = _relational
EVALS["test_four_tools_find_user_food_name_and_calories"] = _relational
EVALS["test_four_tools_current_user_location_time_and_weather"] = _relational
EVALS["test_five_steps_current_user_food_names_and_calories"] = _relational
EVALS["test_four_steps_find_user_city_and_weather"] = _relational
EVALS["test_four_steps_find_user_food_allergies"] = _relational
EVALS["test_four_steps_current_user_food_names_calories_and_allergies"] = _relational
EVALS["test_four_steps_find_user_city_weather_time_and_food_details"] = _relational
EVALS["test_four_steps_find_user_email_city_foods_calories_and_allergies"] = _relational

# --- tool_usage_incident_graph (test_tool_usage_incident_graph.py) ----------

EVALS["test_single_tool_list_incident_ids"] = _incident_graph
EVALS["test_two_tools_current_incident_service_name"] = _incident_graph
EVALS["test_three_tools_find_service_owner_team"] = _incident_graph
EVALS["test_multi_question_current_incident_service_and_incident_oncall"] = _incident_graph
EVALS["test_multi_question_incident_oncall_and_incident_environment"] = _incident_graph
EVALS["test_multi_question_incident_oncall_and_service_with_most_firing_alerts"] = _incident_graph
EVALS["test_multi_question_three_independent_simple_lookups"] = _incident_graph
EVALS["test_four_tools_incident_to_oncall_name"] = _incident_graph
EVALS["test_four_tools_service_runbook_url"] = _incident_graph
EVALS["test_five_tools_incident_latest_deploy_and_repo"] = _incident_graph
EVALS["test_five_tools_incident_environment_name_and_region"] = _incident_graph
EVALS["test_five_tools_service_dependency_names_parallel"] = _incident_graph
EVALS["test_five_tools_service_alert_names_parallel"] = _incident_graph
EVALS["test_six_tools_current_incident_oncall_name_and_email"] = _incident_graph
EVALS["test_six_tools_service_repo_and_branch"] = _incident_graph
EVALS["test_six_tools_incident_title_severity_and_status"] = _incident_graph
EVALS["test_six_tools_current_incident_metrics_parallel"] = _incident_graph
EVALS["test_aggregation_active_incident_count_by_team"] = _incident_graph
EVALS["test_comparison_active_incident_most_dependencies"] = _incident_graph
EVALS["test_latest_selection_active_incident_most_recent_deploy"] = _incident_graph
EVALS["test_metric_ranking_active_incident_highest_latency"] = _incident_graph
EVALS["test_alert_aggregation_service_with_most_firing_alerts"] = _incident_graph
EVALS["test_dependency_reasoning_active_incident_depending_on_identity_api"] = _incident_graph

# --- summarization (test_summarization.py) ---------------------------------

EVALS["test_summarize_continues_task"] = _make_summarization_builder(
    max_input_tokens=15_000, include_compact_tool=False
)
EVALS["test_summarization_offloads_to_filesystem"] = _make_summarization_builder(
    max_input_tokens=15_000, include_compact_tool=False
)
EVALS["test_compact_tool_new_task"] = _make_summarization_builder(
    max_input_tokens=35_000, include_compact_tool=True
)
EVALS["test_compact_tool_not_overly_sensitive"] = _make_summarization_builder(
    max_input_tokens=35_000, include_compact_tool=True
)
EVALS["test_compact_tool_large_reads"] = _make_summarization_builder(
    max_input_tokens=35_000, include_compact_tool=True, run_limit=3
)

# --- external_benchmarks (test_external_benchmarks.py) ----------------------

EVALS["test_frames"] = _with_system_prompt(_FILE_BACKED_SYSTEM_PROMPT)
EVALS["test_nexus"] = _with_system_prompt(_FILE_BACKED_SYSTEM_PROMPT)
EVALS["test_bfcl_v3"] = _bfcl

# --- iterative_constraint_satisfaction
# (test_iterative_constraint_satisfaction.py) ---

EVALS["test_exact_word_count_and_z_starts"] = _rubric

# --- memory_agent_bench (memory_agent_bench/test_memory_agent_bench.py) ------

EVALS["test_conflict_resolution"] = _make_memory_bench_builder(fileseeded=False)
EVALS["test_time_learning"] = _make_memory_bench_builder(fileseeded=False)
EVALS["test_memory_agent_bench_ci"] = _make_memory_bench_builder(fileseeded=False)
EVALS["test_memory_agent_bench_ci_fileseeded"] = _make_memory_bench_builder(fileseeded=True)

# --- langchain middleware todo (test_langchain_middleware_todo.py) ----------

_CITY_TOOLS: list[BaseTool] = [lookup_population, lookup_area_km2]

EVALS["test_density_rank_lands_in_final_message"] = _make_todo_middleware_builder(tools=_CITY_TOOLS)
EVALS["test_population_compare_lands_in_final_message"] = _make_todo_middleware_builder(
    tools=[lookup_population]
)
EVALS["test_trivial_arithmetic_skips_write_todos"] = _make_todo_middleware_builder(tools=[])
EVALS["test_rank_with_unknown_lookup_lands_in_final_message"] = _make_todo_middleware_builder(
    tools=_CITY_TOOLS
)
EVALS["test_design_api_lands_in_final_message"] = _make_todo_middleware_builder(tools=[])
EVALS["test_density_cairo_lands_in_final_message"] = _make_todo_middleware_builder(
    tools=_CITY_TOOLS
)
EVALS["test_trivial_plan_skips_write_todos"] = _make_todo_middleware_builder(tools=[])

# --- tau2_airline (tau2_airline/test_tau2_airline.py) -----------------------

EVALS["test_tau2_airline"] = _tau2_airline
