"""Registry of every eval in the Deep Agents evaluation suite.

This module is the single source of truth for how each eval's agent is
constructed. Both the pytest suite and the Harbor sandbox dispatcher import
from here, ensuring the ``create_deep_agent`` call for a given eval exists in
exactly one place.

Each eval is described by an :class:`EvalSpec` that captures the
``create_deep_agent`` kwargs (``system_prompt``, ``memory``, ``skills``,
``tools``, ``middleware``, ``subagents``, ``backend``, ``store``) or a custom
``builder`` callable for evals whose construction depends on runtime parameters
(e.g. ``repl_name`` for the relational / incident-graph suites).

The :data:`EVALS` dict maps the pytest test function name to its
:class:`EvalSpec`. Parametrized evals (memory_multiturn, followup_quality) use
the test function name without the parametrize suffix; the Harbor dispatcher
receives the base name via ``configurable["eval_name"]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents_evals.mock_tools import (
    INCIDENT_GRAPH_TOOLS,
    RELATIONAL_TOOLS,
    TOOL_SELECTION_TOOLS,
    get_weather_fake,
    incident_graph_tool_error_middleware,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

# ---------------------------------------------------------------------------
# Builder functions for evals whose construction depends on runtime params
# ---------------------------------------------------------------------------


def _relational_builder(
    model: BaseChatModel, *, repl_name: str | None = None
) -> CompiledStateGraph[Any, Any]:
    """Build the relational-data agent.

    When ``repl_name`` is ``None`` the tools are bound directly. When it is
    ``"quickjs"`` the tools are routed through a ``CodeInterpreterMiddleware``
    instead.
    """
    middleware: list[Any] = []
    tools: list[BaseTool] | None = None
    if repl_name == "quickjs":
        from langchain_quickjs import CodeInterpreterMiddleware  # noqa: PLC0415

        ptc: list[str | BaseTool] = [*RELATIONAL_TOOLS]
        middleware = [CodeInterpreterMiddleware(ptc=ptc)]
    elif repl_name is None:
        tools = RELATIONAL_TOOLS
    else:
        msg = f'Unknown repl_name "{repl_name}"'
        raise ValueError(msg)
    return create_deep_agent(model=model, tools=tools, middleware=middleware)


def _incident_graph_builder(
    model: BaseChatModel, *, repl_name: str | None = None
) -> CompiledStateGraph[Any, Any]:
    """Build the incident-management agent.

    Always includes the tool-error middleware. When ``repl_name`` is
    ``"quickjs"`` the tools are routed through a ``CodeInterpreterMiddleware``
    instead of being bound directly.
    """
    middleware: list[Any] = [incident_graph_tool_error_middleware]
    tools: list[BaseTool] | None = None
    if repl_name == "quickjs":
        from langchain_quickjs import CodeInterpreterMiddleware  # noqa: PLC0415

        ptc: list[str | BaseTool] = [*INCIDENT_GRAPH_TOOLS]
        middleware.append(CodeInterpreterMiddleware(ptc=ptc))
    elif repl_name is None:
        tools = INCIDENT_GRAPH_TOOLS
    else:
        msg = f'Unknown repl_name "{repl_name}"'
        raise ValueError(msg)
    return create_deep_agent(model=model, tools=tools, middleware=middleware)


def _composite_backend_builder(
    model: BaseChatModel,
    *,
    repl_name: str | None = None,  # noqa: ARG001
) -> CompiledStateGraph[Any, Any]:
    """Build the composite-backend memory agent.

    Creates a fresh ``InMemoryStore`` and ``CompositeBackend`` on every call so
    state doesn't leak between trials.
    """
    store = InMemoryStore()
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


def _weather_subagent_builder(
    model: BaseChatModel,
    *,
    repl_name: str | None = None,  # noqa: ARG001
) -> CompiledStateGraph[Any, Any]:
    """Build the agent with a named weather subagent."""
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


def _general_purpose_subagent_builder(
    model: BaseChatModel,
    *,
    repl_name: str | None = None,  # noqa: ARG001
) -> CompiledStateGraph[Any, Any]:
    """Build the agent with the weather tool and general-purpose subagent."""
    return create_deep_agent(model=model, tools=[get_weather_fake])


# ---------------------------------------------------------------------------
# Summarization builders (test_summarization.py)
#
# These agents read a large source file that overflows the context window so
# auto-summarization triggers. Unlike the pytest suite (which downloads the
# pinned file into a per-test ``tmp_path``), the file is *environment data*:
# the Harbor task's ``environment/Dockerfile`` provisions the pinned
# ``summarization.py`` (and ``filesystem.py`` for the large-reads eval) into
# ``DEEPAGENTS_EVALS_DATA_DIR``, and the builder roots a ``FilesystemBackend``
# there. The builder never seeds files or hits the network.
# ---------------------------------------------------------------------------

_DATA_DIR_ENV = "DEEPAGENTS_EVALS_DATA_DIR"
"""Env var naming the directory the environment seeds with eval data files."""

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
"""Mirror of ``test_summarization.SYSTEM_PROMPT`` so the registry owns the config."""


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

    Mirrors ``test_summarization._setup_summarization_test``: a
    ``FilesystemBackend`` rooted at the env-provided data dir, a lowered
    ``max_input_tokens`` so summarization triggers, a checkpointer, and the
    optional ``compact_conversation`` tool / model-call-limit middleware. The
    large data file itself is provisioned by the environment, not here.
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
) -> Callable[..., CompiledStateGraph[Any, Any]]:
    """Return a registry builder closure for a summarization eval variant."""

    def _builder(
        model: BaseChatModel,
        *,
        repl_name: str | None = None,  # noqa: ARG001
    ) -> CompiledStateGraph[Any, Any]:
        return _build_summarization_agent(
            model,
            max_input_tokens=max_input_tokens,
            include_compact_tool=include_compact_tool,
            run_limit=run_limit,
        )

    return _builder


# ---------------------------------------------------------------------------
# EvalSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSpec:
    """Describes how to build the agent for a single eval.

    Most evals set only ``category``, ``tier``, and optionally ``system_prompt``
    / ``memory`` / ``skills`` / ``tools``. Evals whose construction depends on
    runtime parameters (e.g. ``repl_name``) set ``builder`` instead; when
    ``builder`` is set, ``build()`` delegates to it and ignores the static
    fields.

    Attributes:
        name: The pytest test function name (without parametrize suffix).
        category: Eval category (file_operations, tool_use, memory, etc.).
        tier: Eval tier (baseline or hillclimb).
        system_prompt: Custom system prompt passed to ``create_deep_agent``.
        memory: Memory file paths passed to ``create_deep_agent``.
        skills: Skill directory paths passed to ``create_deep_agent``.
        tools: Tools passed to ``create_deep_agent`` (replaces default tools).
        middleware: Middleware passed to ``create_deep_agent``.
        subagents: Subagent configs passed to ``create_deep_agent``.
        builder: Custom builder for evals that need runtime construction.
            When set, takes precedence over the static fields.
        supports_repl: Whether this eval supports the ``repl_name`` parameter
            (only ``builder``-based evals do).
    """

    name: str
    category: str = ""
    tier: str = ""
    system_prompt: str | None = None
    memory: list[str] | None = None
    skills: list[str] | None = None
    tools: list[BaseTool] | None = None
    middleware: list[Any] | None = None
    subagents: list[dict[str, Any]] | None = None
    builder: Callable[..., CompiledStateGraph[Any, Any]] | None = None
    supports_repl: bool = False

    def build(
        self,
        model: BaseChatModel,
        *,
        repl_name: str | None = None,
    ) -> CompiledStateGraph[Any, Any]:
        """Build the agent graph for this eval.

        Args:
            model: The chat model to use.
            repl_name: Optional REPL backend (``"quickjs"`` or ``None``).
                Only used by evals with ``supports_repl=True``.

        Returns:
            A compiled LangGraph graph.
        """
        if self.builder is not None:
            return self.builder(model, repl_name=repl_name)

        kwargs: dict[str, Any] = {"model": model}
        if self.system_prompt is not None:
            kwargs["system_prompt"] = self.system_prompt
        if self.memory is not None:
            kwargs["memory"] = self.memory
        if self.skills is not None:
            kwargs["skills"] = self.skills
        if self.tools is not None:
            kwargs["tools"] = self.tools
        if self.middleware is not None:
            kwargs["middleware"] = self.middleware
        if self.subagents is not None:
            kwargs["subagents"] = self.subagents
        return create_deep_agent(**kwargs)


# ---------------------------------------------------------------------------
# Helpers for building the registry
# ---------------------------------------------------------------------------


def _default(name: str, category: str, tier: str, **kwargs: Any) -> EvalSpec:
    """Create an EvalSpec with the standard fields."""
    return EvalSpec(name=name, category=category, tier=tier, **kwargs)


def _builder_eval(
    name: str,
    category: str,
    tier: str,
    builder: Callable[..., CompiledStateGraph[Any, Any]],
    *,
    supports_repl: bool = True,
) -> EvalSpec:
    """Create an EvalSpec backed by a custom builder function."""
    return EvalSpec(
        name=name,
        category=category,
        tier=tier,
        builder=builder,
        supports_repl=supports_repl,
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

EVALS: dict[str, EvalSpec] = {}


def _register(spec: EvalSpec) -> EvalSpec:
    """Register an EvalSpec and return it for chaining."""
    EVALS[spec.name] = spec
    return spec


# --- file_operations / retrieval (test_file_operations.py) -----------------

_FILE_OPS = "file_operations"
_RETRIEVAL = "retrieval"
_BASELINE = "baseline"
_HILLCLIMB = "hillclimb"

_register(_default("test_read_file_seeded_state_backend_file", _FILE_OPS, _BASELINE))
_register(
    _default("test_write_file_simple", _FILE_OPS, _BASELINE, system_prompt="Your name is Foo Bar.")
)
_register(_default("test_write_files_in_parallel", _FILE_OPS, _BASELINE))
_register(_default("test_write_files_in_parallel_confirm_with_verification", _FILE_OPS, _BASELINE))
_register(_default("test_write_files_in_parallel_ambiguous_confirmation", _FILE_OPS, _BASELINE))
_register(_default("test_ls_directory_contains_file_yes_no", _FILE_OPS, _BASELINE))
_register(_default("test_ls_directory_missing_file_yes_no", _FILE_OPS, _BASELINE))
_register(_default("test_edit_file_replace_text", _FILE_OPS, _BASELINE))
_register(_default("test_read_then_write_derived_output", _FILE_OPS, _BASELINE))
_register(_default("test_avoid_unnecessary_tool_calls", _FILE_OPS, _BASELINE))
_register(_default("test_read_files_in_parallel", _FILE_OPS, _BASELINE))
_register(_default("test_grep_finds_matching_paths", _RETRIEVAL, _BASELINE))
_register(_default("test_glob_lists_markdown_files", _RETRIEVAL, _BASELINE))
_register(_default("test_find_magic_phrase_deep_nesting", _RETRIEVAL, _BASELINE))
_register(
    _default("test_identify_quote_author_from_directory_parallel_reads", _RETRIEVAL, _BASELINE)
)
_register(
    _default(
        "test_identify_quote_author_from_directory_unprompted_efficiency",
        _RETRIEVAL,
        _BASELINE,
    )
)
_register(_default("test_read_file_truncation_recovery_with_pagination", _FILE_OPS, _BASELINE))
_register(_default("test_read_file_empty_file_reports_empty", _FILE_OPS, _BASELINE))

# --- memory (test_memory.py) -----------------------------------------------

_MEMORY = "memory"

_register(_default("test_memory_basic_recall", _MEMORY, _BASELINE, memory=["/project/AGENTS.md"]))
_register(
    _default(
        "test_memory_guided_behavior_naming_convention",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default("test_memory_influences_file_content", _MEMORY, _BASELINE, memory=["/style/AGENTS.md"])
)
_register(
    _default(
        "test_memory_multiple_sources_combined",
        _MEMORY,
        _BASELINE,
        memory=["/user/AGENTS.md", "/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_with_missing_file_graceful",
        _MEMORY,
        _BASELINE,
        memory=["/missing/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_prevents_unnecessary_file_reads",
        _MEMORY,
        _BASELINE,
        memory=["/docs/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_does_not_persist_transient_info",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_updates_user_formatting_preference",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_missing_file_graceful_without_claiming_context",
        _MEMORY,
        _BASELINE,
        memory=["/missing/AGENTS.md"],
    )
)
_register(
    _builder_eval(
        "test_memory_middleware_composite_backend",
        _MEMORY,
        _BASELINE,
        _composite_backend_builder,
        supports_repl=False,
    )
)
_register(
    _default(
        "test_memory_stale_fact_overridden_by_verified_file",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_adversarial_instruction_does_not_override_user",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_user_explicit_request_overrides_saved_preference",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_conflicting_identity_prefers_current_user",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_memory_investigation_precedes_memory_save_when_required",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)

# --- memory_multiturn (test_memory_multiturn.py) ---------------------------

_register(
    _default(
        "test_implicit_preference_remembered",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default(
        "test_explicit_preference_remembered",
        _MEMORY,
        _BASELINE,
        memory=["/project/AGENTS.md"],
    )
)
_register(
    _default("test_transient_info_not_persisted", _MEMORY, _BASELINE, memory=["/project/AGENTS.md"])
)

# --- skills (test_skills.py) -----------------------------------------------

_UNIT_TEST = "unit_test"

_register(_default("test_read_skill_full_content", _UNIT_TEST, _BASELINE, skills=["/skills/user/"]))
_register(_default("test_read_skill_by_name", _UNIT_TEST, _BASELINE, skills=["/skills/user/"]))
_register(_default("test_combine_two_skills", _UNIT_TEST, _BASELINE, skills=["/skills/user/"]))
_register(
    _default("test_update_skill_typo_fix_no_read", _UNIT_TEST, _BASELINE, skills=["/skills/user/"])
)
_register(
    _default(
        "test_update_skill_typo_fix_requires_read", _UNIT_TEST, _BASELINE, skills=["/skills/user/"]
    )
)
_register(
    _default(
        "test_find_skill_in_correct_path",
        _UNIT_TEST,
        _BASELINE,
        skills=["/skills/base/", "/skills/project/"],
    )
)

# --- todos (test_todos.py) -------------------------------------------------

_TOOL_USE = "tool_use"

_register(_default("test_write_todos_sequential_updates_returns_text", _TOOL_USE, _BASELINE))
_register(_default("test_write_todos_three_steps_returns_text", _TOOL_USE, _BASELINE))

# --- system_prompt (test_system_prompt.py) ---------------------------------

_register(
    _default(
        "test_custom_system_prompt",
        _UNIT_TEST,
        _BASELINE,
        system_prompt="Your name is Foo Bar.",
    )
)

# --- followup_quality (test_followup_quality.py) ---------------------------

_CONVERSATION = "conversation"

_register(_default("test_followup_question_quality", _CONVERSATION, _HILLCLIMB))

# --- subagents (test_subagents.py) -----------------------------------------

_register(
    _builder_eval(
        "test_task_calls_weather_subagent",
        _UNIT_TEST,
        _BASELINE,
        _weather_subagent_builder,
        supports_repl=False,
    )
)
_register(
    _builder_eval(
        "test_task_calls_general_purpose_subagent",
        _UNIT_TEST,
        _BASELINE,
        _general_purpose_subagent_builder,
        supports_repl=False,
    )
)

# --- tool_selection (test_tool_selection.py) --------------------------------

for _name in (
    "test_direct_request_slack_dm",
    "test_direct_request_github_pr",
    "test_direct_request_multiple_tools",
    "test_indirect_schedule_meeting",
    "test_indirect_notify_team",
    "test_indirect_email_report",
    "test_chain_search_then_email",
    "test_chain_create_issue_then_notify",
):
    _tier = (
        _BASELINE
        if _name
        in (
            "test_direct_request_multiple_tools",
            "test_indirect_schedule_meeting",
            "test_chain_search_then_email",
        )
        else _HILLCLIMB
    )
    _register(
        EvalSpec(
            name=_name,
            category=_TOOL_USE,
            tier=_tier,
            tools=TOOL_SELECTION_TOOLS,
        )
    )

# --- tool_usage_relational (test_tool_usage_relational.py) -----------------

for _name in (
    "test_single_tool_list_user_ids",
    "test_single_tool_get_user_email",
    "test_single_tool_get_food_calories",
    "test_two_tools_user_name_from_current_id",
    "test_two_tools_city_for_user",
    "test_two_tools_find_user_then_email",
    "test_three_tools_current_user_city",
    "test_three_tools_find_user_then_city",
    "test_three_tools_current_user_weather",
    "test_four_tools_current_user_favorite_food_names",
    "test_four_tools_find_user_food_name_and_calories",
    "test_four_tools_current_user_location_time_and_weather",
    "test_five_steps_current_user_food_names_and_calories",
    "test_four_steps_find_user_city_and_weather",
    "test_four_steps_find_user_food_allergies",
    "test_four_steps_current_user_food_names_calories_and_allergies",
    "test_four_steps_find_user_city_weather_time_and_food_details",
    "test_four_steps_find_user_email_city_foods_calories_and_allergies",
):
    _register(_builder_eval(_name, _TOOL_USE, _BASELINE, _relational_builder, supports_repl=True))

# --- tool_usage_incident_graph (test_tool_usage_incident_graph.py) ----------

for _name in (
    "test_single_tool_list_incident_ids",
    "test_two_tools_current_incident_service_name",
    "test_three_tools_find_service_owner_team",
    "test_multi_question_current_incident_service_and_incident_oncall",
    "test_multi_question_incident_oncall_and_incident_environment",
    "test_multi_question_incident_oncall_and_service_with_most_firing_alerts",
    "test_multi_question_three_independent_simple_lookups",
    "test_four_tools_incident_to_oncall_name",
    "test_four_tools_service_runbook_url",
    "test_five_tools_incident_latest_deploy_and_repo",
    "test_five_tools_incident_environment_name_and_region",
    "test_five_tools_service_dependency_names_parallel",
    "test_five_tools_service_alert_names_parallel",
    "test_six_tools_current_incident_oncall_name_and_email",
    "test_six_tools_service_repo_and_branch",
    "test_six_tools_incident_title_severity_and_status",
    "test_six_tools_current_incident_metrics_parallel",
    "test_aggregation_active_incident_count_by_team",
    "test_comparison_active_incident_most_dependencies",
    "test_latest_selection_active_incident_most_recent_deploy",
    "test_metric_ranking_active_incident_highest_latency",
    "test_alert_aggregation_service_with_most_firing_alerts",
    "test_dependency_reasoning_active_incident_depending_on_identity_api",
):
    _register(
        _builder_eval(_name, _TOOL_USE, _BASELINE, _incident_graph_builder, supports_repl=True)
    )

# --- summarization (test_summarization.py) ---------------------------------

_SUMMARIZATION = "summarization"

_register(
    _builder_eval(
        "test_summarize_continues_task",
        _SUMMARIZATION,
        _BASELINE,
        _make_summarization_builder(max_input_tokens=15_000, include_compact_tool=False),
        supports_repl=False,
    )
)
_register(
    _builder_eval(
        "test_summarization_offloads_to_filesystem",
        _SUMMARIZATION,
        _BASELINE,
        _make_summarization_builder(max_input_tokens=15_000, include_compact_tool=False),
        supports_repl=False,
    )
)
_register(
    _builder_eval(
        "test_compact_tool_new_task",
        _SUMMARIZATION,
        _BASELINE,
        _make_summarization_builder(max_input_tokens=35_000, include_compact_tool=True),
        supports_repl=False,
    )
)
_register(
    _builder_eval(
        "test_compact_tool_not_overly_sensitive",
        _SUMMARIZATION,
        _HILLCLIMB,
        _make_summarization_builder(max_input_tokens=35_000, include_compact_tool=True),
        supports_repl=False,
    )
)
_register(
    _builder_eval(
        "test_compact_tool_large_reads",
        _SUMMARIZATION,
        _HILLCLIMB,
        _make_summarization_builder(
            max_input_tokens=35_000, include_compact_tool=True, run_limit=3
        ),
        supports_repl=False,
    )
)
