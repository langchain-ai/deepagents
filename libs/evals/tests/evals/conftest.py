from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from deepagents import TodoMode, __version__ as deepagents_version
from deepagents.graph import get_default_model

pytest_plugins = ["tests.evals.pytest_reporter"]


def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks and fail fast if LangSmith tracing is not enabled.

    All eval tests require `@pytest.mark.langsmith` and
    `LANGSMITH_TRACING=true`. Detect this early so the entire suite is skipped
    with a clear message instead of failing one-by-one.
    """
    config.addinivalue_line(
        "markers",
        "eval_category(name): tag an eval test with a category for grouping and reporting",
    )
    config.addinivalue_line(
        "markers",
        "eval_tier(name): tag an eval as 'baseline' (regression gate) or 'hillclimb' (progress tracking)",
    )

    tracing_enabled = any(
        os.environ.get(var, "").lower() == "true"
        for var in (
            "LANGSMITH_TRACING_V2",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING",
        )
    )
    if not tracing_enabled:
        pytest.exit(
            "Aborting: LangSmith tracing is not enabled. "
            "All eval tests require LangSmith tracing. "
            "Set one of LANGSMITH_TRACING / LANGSMITH_TRACING_V2 / "
            "LANGCHAIN_TRACING_V2 to 'true' and ensure a valid "
            "LANGSMITH_API_KEY is set, then re-run.",
            returncode=1,
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model to run evals against. If omitted, uses deepagents.graph.get_default_model().model.",
    )
    parser.addoption(
        "--eval-category",
        action="append",
        default=[],
        help="Run only evals tagged with this category (repeatable). E.g. --eval-category memory --eval-category tool_use",
    )
    parser.addoption(
        "--eval-tier",
        action="append",
        default=[],
        help="Run only evals tagged with this tier (repeatable). E.g. --eval-tier baseline --eval-tier hillclimb",
    )
    parser.addoption(
        "--openrouter-provider",
        action="store",
        default=None,
        help="Pin OpenRouter to a specific provider. E.g. --openrouter-provider MiniMax",
    )
    parser.addoption(
        "--todo-mode",
        action="store",
        default="tool",
        choices=["tool", "prompt", "filesystem"],
        help="Planning mode: 'tool' (TodoListMiddleware), 'prompt' (planning guidance), 'filesystem' (PLAN.md).",
    )
    parser.addoption(
        "--todo-variants",
        action="store",
        default=None,
        help=(
            "Run each test across multiple todo-mode variants in a single session. "
            "Comma-separated list (e.g. 'tool,prompt,filesystem') or 'all' for every variant. "
            "Overrides --todo-mode when set."
        ),
    )
    parser.addoption(
        "--trials",
        action="store",
        default=1,
        type=int,
        help="Number of times to repeat each test (default: 1). Useful for measuring eval variance.",
    )
    parser.addoption(
        "--include-todos",
        action="store",
        default=None,
        choices=["true", "false"],
        help="Deprecated. Use --todo-mode instead.",
    )


def _filter_by_marker(
    config: pytest.Config,
    items: list[pytest.Item],
    *,
    option: str,
    marker_name: str,
) -> None:
    """Deselect items whose *marker_name* value is not in the CLI *option* list.

    Exits the test session with returncode 1 if any requested values are not
    found among collected tests.

    Args:
        config: The pytest config object.
        items: Mutable list of collected test items (modified in-place).
        option: CLI option name (e.g. `--eval-category`).
        marker_name: Pytest marker to read (e.g. `eval_category`).
    """
    values = config.getoption(option)
    if not values:
        return

    known = {m.args[0] for item in items if (m := item.get_closest_marker(marker_name)) and m.args}
    unknown = set(values) - known
    if unknown:
        msg = (
            f"Unknown {option} values: {sorted(unknown)}. "
            f"Known values in collected tests: {sorted(known)}"
        )
        pytest.exit(msg, returncode=1)

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        marker = item.get_closest_marker(marker_name)
        if marker and marker.args and marker.args[0] in values:
            selected.append(item)
        else:
            deselected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _filter_by_marker(config, items, option="--eval-category", marker_name="eval_category")
    _filter_by_marker(config, items, option="--eval-tier", marker_name="eval_tier")


_ALL_TODO_VARIANTS: list[TodoMode] = ["tool", "prompt", "filesystem"]


def _get_todo_variants(config: pytest.Config) -> list[TodoMode]:
    """Resolve the list of todo-mode variants to parametrize over.

    Returns a single-element list when ``--todo-variants`` is not set (the
    resolved ``--todo-mode`` / ``--include-todos`` value), or the expanded
    variant list otherwise.
    """
    raw = config.getoption("--todo-variants")
    if raw is None:
        return [_resolve_todo_mode_option(config)]
    if raw.strip().lower() == "all":
        return list(_ALL_TODO_VARIANTS)
    variants: list[TodoMode] = []
    for part in raw.split(","):
        cleaned = part.strip()
        if cleaned not in ("tool", "prompt", "filesystem"):
            msg = f"Invalid --todo-variants value: {cleaned!r}. Must be 'tool', 'prompt', 'filesystem', or 'all'."
            raise pytest.UsageError(msg)
        variants.append(cleaned)
    return variants


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "model_name" not in metafunc.fixturenames:
        return

    model_opt = metafunc.config.getoption("--model")
    model_name = model_opt or str(get_default_model().model)
    trials: int = metafunc.config.getoption("--trials")

    if trials > 1:
        values = [model_name] * trials
        ids = [f"{model_name}-t{i}" for i in range(1, trials + 1)]
        metafunc.parametrize("model_name", values, ids=ids)
    else:
        metafunc.parametrize("model_name", [model_name])

    if "todo_mode" in metafunc.fixturenames:
        variants = _get_todo_variants(metafunc.config)
        if len(variants) > 1 or metafunc.config.getoption("--todo-variants") is not None:
            metafunc.parametrize("todo_mode", variants)


@pytest.fixture
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="session")
def langsmith_experiment_metadata(request: pytest.FixtureRequest) -> dict[str, Any]:
    model_opt = request.config.getoption("--model")
    default_model = get_default_model()
    model_name = model_opt or str(
        getattr(default_model, "model", None) or getattr(default_model, "model_name", "")
    )
    variants = _get_todo_variants(request.config)
    trials: int = request.config.getoption("--trials")
    return {
        "model": model_name,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "deepagents_version": deepagents_version,
        "todo_variants": variants,
        "trials": trials,
    }


@pytest.fixture
def model(model_name: str, request: pytest.FixtureRequest) -> BaseChatModel:
    kwargs: dict[str, Any] = {}
    provider = request.config.getoption("--openrouter-provider")
    if provider:
        if not model_name.startswith("openrouter:"):
            msg = "--openrouter-provider requires an openrouter: model prefix"
            raise ValueError(msg)
        kwargs["openrouter_provider"] = {
            "only": [provider],
            "allow_fallbacks": False,
        }
    if model_name.startswith("openrouter:"):
        # OpenRouter SDK passes timeout=None to httpx, disabling its default
        # 5s read timeout. This causes indefinite hangs on TCP stalls.
        # See: https://github.com/OpenRouterTeam/python-sdk/issues/72
        kwargs["timeout"] = 120_000  # ms
    return init_chat_model(model_name, **kwargs)


def _resolve_todo_mode_option(config: pytest.Config) -> TodoMode:
    """Resolve ``--todo-mode`` / ``--include-todos`` into a `TodoMode` value."""
    include_todos = config.getoption("--include-todos")
    todo_mode = config.getoption("--todo-mode")
    if include_todos is not None:
        return "tool" if include_todos == "true" else "prompt"
    return todo_mode


@pytest.fixture
def todo_mode(request: pytest.FixtureRequest) -> TodoMode:
    """The planning mode to use in `create_deep_agent` calls.

    When ``--todo-variants`` is active, the value comes from parametrization.
    Otherwise falls back to ``--todo-mode`` / ``--include-todos``.
    """
    if hasattr(request, "param"):
        return request.param
    return _resolve_todo_mode_option(request.config)


@pytest.fixture
def include_todos(request: pytest.FixtureRequest) -> bool:
    """Deprecated. Use ``todo_mode`` fixture instead."""
    return _resolve_todo_mode_option(request.config) == "tool"
