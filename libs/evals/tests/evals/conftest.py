from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import pytest
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from deepagents import __version__ as deepagents_version

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
    config.addinivalue_line(
        "markers",
        "repl(*allowed): declare optional REPL backends allowed for a test/module; used with --repl quickjs|langchain",
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

    if not config.getoption("--model"):
        pytest.exit(
            "Aborting: --model is required. Pass an explicit model identifier, "
            "e.g. `--model claude-sonnet-4-6`.",
            returncode=1,
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model to run evals against (required). E.g. --model claude-sonnet-4-6.",
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
        "--openai-reasoning-effort",
        action="store",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default=None,
        help="Apply reasoning effort to OpenAI models. E.g. --openai-reasoning-effort high",
    )
    parser.addoption(
        "--repl",
        action="store",
        choices=("quickjs", "langchain"),
        default=None,
        help="Optional REPL middleware for tests marked with @pytest.mark.repl. If omitted, those tests bind their tools directly instead of routing through a REPL.",
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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "model_name" not in metafunc.fixturenames:
        return

    model_name = metafunc.config.getoption("--model")
    metafunc.parametrize("model_name", [model_name])


@pytest.fixture
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


ReplName = Literal["quickjs", "langchain"]


@pytest.fixture
def repl_name(request: pytest.FixtureRequest) -> ReplName | None:
    marker = request.node.get_closest_marker("repl")
    selected = request.config.getoption("--repl")
    if selected is None:
        return None
    if marker is None or not marker.args:
        pytest.skip("--repl was provided but this test is not marked with @pytest.mark.repl(...)")
    allowed = tuple(str(arg) for arg in marker.args)
    if selected not in allowed:
        pytest.skip(
            f"--repl={selected} is not supported for this test; allowed values: {', '.join(allowed)}"
        )
    return selected


@pytest.fixture(scope="session")
def langsmith_experiment_metadata(request: pytest.FixtureRequest) -> dict[str, Any]:
    return {
        "model": request.config.getoption("--model"),
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "deepagents_version": deepagents_version,
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
    if model_name.startswith("openai:"):
        # Match the SDK's built-in `openai` provider profile, which sets
        # `use_responses_api=True` for all openai: models. The fixture
        # pre-builds the model so the profile layer doesn't apply
        # automatically — mirror it explicitly. Also required for
        # `reasoning_effort` + function tools, which OpenAI gates to
        # /v1/responses for gpt-5.x.
        kwargs["use_responses_api"] = True
    reasoning_effort = request.config.getoption("--openai-reasoning-effort")
    if reasoning_effort:
        if not model_name.startswith("openai:"):
            msg = "--openai-reasoning-effort requires an openai: model prefix"
            raise ValueError(msg)
        kwargs["reasoning_effort"] = reasoning_effort
    return init_chat_model(model_name, **kwargs)
