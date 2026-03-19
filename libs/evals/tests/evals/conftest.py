from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from deepagents import __version__ as deepagents_version
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
        help="Run only evals tagged with this category (repeatable). E.g. --eval-category memory --eval-category hitl",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    categories = config.getoption("--eval-category")
    if not categories:
        return

    known = {
        m.args[0] for item in items if (m := item.get_closest_marker("eval_category")) and m.args
    }
    unknown = set(categories) - known
    if unknown:
        msg = (
            f"Unknown --eval-category values: {sorted(unknown)}. "
            f"Known categories in collected tests: {sorted(known)}"
        )
        pytest.exit(msg, returncode=1)

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        marker = item.get_closest_marker("eval_category")
        if marker and marker.args and marker.args[0] in categories:
            selected.append(item)
        else:
            deselected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "model_name" not in metafunc.fixturenames:
        return

    model_opt = metafunc.config.getoption("--model")
    model_name = model_opt or str(get_default_model().model)
    metafunc.parametrize("model_name", [model_name])


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
    return {
        "model": model_name,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "deepagents_version": deepagents_version,
    }


@pytest.fixture
def model(model_name: str) -> BaseChatModel:
    return init_chat_model(model_name)
