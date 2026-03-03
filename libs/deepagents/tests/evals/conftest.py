from __future__ import annotations

import pytest

from deepagents.graph import get_default_model

pytest_plugins = ["tests.evals.pytest_reporter"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model to run evals against. If omitted, uses deepagents.graph.get_default_model().model.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "model" not in metafunc.fixturenames:
        return

    model_opt = metafunc.config.getoption("--model")
    model = model_opt or str(get_default_model().model)
    metafunc.parametrize("model", [model])


@pytest.fixture
def model(request: pytest.FixtureRequest) -> str:
    return str(request.param)
