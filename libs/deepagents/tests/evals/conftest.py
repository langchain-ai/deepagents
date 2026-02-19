from __future__ import annotations

import pytest

from deepagents.graph import get_default_model


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="append",
        default=None,
        help=("Model(s) to run evals against. May be provided multiple times. If omitted, uses deepagents.graph.get_default_model().model."),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "model" not in metafunc.fixturenames:
        return

    models_opt = metafunc.config.getoption("--model")
    models = models_opt or [str(get_default_model().model)]
    metafunc.parametrize("model", models)


@pytest.fixture
def model(request: pytest.FixtureRequest) -> str:
    return str(request.param)
