from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

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
    if "model_name" not in metafunc.fixturenames:
        return

    model_opt = metafunc.config.getoption("--model")
    model_name = model_opt or str(get_default_model().model)
    metafunc.parametrize("model_name", [model_name])


@pytest.fixture
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def model(model_name: str) -> BaseChatModel:
    if model_name == "nvidia/nemotron-3-super-120b-a12b":
        return ChatOpenAI(
            model="private/nvidia/nemotron-3-super-120b-a12b",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
    if model_name.startswith("baseten:"):
        return ChatOpenAI(
            model=model_name.removeprefix("baseten:"),
            base_url="https://inference.baseten.co/v1",
            api_key=os.environ["BASETEN_API_KEY"],
        )
    return init_chat_model(model_name)
