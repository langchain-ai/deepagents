from __future__ import annotations

import functools
from typing import Any

import pytest

from langchain_quickjs import CodeInterpreterMiddleware


@pytest.fixture(autouse=True, params=["worker", "inline"])
def execution(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every unit test under both execution modes.

    Middleware built without an explicit `execution` gets this run's mode.
    Tests marked `worker_only` check worker threads and skip under inline.
    """
    mode = request.param
    if mode == "inline" and request.node.get_closest_marker("worker_only"):
        pytest.skip("checks worker threads")
    init = CodeInterpreterMiddleware.__init__

    @functools.wraps(init)
    def init_with_mode(
        self: CodeInterpreterMiddleware, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault("execution", mode)
        init(self, *args, **kwargs)

    monkeypatch.setattr(CodeInterpreterMiddleware, "__init__", init_with_mode)
    return mode
