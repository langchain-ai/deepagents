from __future__ import annotations

import cProfile
import io
import pstats
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from deepagents.graph import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fake_chat_model import GenericFakeChatModel
from langchain_repl import ReplMiddleware

_EXPRESSION = "print(1 + 1)"
_ITERATIONS = 5
_PARALLEL_WORKERS = 20
_TOP_N = 60
_OUTPUT = Path("/tmp/repl-parallel-middleware.prof")


def _script(code: str, *, final_message: str = "Done.") -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "repl",
                        "args": {"code": code},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=final_message),
        ]
    )


def _make_agent(code: str, middleware: ReplMiddleware) -> Any:
    return create_deep_agent(
        model=GenericFakeChatModel(
            messages=_script(code, final_message="The answer is 2.")
        ),
        middleware=[middleware],
    )


def _invoke_once(middleware: ReplMiddleware) -> None:
    result = _make_agent(_EXPRESSION, middleware).invoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 1 + 1")]}
    )
    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["2"]
    assert result["messages"][-1].content == "The answer is 2."


def _run_parallel_batch() -> None:
    shared_middleware = ReplMiddleware()
    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
        list(
            executor.map(
                lambda _: _invoke_once(shared_middleware),
                range(_PARALLEL_WORKERS),
            )
        )


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(_ITERATIONS):
        _run_parallel_batch()
    profiler.disable()
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(_OUTPUT))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(_TOP_N)
    print(
        f"profiled {_ITERATIONS} parallel batches with {_PARALLEL_WORKERS} workers each"
    )
    print(f"wrote profile to {_OUTPUT}")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
