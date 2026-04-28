from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from deepagents.graph import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage

from langchain_repl import ReplMiddleware
from tests.benchmarks._fake_chat_model import GenericFakeChatModel

_EXPRESSION = "print(1 + 1)"
_DEFAULT_ITERATIONS = 50
_DEFAULT_TOP_N = 40
_DEFAULT_OUTPUT = Path("/tmp/repl-middleware.prof")
_DEFAULT_PARALLEL_WORKERS = 1


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


def _make_agent(code: str) -> Any:
    return create_deep_agent(
        model=GenericFakeChatModel(
            messages=_script(code, final_message="The answer is 2.")
        ),
        middleware=[ReplMiddleware()],
    )


def _run_once() -> None:
    result = _make_agent(_EXPRESSION).invoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 1 + 1")]}
    )
    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["2"]
    assert result["messages"][-1].content == "The answer is 2."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=_DEFAULT_ITERATIONS)
    parser.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.iterations):
        _run_once()
    profiler.disable()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(args.output))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(args.top_n)
    print(f"profiled {args.iterations} end-to-end middleware invocations")
    print(f"wrote profile to {args.output}")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
