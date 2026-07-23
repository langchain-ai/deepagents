"""CPU/latency profiler for a deep agent with MANY OpenAI tools + summarization.

Sibling of `profile_long_agent.py` (which uses a ChatAnthropic mock). This one
builds a real `create_deep_agent` with a **ChatOpenAI** model and a large pool of
tools with rich Pydantic schemas, so the OpenAI tool-schema-generation path
(`convert_to_openai_tool`) is exercised on the *real* code paths:

  1. `bind_tools` (once) — cached via `_cached_openai_tool_schema`.
  2. SummarizationMiddleware `wrap_model_call` -> `_count_tokens(..., tools=...)`
     -> `count_tokens_approximately(messages, tools=tools)` -> `convert_to_openai_tool`
     for EVERY tool, on EVERY turn, UNCACHED. This is the suspected hot path.

By default the model is a mock subclass of ChatOpenAI (no network) so the CPU
flamegraph isolates client-side schema-gen instead of HTTP latency. Pass
`--real` to hit the actual OpenAI API.

Examples:

    # 80 tools, no network, summarization fires, CPU flamegraph:
    uv run --project libs/deepagents --with pyinstrument profile_openai_agent.py \
        --tools 80 --turns 40 -o flame_openai.html

    # same, with LangSmith tracing on (needs LANGSMITH_API_KEY):
    uv run --project libs/deepagents --with pyinstrument profile_openai_agent.py \
        --tools 80 --turns 40 --trace -o flame_openai_trace.html

Open the .html (interactive) or load the sibling .speedscope.json at
https://speedscope.app for sandwich/left-heavy views.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--turns", type=int, default=40, help="Number of user turns (message history length).")
    p.add_argument("--tools", type=int, default=80, help="Size of the tool pool (stresses OpenAI schema conversion).")
    p.add_argument("--calls-per-turn", type=int, default=3, help="Tool calls the model makes each turn.")
    p.add_argument("--payload-kb", type=int, default=8, help="Approx KB returned by each tool call.")
    p.add_argument("--context-window", type=int, default=32000,
                   help="Override model max_input_tokens so summarization fires quickly (mock only).")
    p.add_argument("--real", action="store_true", help="Use the real OpenAI API instead of the mock model.")
    p.add_argument("--model", default="gpt-4o", help="OpenAI model name.")
    p.add_argument("--trace", dest="trace", action="store_true", help="Enable LangSmith tracing.")
    p.add_argument("--no-trace", dest="trace", action="store_false", help="Disable LangSmith tracing (default).")
    p.set_defaults(trace=False)
    p.add_argument("--project", default="deepagents-openai-schema-profile", help="LangSmith project name when --trace.")
    p.add_argument("--profiler", choices=["pyinstrument", "none"], default="pyinstrument")
    p.add_argument("-o", "--output", default="flame_openai.html", help="Flamegraph HTML output path.")
    args = p.parse_args()
    # Bounded clamps (Corridor: keep memory/CPU bounded; validate up front).
    args.turns = max(1, min(args.turns, 2000))
    args.tools = max(1, min(args.tools, 500))
    args.payload_kb = max(0, min(args.payload_kb, 1000))
    args.calls_per_turn = max(1, min(args.calls_per_turn, 50))
    args.context_window = max(4000, min(args.context_window, 2_000_000))
    return args


# ─── LangSmith wiring (must happen before agent import/invoke) ─────────────────
def configure_langsmith(enabled: bool, project: str) -> None:
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("WARNING: --trace set but LANGSMITH_API_KEY is not in the environment; "
              "traces will not be uploaded.", file=sys.stderr)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
    print(f"LangSmith tracing ON -> project '{project}'")


# ─── Tool pool with rich schemas ──────────────────────────────────────────────
def build_tools(n_tools: int, payload_kb: int) -> tuple[list, list[str]]:
    """Create `n_tools` StructuredTools with non-trivial Pydantic arg schemas.

    Rich schemas (enums, lists, nested defaults, descriptions) make the
    tool->OpenAI-JSON-schema conversion visible in the flamegraph.
    """
    from enum import Enum

    from langchain_core.tools import StructuredTool

    payload = "x" * (payload_kb * 1024)

    class Mode(str, Enum):
        semantic = "semantic"
        lexical = "lexical"
        hybrid = "hybrid"

    class Filter(BaseModel):
        field: str = Field(description="Field to filter on.")
        op: str = Field(default="eq", description="One of: eq, ne, gt, lt, contains.")
        value: str = Field(description="Value to compare against.")

    class SearchArgs(BaseModel):
        query: str = Field(description="Natural-language search query.")
        limit: int = Field(default=10, ge=1, le=100, description="Max results to return.")
        mode: Mode = Field(default=Mode.semantic, description="Search mode.")
        include_globs: list[str] = Field(default_factory=list, description="Glob patterns to include.")
        exclude_globs: list[str] = Field(default_factory=list, description="Glob patterns to exclude.")
        filters: list[Filter] = Field(default_factory=list, description="Structured filters (nested schema).")
        depth: int = Field(default=1, ge=0, le=10, description="Traversal depth.")
        verbose: bool = Field(default=False, description="Whether to return verbose metadata.")

    tools: list = []
    names: list[str] = []
    for i in range(n_tools):
        name = f"query_subsystem_{i:03d}"
        names.append(name)

        def _make(idx: int):
            def _fn(
                query: str,
                limit: int = 10,
                mode: str = "semantic",
                include_globs: list[str] | None = None,
                exclude_globs: list[str] | None = None,
                filters: list[dict] | None = None,
                depth: int = 1,
                verbose: bool = False,
            ) -> str:
                return f"[tool {idx}] results for '{query}' (mode={mode}, limit={limit}):\n{payload}"
            return _fn

        tools.append(
            StructuredTool.from_function(
                func=_make(i),
                name=name,
                description=f"Search subsystem #{i} for code/config matching a query. "
                            f"Supports filtering, traversal depth, and verbose metadata.",
                args_schema=SearchArgs,
            )
        )
    return tools, names


# ─── Model ───────────────────────────────────────────────────────────────────
def build_model(model_name: str, tool_names: list[str], calls_per_turn: int, real: bool, context_window: int):
    from langchain_openai import ChatOpenAI

    if real:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: --real requires OPENAI_API_KEY in the environment.", file=sys.stderr)
            sys.exit(1)
        return ChatOpenAI(model=model_name, temperature=0)

    # Mock: exercise real ChatOpenAI schema/bind paths but skip the network.
    os.environ.setdefault("OPENAI_API_KEY", "sk-mock-not-used")

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    call_counter = itertools.count()

    class MockChatOpenAI(ChatOpenAI):
        """Deterministic tool-call loop; subclasses ChatOpenAI for real bind_tools/schema paths."""

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            # Distinguish the agent tool-loop from the summarization sub-call:
            # agent human turns start with "Investigate subsystem batch".
            last_human = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)),
                None,
            )
            is_agent_turn = bool(
                last_human is not None
                and isinstance(last_human.content, str)
                and last_human.content.startswith("Investigate subsystem batch")
            )
            if not is_agent_turn:
                # Summarization model call — return plain summary text.
                msg = AIMessage(content="Condensed summary of prior investigation steps.")
                return ChatResult(generations=[ChatGeneration(message=msg)])

            last_h_idx = max(
                (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
                default=0,
            )
            done = sum(1 for m in messages[last_h_idx:] if isinstance(m, ToolMessage))
            if done < calls_per_turn:
                seq = next(call_counter)
                name = tool_names[seq % len(tool_names)]
                msg = AIMessage(
                    content="",
                    tool_calls=[{
                        "id": f"call_{seq}",
                        "name": name,
                        "args": {"query": f"topic {seq}", "limit": 5, "mode": "hybrid",
                                 "include_globs": ["**/*.py"], "depth": 2, "verbose": True},
                    }],
                )
            else:
                msg = AIMessage(content="Turn complete; summarized findings.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    model = MockChatOpenAI(model=model_name, temperature=0)
    # Shrink the context window so fraction-based summarization fires quickly.
    try:
        profile = dict(model.profile or {})
        profile["max_input_tokens"] = context_window
        model.profile = profile  # type: ignore[assignment]
        print(f"Mock model context window set to {context_window} tokens (summarization trigger ~= {int(context_window*0.85)}).")
    except Exception as e:  # noqa: BLE001
        print(f"NOTE: could not override model.profile ({type(e).__name__}: {e}); "
              "using real profile — summarization may need more turns.", file=sys.stderr)
    return model


# ─── Agent build ───────────────────────────────────────────────────────────────
def build_agent(model, tools):
    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent

    return create_deep_agent(model=model, tools=tools, checkpointer=InMemorySaver())


# ─── Run loop ────────────────────────────────────────────────────────────────
def run_workload(agent, turns: int) -> dict:
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "openai-schema-profile"}, "recursion_limit": 1000}
    per_turn: list[float] = []
    result = None
    for i in range(turns):
        t0 = time.perf_counter()
        result = agent.invoke(
            {"messages": [HumanMessage(content=f"Investigate subsystem batch {i}.")]},
            config,
        )
        per_turn.append(time.perf_counter() - t0)

    msgs = result["messages"] if result else []
    n_summaries = sum(
        1 for m in msgs
        if getattr(m, "additional_kwargs", {}).get("lc_source") == "summarization"
    )
    return {"per_turn": per_turn, "n_messages": len(msgs), "n_summaries": n_summaries}


def print_summary(args, stats: dict, wall: float) -> None:
    per_turn = stats["per_turn"]
    n = len(per_turn)
    first5 = sum(per_turn[:5]) / min(5, n)
    last5 = sum(per_turn[-5:]) / min(5, n)
    print("\n" + "=" * 60)
    print("LATENCY SUMMARY (OpenAI schema-gen profile)")
    print("=" * 60)
    print(f"model              : {args.model}  ({'REAL API' if args.real else 'mock (no network)'})")
    print(f"turns              : {args.turns}")
    print(f"tool pool          : {args.tools}   calls/turn: {args.calls_per_turn}   payload: {args.payload_kb}KB")
    print(f"tracing            : {'ON -> ' + args.project if args.trace else 'off'}")
    print(f"total wall time    : {wall:.2f}s")
    print(f"mean turn latency  : {sum(per_turn) / n * 1000:.1f}ms")
    print(f"first-5 avg turn   : {first5 * 1000:.1f}ms")
    print(f"last-5 avg turn    : {last5 * 1000:.1f}ms   (growth: {last5 / first5:.2f}x)")
    print(f"final msg count    : {stats['n_messages']}")
    print(f"summarization events (approx, summary msgs in final state): {stats['n_summaries']}")
    print("=" * 60)


# ─── Profiler wrapper ────────────────────────────────────────────────────────
def run_with_pyinstrument(fn, html_path: str):
    try:
        from pyinstrument import Profiler
        from pyinstrument.renderers import SpeedscopeRenderer
    except ImportError:
        print("pyinstrument not installed. Re-run with:\n"
              "  uv run --project libs/deepagents --with pyinstrument profile_openai_agent.py ...",
              file=sys.stderr)
        sys.exit(1)

    profiler = Profiler(interval=0.001, async_mode="disabled")
    profiler.start()
    result = fn()
    profiler.stop()

    with open(html_path, "w") as f:
        f.write(profiler.output_html())
    speedscope_path = os.path.splitext(html_path)[0] + ".speedscope.json"
    with open(speedscope_path, "w") as f:
        f.write(profiler.output(SpeedscopeRenderer()))
    print(f"\nFlamegraph written:\n  HTML       : {os.path.abspath(html_path)}\n"
          f"  speedscope : {os.path.abspath(speedscope_path)}  (load at https://speedscope.app)")
    print("\n--- hottest call tree (console) ---")
    print(profiler.output_text(unicode=True, color=False, show_all=False))
    return result


def main() -> None:
    args = parse_args()
    configure_langsmith(args.trace, args.project)

    print(f"Building agent: {args.tools} tools, model={args.model} "
          f"({'real' if args.real else 'mock'}) ...")
    tools, tool_names = build_tools(args.tools, args.payload_kb)
    model = build_model(args.model, tool_names, args.calls_per_turn, args.real, args.context_window)
    agent = build_agent(model, tools)

    print(f"Running {args.turns} turns (profiler={args.profiler}) ...")

    def _run():
        t0 = time.perf_counter()
        stats = run_workload(agent, args.turns)
        return stats, time.perf_counter() - t0

    if args.profiler == "pyinstrument":
        stats, wall = run_with_pyinstrument(_run, args.output)
    else:
        stats, wall = _run()

    print_summary(args, stats, wall)
    if args.trace:
        print("\nView traces in LangSmith under project:", args.project)


if __name__ == "__main__":
    main()
