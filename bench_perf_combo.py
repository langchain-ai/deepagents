"""Benchmark create_deep_agent perf: wall-clock, blob storage, peak memory.

Profiles across message history lengths (N turns). Uses InMemorySaver.
No sys.path hacks — uses whatever langchain/langgraph the current env pins.

Run:
    uv run --project libs/deepagents python bench_perf_combo.py
"""

from __future__ import annotations

import gc
import itertools
import json
import pathlib
import sys
import time
import tracemalloc
from typing import Any, Optional

from deepagents.graph import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

WRITE_FILE_KB = 30
WRITE_FILES_PER_TURN = 2
LARGE_TOOL_KB = 80

_WRITE_CONTENT = "w" * (WRITE_FILE_KB * 1024)
_LARGE_RESULT = "x" * (LARGE_TOOL_KB * 1024)

_turn_counter = itertools.count()


class _MockModel(ChatAnthropic):
    """Per turn: write WRITE_FILES_PER_TURN files, call external_search, done."""

    model_name: str = "claude-haiku-4-5-20251001"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_human_idx = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
            default=0,
        )
        tms = [m for m in messages[last_human_idx:] if isinstance(m, ToolMessage)]
        writes_done = sum(1 for m in tms if "updated file" in (m.content or "").lower())
        search_done = any("results for" in (m.content or "").lower() for m in tms)

        turn = next(_turn_counter)

        if writes_done < WRITE_FILES_PER_TURN:
            path = f"/workspace/turn_{turn}_file_{writes_done}.txt"
            msg = AIMessage(
                content="",
                tool_calls=[{"id": f"call_w_{turn}_{writes_done}", "name": "write_file",
                             "args": {"file_path": path, "content": _WRITE_CONTENT}}],
            )
        elif not search_done:
            msg = AIMessage(
                content="",
                tool_calls=[{"id": f"call_s_{turn}", "name": "external_search",
                             "args": {"query": f"topic {turn}"}}],
            )
        else:
            msg = AIMessage(content=f"Turn {turn} complete.")

        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def external_search(query: str) -> str:
    """Search external knowledge base. Returns large results."""
    return f"Results for '{query}':\n\n" + _LARGE_RESULT


def _saver_storage_bytes(saver: InMemorySaver, thread_id: str) -> dict[str, int]:
    """Total bytes stored in a saver, broken down by storage region.

    DeltaChannel (combo) stores sentinel blobs + real data in writes.
    Baseline stores full state in blobs per checkpoint.
    """
    blobs_bytes = 0
    for key, blob in saver.blobs.items():
        if key[0] == thread_id and isinstance(blob, tuple) and len(blob) == 2:
            blobs_bytes += len(blob[1])

    writes_bytes = 0
    for (tid, _ns, _cid), step_writes in saver.writes.items():
        if tid != thread_id:
            continue
        for write in step_writes.values():
            # write = (task_id, channel, dumps_typed_tuple, task_path)
            # dumps_typed_tuple = (type_str, value_bytes)
            for el in write:
                if isinstance(el, (bytes, bytearray, memoryview)):
                    writes_bytes += len(el)
                elif isinstance(el, tuple):
                    for inner in el:
                        if isinstance(inner, (bytes, bytearray, memoryview)):
                            writes_bytes += len(inner)

    storage_bytes = 0
    ns_storage = saver.storage.get(thread_id, {})
    for _ns, checkpoints in ns_storage.items():
        for cid, entry in checkpoints.items():
            for el in entry:
                if isinstance(el, (bytes, bytearray, memoryview)):
                    storage_bytes += len(el)
                elif isinstance(el, tuple):
                    for inner in el:
                        if isinstance(inner, (bytes, bytearray, memoryview)):
                            storage_bytes += len(inner)

    return {
        "blobs_mb": round(blobs_bytes / 1024 / 1024, 2),
        "writes_mb": round(writes_bytes / 1024 / 1024, 2),
        "storage_mb": round(storage_bytes / 1024 / 1024, 2),
        "total_mb": round((blobs_bytes + writes_bytes + storage_bytes) / 1024 / 1024, 2),
    }


def run(turns: int, durability: str = "exit") -> dict[str, Any]:
    global _turn_counter
    _turn_counter = itertools.count()

    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=_MockModel(model_name="claude-haiku-4-5-20251001"),
        tools=[external_search],
        checkpointer=checkpointer,
    )

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    config = {"configurable": {"thread_id": "bench-run"}}
    for i in range(turns):
        agent.invoke(
            {"messages": [HumanMessage(content=f"Research topic {i} and write your findings.")]},
            config,
            durability=durability,
        )

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    storage = _saver_storage_bytes(checkpointer, "bench-run")
    return {
        "turns": turns,
        "durability": durability,
        "elapsed_s": round(elapsed, 2),
        "peak_mb": round(peak / 1024 / 1024, 1),
        **storage,
    }


def main() -> None:
    exit_ns = [5, 25, 50, 100, 200]
    async_ns = [5, 25, 50, 100]  # baseline async OOMs past 100
    results = []
    for n in exit_ns:
        print(f"running: N={n} durability=exit", flush=True)
        r = run(turns=n, durability="exit")
        print(f"  -> {r}", flush=True)
        results.append(r)
    for n in async_ns:
        print(f"running: N={n} durability=async", flush=True)
        r = run(turns=n, durability="async")
        print(f"  -> {r}", flush=True)
        results.append(r)

    out_path = pathlib.Path("bench_results.json")
    # versions for provenance
    try:
        import langchain
        import langchain_core
        import langgraph
        versions = {
            "langchain": langchain.__version__,
            "langchain_core": langchain_core.__version__,
            "langgraph": langgraph.__version__,
        }
    except Exception as e:  # noqa: BLE001
        versions = {"error": str(e)}

    out = {"versions": versions, "results": results}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
