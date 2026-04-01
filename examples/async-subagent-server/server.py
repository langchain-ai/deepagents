"""Async Subagent Server — Agent Protocol over FastAPI.

A minimal self-hosted Agent Protocol server that exposes a DeepAgents
researcher as an async subagent. Any DeepAgents supervisor can connect
to this server using the AsyncSubAgent configuration.

Implements the endpoints the DeepAgents async subagent middleware calls
(via the LangGraph SDK):

    POST /threads                              create a thread
    POST /threads/{thread_id}/runs             start (or interrupt+restart) a run
    GET  /threads/{thread_id}/runs/{run_id}    poll run status
    GET  /threads/{thread_id}                  fetch thread (values.messages used on success)
    POST /threads/{thread_id}/runs/{run_id}/cancel  cancel a run
    GET  /ok                                   health check

Persistence uses an in-memory SQLite database (no files, no setup required).
The schema is created automatically on startup.

Run:
    ANTHROPIC_API_KEY=... uvicorn server:app --port 2024

Then point a DeepAgents supervisor at:
    RESEARCHER_URL=http://localhost:2024
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv(Path(__file__).parent / ".env")

# ── Database ──────────────────────────────────────────────────────────────────

# In-memory SQLite shared across all connections in this process.
_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.row_factory = sqlite3.Row


def _init_db() -> None:
    """Create the threads and runs tables if they don't already exist.

    threads — one row per conversation thread
        messages  JSON array of {role, content} objects
        values    JSON object stored as the thread's final state (values.messages)

    runs    — one row per run attempt on a thread
        status    one of: pending | running | success | error | cancelled
    """
    _conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id  TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            messages   TEXT NOT NULL DEFAULT '[]',
            values_    TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            thread_id    TEXT NOT NULL REFERENCES threads(thread_id),
            assistant_id TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            error        TEXT
        );
    """)
    _conn.commit()


# ── DB helpers ────────────────────────────────────────────────────────────────

import json  # noqa: E402  (after stdlib, before third-party)


def _get_thread(thread_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT thread_id, created_at, messages, values_ FROM threads WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "thread_id": row["thread_id"],
        "created_at": row["created_at"],
        "messages": json.loads(row["messages"]),
        "values": json.loads(row["values_"]),
    }


def _get_run(run_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT run_id, thread_id, assistant_id, status, created_at, error FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


# ── Agent ─────────────────────────────────────────────────────────────────────
#
# Replace this with your own agent. The only requirement is that it accepts
# a messages array and returns an object with a messages array.

import os  # noqa: E402


@tool
async def web_search(query: str) -> str:
    """Search the web for information. Use this to find current data, news, and analysis.

    Args:
        query: The search query.
    """
    if os.environ.get("TAVILY_API_KEY"):
        import httpx

        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": 5},
                timeout=30,
            )
        data = res.json()
        results = data.get("results") or []
        if not results:
            return f'No results for "{query}"'
        return "\n\n".join(
            f"{i + 1}. **{r['title']}**\n   {r['content']}\n   Source: {r['url']}"
            for i, r in enumerate(results)
        )

    # Stub search — replace with a real search API or remove this branch.
    return "\n".join([
        f'[stub] Search results for "{query}":',
        f"1. Key finding: Recent developments show significant progress in {query}",
        f"2. Expert analysis: Industry leaders are investing heavily in {query}",
        f"3. Market data: The {query} sector has seen notable activity this quarter",
    ])


from deepagents import create_deep_agent  # noqa: E402

_agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5"),
    system_prompt=(
        "You are a thorough research agent. Investigate topics using web search and produce "
        "a well-structured research summary (300–500 words). Cite sources where possible.\n\n"
        "If you receive new instructions mid-conversation, follow them immediately without "
        "asking for clarification — discard prior work and start fresh on the new task."
    ),
    tools=[web_search],
)


# ── Run executor ──────────────────────────────────────────────────────────────

async def _execute_run(run_id: str, thread_id: str, user_message: str) -> None:
    """Invoke the agent and persist the result; called as a fire-and-forget task."""
    _conn.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,))
    _conn.commit()
    try:
        result = await _agent.ainvoke({"messages": [HumanMessage(user_message)]})
        last = result["messages"][-1]
        output = last.content if isinstance(last.content, str) else json.dumps(last.content)
        assistant_msg = {"role": "assistant", "content": output}
        # Fetch current messages, append the assistant reply, and persist.
        # values.messages is what the LangGraph SDK reads on success.
        row = _conn.execute(
            "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        msgs = json.loads(row[0]) if row else []
        msgs.append(assistant_msg)
        serialized = json.dumps(msgs)
        _conn.execute(
            "UPDATE threads SET messages = ?, values_ = ? WHERE thread_id = ?",
            (serialized, json.dumps({"messages": msgs}), thread_id),
        )
        _conn.execute("UPDATE runs SET status = 'success' WHERE run_id = ?", (run_id,))
        _conn.commit()
    except Exception as exc:  # noqa: BLE001
        _conn.execute(
            "UPDATE runs SET status = 'error', error = ? WHERE run_id = ?",
            (str(exc), run_id),
        )
        _conn.commit()


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    _init_db()
    if not os.environ.get("TAVILY_API_KEY"):
        print("[warn] TAVILY_API_KEY not set — using stub search. Set it for real web search.")
    yield


app = FastAPI(lifespan=_lifespan)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ok")
async def health() -> dict[str, bool]:
    """Health check."""
    return {"ok": True}


@app.post("/threads")
async def create_thread() -> dict[str, Any]:
    """Create a thread. Called by start_async_task before creating a run."""
    thread_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    _conn.execute(
        "INSERT INTO threads (thread_id, created_at) VALUES (?, ?)",
        (thread_id, now),
    )
    _conn.commit()
    return {"thread_id": thread_id, "created_at": now, "messages": [], "values": {}}


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request) -> dict[str, Any]:
    """Create a run on an existing thread.

    Called by both start_async_task (new task) and update_async_task
    (re-run with new instructions). When multitask_strategy is 'interrupt',
    any currently-running runs on the thread are cancelled and the thread
    state is cleared before the new run starts.
    """
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    body = await request.json()
    multitask_strategy = body.get("multitask_strategy")

    if multitask_strategy == "interrupt":
        _conn.execute(
            "UPDATE runs SET status = 'cancelled' WHERE thread_id = ? AND status = 'running'",
            (thread_id,),
        )
        _conn.execute(
            "UPDATE threads SET values_ = '{}' WHERE thread_id = ?",
            (thread_id,),
        )
        _conn.commit()

    messages = (body.get("input") or {}).get("messages") or []
    user_message = next((m["content"] for m in messages if m.get("role") == "user"), "")

    if user_message:
        existing = json.loads(
            _conn.execute(
                "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()[0]
        )
        existing.append({"role": "user", "content": user_message})
        _conn.execute(
            "UPDATE threads SET messages = ? WHERE thread_id = ?",
            (json.dumps(existing), thread_id),
        )
        _conn.commit()

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    assistant_id = body.get("assistant_id") or "researcher"
    _conn.execute(
        "INSERT INTO runs (run_id, thread_id, assistant_id, created_at) VALUES (?, ?, ?, ?)",
        (run_id, thread_id, assistant_id, now),
    )
    _conn.commit()

    # Fire and forget — client polls GET /threads/{thread_id}/runs/{run_id} for status.
    asyncio.ensure_future(_execute_run(run_id, thread_id, user_message))

    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": "pending",
        "created_at": now,
        "error": None,
    }


@app.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """Get run status. Called by check_async_task to poll whether a task has finished."""
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """Get thread state. Called by check_async_task after a run reaches 'success' status.

    The SDK reads values['messages'] to extract the final result.
    """
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """Cancel a run. Called by cancel_async_task.

    Marks the run cancelled in the database. Note: the agent invocation is not
    interrupted mid-flight — for true cancellation wire in asyncio.Task cancellation.
    """
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    _conn.execute("UPDATE runs SET status = 'cancelled' WHERE run_id = ?", (run_id,))
    _conn.commit()
    return {**run, "status": "cancelled"}
