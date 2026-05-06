from __future__ import annotations

import uuid
from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves annotations at runtime
)
from typing import Any

import pytest
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from langchain_quickjs import REPLMiddleware

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:  # pragma: no cover - env-dependent optional dependency
    AsyncPostgresSaver = None  # type: ignore[assignment]

try:
    from docker.errors import DockerException
    from testcontainers.postgres import PostgresContainer
except ImportError:  # pragma: no cover - env-dependent optional dependency
    DockerException = RuntimeError  # type: ignore[assignment]
    PostgresContainer = None  # type: ignore[assignment]


pytestmark = [
    pytest.mark.skipif(
        AsyncPostgresSaver is None,
        reason="langgraph-checkpoint-postgres not installed",
    ),
    pytest.mark.skipif(
        PostgresContainer is None,
        reason="testcontainers[postgres] not installed",
    ),
]


class _FakeChatModel(GenericFakeChatModel):
    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, _tools: list[Any], **_: Any) -> _FakeChatModel:
        return self


def _script(code: str, *, final_message: str = "done") -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": code},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content=final_message),
        ]
    )


def _eval_tool_message(result: dict[str, Any]) -> ToolMessage:
    messages = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


def _assert_result_contains(content: str, expected: str) -> None:
    assert "<error" not in content, content
    assert "<result" in content, content
    assert expected in content, content


def _normalize_psycopg_conn_string(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql://", 1)


@pytest.fixture
def postgres_url() -> Iterator[str]:
    try:
        assert PostgresContainer is not None  # narrowed by pytest skip
        with PostgresContainer("postgres:16-alpine") as postgres:
            yield _normalize_psycopg_conn_string(postgres.get_connection_url())
    except DockerException as exc:
        msg = f"Docker unavailable for Postgres test container: {exc}"
        pytest.skip(msg)


async def test_ptc_task_with_postgres_checkpointer_keeps_loop_affinity(
    postgres_url: str,
) -> None:
    """Regression for graph-in-graph loop affinity with Postgres checkpoints.

    Historically this path failed with cross-loop runtime errors such as:
    - "Future ... attached to a different loop"
    - "Lock ... is bound to a different event loop"

    The assertion here is intentionally behavioral: the nested `task` call
    should complete successfully and return the subagent result.
    """
    # AsyncPostgresSaver.from_conn_string returns an async iterator/context manager.
    assert AsyncPostgresSaver is not None  # narrowed by pytest skip
    async with AsyncPostgresSaver.from_conn_string(postgres_url) as checkpointer:
        await checkpointer.setup()
        thread_id = f"qjs-pg-{uuid.uuid4().hex[:8]}"
        agent = create_agent(
            model=_FakeChatModel(
                messages=_script(
                    "await tools.task({"
                    "description: 'say hi', subagent_type: 'researcher'"
                    "})",
                    final_message="done",
                )
            ),
            middleware=[
                SubAgentMiddleware(
                    backend=None,
                    subagents=[
                        {
                            "name": "researcher",
                            "description": "returns one short answer",
                            "system_prompt": (
                                "You are a researcher. "
                                "Reply with exactly 'subagent-ok'."
                            ),
                            "model": _FakeChatModel(
                                messages=iter([AIMessage(content="subagent-ok")])
                            ),
                            "tools": [],
                        }
                    ],
                ),
                REPLMiddleware(ptc=["task"]),
            ],
            checkpointer=checkpointer,
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Use eval and call task for researcher")
                ]
            },
            config={"configurable": {"thread_id": thread_id}},
        )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "subagent-ok")
