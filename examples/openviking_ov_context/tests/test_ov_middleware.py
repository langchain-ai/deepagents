"""Unit tests for the OV lifecycle-hook demo."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping
import unittest

from langchain_core.messages import AIMessage, ToolMessage
from openviking.integrations.langchain.ov_context.state import OVContext

from commit_policy import CommitConfig, CommitPolicy
from ov_middleware import OVMiddleware
from tests.mock_ov_message_writer import MockOVMessageWriter


class FakeOVClient:
    def __init__(self, *, archived: bool = True) -> None:
        self.archived = archived
        self.pending_tokens = 5
        self.commit_calls: list[tuple[str, int]] = []

    async def commit_session(
        self,
        session_id: str,
        *,
        keep_recent_count: int,
    ) -> Mapping[str, Any]:
        self.commit_calls.append((session_id, keep_recent_count))
        return {
            "session_id": session_id,
            "archived": self.archived,
            "task_id": "task-test",
            "archive_uri": "viking://test/archive_001" if self.archived else None,
        }

    async def get_session(self, session_id: str) -> Mapping[str, Any]:
        return {"session_id": session_id, "pending_tokens": self.pending_tokens}


def fake_context(**overrides: Any) -> dict[str, Any]:
    values = {"session_id": "session-test", "run_id": "run-test", **overrides}
    return OVContext(**values).to_state()


class OVMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = FakeOVClient()
        self.policy = CommitPolicy(
            CommitConfig(
                token_budget=10,
                commit_token_threshold_ratio=Decimal("0.5"),
                completed_run_threshold=1,
                keep_recent_count=1,
            ),
            self.client,
        )
        self.writer = MockOVMessageWriter("session-test")
        self.session_reads: list[str] = []

        async def get_session(session_id: str) -> Mapping[str, Any]:
            self.session_reads.append(session_id)
            return await self.client.get_session(session_id)

        class SessionClient:
            async def get_session(inner_self, session_id: str) -> Mapping[str, Any]:
                return await get_session(session_id)

        self.middleware = OVMiddleware(
            writer=self.writer,
            session_client=SessionClient(),
            commit_policy=self.policy,
        )

    async def test_after_model_flushes_and_passes_live_metric_to_commit_policy(
        self,
    ) -> None:
        original_context = fake_context(completed_runs=2)
        message = AIMessage(content="done")

        update = await self.middleware.aafter_model(
            {"ov_context": original_context, "messages": [message]},
            runtime=None,
        )

        self.assertIsNotNone(update)
        updated_context = update["ov_context"]
        self.assertEqual(updated_context["write_index"], 1)
        self.assertEqual(updated_context["messages"], [message])
        self.assertEqual(updated_context["completed_runs"], 0)
        self.assertEqual(original_context["write_index"], 0)
        self.assertEqual(self.writer.calls, [("session-test", 0, (message,))])
        self.assertEqual(self.session_reads, ["session-test"])
        self.assertEqual(self.client.commit_calls, [("session-test", 1)])

    async def test_after_model_defers_metrics_and_commit_while_tool_is_pending(
        self,
    ) -> None:
        context = fake_context()
        message = AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "lookup", "args": {}}],
        )

        update = await self.middleware.aafter_model(
            {"ov_context": context, "messages": [message]},
            runtime=None,
        )

        self.assertEqual(update["ov_context"]["write_index"], 1)
        self.assertEqual(self.writer.calls, [("session-test", 0, (message,))])
        self.assertEqual(self.session_reads, [])
        self.assertEqual(self.client.commit_calls, [])

    async def test_after_model_returns_none_when_writer_has_no_delta(self) -> None:
        message = AIMessage(content="already written")
        context = fake_context(messages=[message], write_index=1)
        self.writer.write_index = 1

        update = await self.middleware.aafter_model(
            {"ov_context": context, "messages": [message]},
            runtime=None,
        )

        self.assertIsNone(update)
        self.assertEqual(self.session_reads, [])
        self.assertEqual(self.client.commit_calls, [])

    async def test_return_direct_tool_result_is_flushed_at_after_agent(self) -> None:
        assistant = AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "lookup", "args": {}}],
        )
        tool_result = ToolMessage(content="direct result", tool_call_id="call-1")
        original = fake_context()

        first = await self.middleware.aafter_model(
            {"ov_context": original, "messages": [assistant]}, runtime=None
        )
        self.assertEqual(first["ov_context"]["write_index"], 1)
        self.assertEqual(self.client.commit_calls, [])

        final = await self.middleware.aafter_agent(
            {"ov_context": first["ov_context"], "messages": [assistant, tool_result]},
            runtime=None,
        )
        self.assertEqual(final["ov_context"]["write_index"], 2)
        self.assertEqual(final["ov_context"]["messages"], [assistant, tool_result])
        self.assertEqual(final["ov_context"]["last_completed_run"], "run-test")
        self.assertIsNone(final["ov_context"]["run_id"])
        self.assertEqual(self.writer.calls[-1], ("session-test", 1, (tool_result,)))
        self.assertEqual(len(self.client.commit_calls), 1)

    async def test_after_agent_flushes_increments_run_and_ends_it(self) -> None:
        self.client.archived = False
        context = fake_context()
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "lookup", "args": {}}],
            ),
            ToolMessage(content="result", tool_call_id="call-1"),
        ]

        update = await self.middleware.aafter_agent(
            {"ov_context": context, "messages": messages},
            runtime=None,
        )

        updated_context = update["ov_context"]
        self.assertEqual(updated_context["write_index"], 2)
        self.assertEqual(updated_context["completed_runs"], 1)
        self.assertEqual(updated_context["last_completed_run"], "run-test")
        self.assertIsNone(updated_context["run_id"])
        self.assertEqual(self.session_reads, ["session-test"])
        self.assertEqual(self.client.commit_calls, [("session-test", 1)])

        repeated = await self.middleware.aafter_agent(
            {"ov_context": updated_context, "messages": messages},
            runtime=None,
        )
        self.assertIsNone(repeated)
        self.assertEqual(self.session_reads, ["session-test"])

    async def test_after_agent_resets_completed_runs_after_archive(self) -> None:
        self.client.pending_tokens = 1  # Keep the Run-count trigger distinct.
        message = AIMessage(content="final answer")
        context = fake_context()

        update = await self.middleware.aafter_agent(
            {"ov_context": context, "messages": [message]},
            runtime=None,
        )

        updated_context = update["ov_context"]
        self.assertEqual(self.client.commit_calls, [("session-test", 1)])
        self.assertEqual(updated_context["completed_runs"], 0)
        self.assertEqual(updated_context["last_completed_run"], "run-test")
        self.assertIsNone(updated_context["run_id"])


if __name__ == "__main__":
    unittest.main()
