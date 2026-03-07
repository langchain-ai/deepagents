"""Tests for TaskIQ-backed background runtime."""

import asyncio

import pytest

from deepagents_cli.background_runtime import (
    BackgroundApprovalDecision,
    BackgroundRuntime,
    BackgroundTaskStatus,
)


class TestBackgroundRuntimeLifecycle:
    """Lifecycle and task-state tests for BackgroundRuntime."""

    async def test_start_and_shutdown(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        await runtime.shutdown()

    async def test_submit_and_wait_success(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf hello")
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.SUCCEEDED
            assert final.result_text == "hello"
            assert final.exit_code == 0
            notices = runtime.consume_tui_notifications()
            assert any("completed" in item for item in notices)
        finally:
            await runtime.shutdown()

    async def test_submit_failure(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("sh -c 'exit 3'")
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.FAILED
            assert final.exit_code == 3
        finally:
            await runtime.shutdown()

    async def test_kill_running_task(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("sleep 10")
            killed = await runtime.kill_task(task_id)
            assert killed is True
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.KILLED
        finally:
            await runtime.shutdown()

    async def test_kill_finished_task_returns_false(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf done")
            await runtime.wait_task(task_id, timeout_seconds=5)
            killed = await runtime.kill_task(task_id)
            assert killed is False
        finally:
            await runtime.shutdown()

    async def test_wait_timeout(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=True)
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf never-runs")
            with pytest.raises(TimeoutError):
                await runtime.wait_task(task_id, timeout_seconds=0.05)
        finally:
            await runtime.shutdown()

    async def test_hitl_approval_and_rejection(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=True)
        await runtime.start()
        try:
            approved_task = await runtime.submit_shell_task("printf approve-ok")
            event = None
            for _ in range(50):
                event = runtime.pop_hitl_event()
                if event is not None:
                    break
                await asyncio.sleep(0.01)
            assert event is not None
            runtime.resolve_hitl_event(
                event.event_id,
                decision=BackgroundApprovalDecision.APPROVE,
            )
            approved = await runtime.wait_task(approved_task, timeout_seconds=5)
            assert approved.status == BackgroundTaskStatus.SUCCEEDED

            rejected_task = await runtime.submit_shell_task("printf reject-no")
            event2 = None
            for _ in range(50):
                event2 = runtime.pop_hitl_event()
                if event2 is not None:
                    break
                await asyncio.sleep(0.01)
            assert event2 is not None
            runtime.resolve_hitl_event(
                event2.event_id,
                decision=BackgroundApprovalDecision.REJECT,
                message="Rejected by test",
            )
            rejected = await runtime.wait_task(rejected_task, timeout_seconds=5)
            assert rejected.status == BackgroundTaskStatus.FAILED
            assert rejected.error_text is not None
            assert "Rejected by test" in rejected.error_text
        finally:
            await runtime.shutdown()

    async def test_concurrent_submissions_are_stable(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            task_ids = await asyncio.gather(
                *(runtime.submit_shell_task(f"printf t{i}") for i in range(5))
            )
            finals = await asyncio.gather(
                *(runtime.wait_task(task_id, timeout_seconds=5) for task_id in task_ids)
            )
            assert all(item.status == BackgroundTaskStatus.SUCCEEDED for item in finals)
        finally:
            await runtime.shutdown()
