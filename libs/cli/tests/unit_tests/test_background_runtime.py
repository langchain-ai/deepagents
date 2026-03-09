"""Tests for TaskIQ-backed background runtime."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from deepagents_cli.background_runtime import (
    BackgroundApprovalDecision,
    BackgroundRuntime,
    BackgroundTaskStatus,
)


def _local_backend() -> LocalShellBackend:
    return LocalShellBackend()


class TestBackgroundRuntimeLifecycle:
    """Lifecycle and task-state tests for BackgroundRuntime."""

    async def test_start_and_shutdown(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        await runtime.start()
        await runtime.shutdown()

    async def test_submit_and_wait_success(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf hello")
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.SUCCEEDED
            assert final.result_text is not None
            assert "hello" in final.result_text
            assert final.exit_code == 0
            notices = runtime.consume_tui_notifications()
            assert any("completed" in item for item in notices)
        finally:
            await runtime.shutdown()

    async def test_submit_failure(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("sh -c 'exit 3'")
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.FAILED
            assert final.exit_code == 3
        finally:
            await runtime.shutdown()

    async def test_kill_running_task(self) -> None:
        # Use async mock to avoid leaving threads running after kill
        never_finish = asyncio.Event()

        async def slow_execute(_command: str, **_kwargs: object) -> ExecuteResponse:
            await never_finish.wait()
            return ExecuteResponse(output="done", exit_code=0)

        mock_backend = AsyncMock()
        mock_backend.aexecute = slow_execute

        runtime = BackgroundRuntime(require_hitl_for_shell=False, backend=mock_backend)
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("sleep 10")
            # Give the task time to reach RUNNING state
            await asyncio.sleep(0.1)
            killed = await runtime.kill_task(task_id)
            assert killed is True
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.KILLED
        finally:
            await runtime.shutdown()

    async def test_kill_finished_task_returns_false(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf done")
            await runtime.wait_task(task_id, timeout_seconds=5)
            killed = await runtime.kill_task(task_id)
            assert killed is False
        finally:
            await runtime.shutdown()

    async def test_wait_timeout(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=True, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf never-runs")
            with pytest.raises(TimeoutError):
                await runtime.wait_task(task_id, timeout_seconds=0.05)
        finally:
            await runtime.shutdown()

    async def test_hitl_approval_and_rejection(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=True, backend=_local_backend()
        )
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
            runtime.consume_tui_notifications()
            runtime.consume_status_updates()

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
            assert rejected.status == BackgroundTaskStatus.REJECTED
            assert rejected.stderr_text is not None
            assert "Rejected by test" in rejected.stderr_text
            assert not any(
                "rejected by" in item.lower()
                for item in runtime.consume_tui_notifications()
            )
            assert not any(
                "rejected by" in item.lower()
                for item in runtime.consume_status_updates()
            )
        finally:
            await runtime.shutdown()

    async def test_concurrent_submissions_are_stable(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
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

    async def test_wait_for_no_pending_hitl_returns_immediately_when_idle(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=True, backend=_local_backend()
        )
        await runtime.start()
        try:
            await asyncio.wait_for(runtime.wait_for_no_pending_hitl(), timeout=0.2)
            assert runtime.pending_hitl_count() == 0
        finally:
            await runtime.shutdown()

    async def test_wait_for_no_pending_hitl_blocks_until_resolution(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=True, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf gated")

            event = None
            for _ in range(50):
                event = runtime.pop_hitl_event()
                if event is not None:
                    break
                await asyncio.sleep(0.01)
            assert event is not None
            assert runtime.pending_hitl_count() == 1

            idle_wait = asyncio.create_task(runtime.wait_for_no_pending_hitl())
            await asyncio.sleep(0)
            assert not idle_wait.done()

            runtime.resolve_hitl_event(
                event.event_id,
                decision=BackgroundApprovalDecision.APPROVE,
            )
            await asyncio.wait_for(idle_wait, timeout=1)
            assert runtime.pending_hitl_count() == 0

            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.SUCCEEDED
        finally:
            await runtime.shutdown()

    async def test_kill_pending_approval_unblocks_hitl_idle_wait(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=True, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf never-approve")

            event = None
            for _ in range(50):
                event = runtime.pop_hitl_event()
                if event is not None:
                    break
                await asyncio.sleep(0.01)
            assert event is not None
            assert runtime.pending_hitl_count() == 1

            idle_wait = asyncio.create_task(runtime.wait_for_no_pending_hitl())
            await asyncio.sleep(0)
            assert not idle_wait.done()

            killed = await runtime.kill_task(task_id)
            assert killed is True
            await asyncio.wait_for(idle_wait, timeout=1)
            assert runtime.pending_hitl_count() == 0
        finally:
            await runtime.shutdown()

    async def test_kill_rejected_task_returns_false(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=True, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf reject-no")
            event = None
            for _ in range(50):
                event = runtime.pop_hitl_event()
                if event is not None:
                    break
                await asyncio.sleep(0.01)
            assert event is not None
            runtime.resolve_hitl_event(
                event.event_id,
                decision=BackgroundApprovalDecision.REJECT,
                message="Rejected by user",
            )
            await runtime.wait_task(task_id, timeout_seconds=5)
            killed = await runtime.kill_task(task_id)
            assert killed is False
        finally:
            await runtime.shutdown()

    async def test_submit_before_start_raises(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        with pytest.raises(RuntimeError, match="has not been started"):
            await runtime.submit_shell_task("echo hi")

    async def test_wait_unknown_task_raises(self) -> None:
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        await runtime.start()
        try:
            with pytest.raises(ValueError, match="Unknown background task"):
                await runtime.wait_task("nonexistent")
        finally:
            await runtime.shutdown()

    async def test_shutdown_with_active_task_completes(self) -> None:
        # Use async mock to avoid leaving threads running after shutdown
        never_finish = asyncio.Event()

        async def slow_execute(_command: str, **_kwargs: object) -> ExecuteResponse:
            await never_finish.wait()
            return ExecuteResponse(output="done", exit_code=0)

        mock_backend = AsyncMock()
        mock_backend.aexecute = slow_execute

        runtime = BackgroundRuntime(require_hitl_for_shell=False, backend=mock_backend)
        await runtime.start()
        await runtime.submit_shell_task("sleep 30")
        await asyncio.wait_for(runtime.shutdown(), timeout=5)

    async def test_unsupported_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported background runtime mode"):
            BackgroundRuntime(mode="distributed")  # type: ignore[arg-type]

    async def test_submit_without_backend_fails(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("echo hi")
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.FAILED
            assert final.stderr_text is not None
            assert "No execution backend" in final.stderr_text
        finally:
            await runtime.shutdown()

    async def test_kill_local_terminates_subprocess(self) -> None:
        """Kill with LocalShellBackend actually terminates the process."""
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("sleep 60")
            # Give the subprocess time to start
            await asyncio.sleep(0.2)
            killed = await runtime.kill_task(task_id)
            assert killed is True
            # Should complete quickly since SIGTERM was sent
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.KILLED
        finally:
            await runtime.shutdown()

    async def test_shutdown_local_terminates_processes(self) -> None:
        """Shutdown with LocalShellBackend terminates running subprocesses."""
        runtime = BackgroundRuntime(
            require_hitl_for_shell=False, backend=_local_backend()
        )
        await runtime.start()
        await runtime.submit_shell_task("sleep 60")
        # Give the subprocess time to start
        await asyncio.sleep(0.2)
        # Shutdown should complete quickly because it kills the subprocess
        await asyncio.wait_for(runtime.shutdown(), timeout=5)

    async def test_backend_can_be_set_after_init(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        runtime.set_backend(_local_backend())
        await runtime.start()
        try:
            task_id = await runtime.submit_shell_task("printf deferred")
            final = await runtime.wait_task(task_id, timeout_seconds=5)
            assert final.status == BackgroundTaskStatus.SUCCEEDED
            assert final.result_text is not None
            assert "deferred" in final.result_text
        finally:
            await runtime.shutdown()
