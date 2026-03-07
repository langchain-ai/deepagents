"""Tests for app-level HITL bridge with background runtime."""

from __future__ import annotations

import asyncio

from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.background_runtime import BackgroundRuntime, BackgroundTaskStatus
from deepagents_cli.widgets.messages import AppMessage


class TestBackgroundHitlBridge:
    """Ensure background HITL approvals are bridged through existing app UI flow."""

    async def test_background_hitl_approve_flow(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=True)
        await runtime.start()
        try:
            app = DeepAgentsApp(background_runtime=runtime)

            async def approve_request(
                _requests, _assistant_id
            ) -> asyncio.Future[dict[str, str]]:
                await asyncio.sleep(0)
                fut = asyncio.get_running_loop().create_future()
                fut.set_result({"type": "approve"})
                return fut

            app._request_approval = approve_request  # type: ignore[method-assign]

            async with app.run_test() as pilot:
                await pilot.pause()
                task_id = await runtime.submit_shell_task("printf approved")
                final = await runtime.wait_task(task_id, timeout_seconds=5)
                assert final.status == BackgroundTaskStatus.SUCCEEDED
        finally:
            await runtime.shutdown()

    async def test_background_hitl_reject_flow(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=True)
        await runtime.start()
        try:
            app = DeepAgentsApp(background_runtime=runtime)

            async def reject_request(
                _requests, _assistant_id
            ) -> asyncio.Future[dict[str, str]]:
                await asyncio.sleep(0)
                fut = asyncio.get_running_loop().create_future()
                fut.set_result({"type": "reject"})
                return fut

            app._request_approval = reject_request  # type: ignore[method-assign]

            async with app.run_test() as pilot:
                await pilot.pause()
                task_id = await runtime.submit_shell_task("printf rejected")
                final = await runtime.wait_task(task_id, timeout_seconds=5)
                assert final.status == BackgroundTaskStatus.FAILED
        finally:
            await runtime.shutdown()

    async def test_background_hitl_events_processed_serially(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=True)
        await runtime.start()
        try:
            app = DeepAgentsApp(background_runtime=runtime)
            seen: list[int] = []

            async def approve_request(
                _requests, _assistant_id
            ) -> asyncio.Future[dict[str, str]]:
                await asyncio.sleep(0)
                seen.append(len(seen) + 1)
                fut = asyncio.get_running_loop().create_future()
                fut.set_result({"type": "approve"})
                return fut

            app._request_approval = approve_request  # type: ignore[method-assign]

            async with app.run_test() as pilot:
                await pilot.pause()
                task_ids = await asyncio.gather(
                    runtime.submit_shell_task("printf one"),
                    runtime.submit_shell_task("printf two"),
                )
                finals = await asyncio.gather(
                    *(
                        runtime.wait_task(task_id, timeout_seconds=5)
                        for task_id in task_ids
                    )
                )
                assert all(
                    item.status == BackgroundTaskStatus.SUCCEEDED for item in finals
                )
                assert len(seen) == 2
        finally:
            await runtime.shutdown()

    async def test_completed_background_task_shows_tui_message(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            app = DeepAgentsApp(background_runtime=runtime)
            async with app.run_test() as pilot:
                await pilot.pause()
                task_id = await runtime.submit_shell_task("printf done")
                final = await runtime.wait_task(task_id, timeout_seconds=5)
                assert final.status == BackgroundTaskStatus.SUCCEEDED
                deadline = asyncio.get_running_loop().time() + 2.0
                saw_completion_message = False
                while True:
                    await pilot.pause()
                    app_msgs = app.query(AppMessage)
                    if any(
                        "Background task" in str(widget._content)
                        and "completed" in str(widget._content)
                        for widget in app_msgs
                    ):
                        saw_completion_message = True
                        break
                    if asyncio.get_running_loop().time() >= deadline:
                        break
                assert saw_completion_message
        finally:
            await runtime.shutdown()

    async def test_bridge_resolves_pending_hitl_idle_wait(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=True)
        await runtime.start()
        try:
            app = DeepAgentsApp(background_runtime=runtime)

            async def approve_request(
                _requests, _assistant_id
            ) -> asyncio.Future[dict[str, str]]:
                await asyncio.sleep(0)
                fut = asyncio.get_running_loop().create_future()
                fut.set_result({"type": "approve"})
                return fut

            app._request_approval = approve_request  # type: ignore[method-assign]

            async with app.run_test() as pilot:
                await pilot.pause()
                task_id = await runtime.submit_shell_task("printf approved")
                idle_wait = asyncio.create_task(runtime.wait_for_no_pending_hitl())
                await asyncio.wait_for(idle_wait, timeout=2)
                final = await runtime.wait_task(task_id, timeout_seconds=5)
                assert final.status == BackgroundTaskStatus.SUCCEEDED
                assert runtime.pending_hitl_count() == 0
        finally:
            await runtime.shutdown()
