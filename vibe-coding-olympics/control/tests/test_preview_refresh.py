from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from control_server import preview_refresh


class TestPreviewRefresh(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_preview_waits_for_ready_port_before_opening(self) -> None:
        with (
            patch.object(preview_refresh, "_REFRESH_DELAY_SECS", 0),
            patch.object(
                preview_refresh,
                "_wait_for_ready_port",
                new=AsyncMock(return_value=True),
            ) as wait,
            patch.object(
                preview_refresh,
                "_open_url",
                new=AsyncMock(),
            ) as open_url,
        ):
            refreshed = await preview_refresh.refresh_preview_when_ready("3001")

        self.assertTrue(refreshed)
        wait.assert_awaited_once_with(3001)
        open_url.assert_awaited_once_with("http://localhost:3001")

    async def test_refresh_preview_skips_invalid_port(self) -> None:
        with patch.object(
            preview_refresh,
            "_open_url",
            new=AsyncMock(),
        ) as open_url:
            refreshed = await preview_refresh.refresh_preview_when_ready("bad")

        self.assertFalse(refreshed)
        open_url.assert_not_awaited()

    def test_schedule_preview_refresh_is_explicitly_opt_in(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(
                preview_refresh,
                "refresh_preview_when_ready",
                new=Mock(),
            ) as refresh,
            patch.object(preview_refresh.asyncio, "create_task") as create_task,
        ):
            preview_refresh.schedule_preview_refresh("3001")

        refresh.assert_not_called()
        create_task.assert_not_called()

    def test_schedule_preview_refresh_uses_configured_port_when_enabled(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {preview_refresh.OPEN_ON_CLEAR_ENV: "1"},
                clear=True,
            ),
            patch.object(
                preview_refresh,
                "refresh_preview_when_ready",
                new=Mock(return_value=object()),
            ) as refresh,
            patch.object(preview_refresh.asyncio, "create_task") as create_task,
        ):
            preview_refresh.schedule_preview_refresh("3001")

        refresh.assert_called_once_with("3001")
        create_task.assert_called_once()


if __name__ == "__main__":
    unittest.main()
