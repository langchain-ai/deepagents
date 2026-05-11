import asyncio
import unittest

from control_server.round_timer import RoundTimer


class TestRoundTimer(unittest.TestCase):
    def test_snapshot_idle_before_start(self) -> None:
        timer = RoundTimer()
        snap = timer.snapshot()
        self.assertFalse(snap.running)
        self.assertEqual(snap.remaining_secs, 0.0)
        self.assertIsNone(snap.started_at)

    def test_callback_fires_after_duration(self) -> None:
        async def scenario() -> int:
            calls = 0
            event = asyncio.Event()

            async def on_expire() -> None:
                nonlocal calls
                calls += 1
                event.set()

            timer = RoundTimer()
            await timer.start(0.05, on_expire)
            self.assertTrue(timer.snapshot().running)
            await asyncio.wait_for(event.wait(), timeout=1.0)
            return calls

        self.assertEqual(asyncio.run(scenario()), 1)

    def test_cancel_prevents_callback(self) -> None:
        async def scenario() -> int:
            calls = 0

            async def on_expire() -> None:
                nonlocal calls
                calls += 1

            timer = RoundTimer()
            await timer.start(0.5, on_expire)
            await asyncio.sleep(0.05)
            await timer.cancel()
            await asyncio.sleep(0.6)
            return calls

        self.assertEqual(asyncio.run(scenario()), 0)

    def test_restart_supersedes_previous_callback(self) -> None:
        async def scenario() -> tuple[int, int]:
            first = 0
            second = 0

            async def cb_first() -> None:
                nonlocal first
                first += 1

            async def cb_second() -> None:
                nonlocal second
                second += 1

            timer = RoundTimer()
            await timer.start(0.5, cb_first)
            await asyncio.sleep(0.05)
            await timer.start(0.05, cb_second)
            await asyncio.sleep(0.3)
            return first, second

        first, second = asyncio.run(scenario())
        self.assertEqual(first, 0)
        self.assertEqual(second, 1)


if __name__ == "__main__":
    unittest.main()
