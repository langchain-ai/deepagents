import asyncio
import unittest

from control_server.round_timer import (
    RoundTimer,
    TimerSnapshot,
    timer_warning_for_remaining,
)


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

    def test_start_delay_does_not_consume_visible_round_time(self) -> None:
        async def scenario() -> tuple[float, int]:
            calls = 0

            async def on_expire() -> None:
                nonlocal calls
                calls += 1

            timer = RoundTimer()
            await timer.start(0.05, on_expire, start_delay_secs=0.1)
            remaining = timer.snapshot().remaining_secs
            await asyncio.sleep(0.07)
            delayed_calls = calls
            await asyncio.sleep(0.15)
            return remaining, delayed_calls

        remaining, delayed_calls = asyncio.run(scenario())
        self.assertLessEqual(remaining, 0.05)
        self.assertGreater(remaining, 0.0)
        self.assertEqual(delayed_calls, 0)

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

    def test_concurrent_starts_fire_callback_at_most_once(self) -> None:
        async def scenario() -> int:
            calls = 0
            event = asyncio.Event()

            async def on_expire() -> None:
                nonlocal calls
                calls += 1
                event.set()

            timer = RoundTimer()
            # Race four `start()` calls. The lock plus cancel-on-restart
            # must collapse them to a single live countdown that fires
            # exactly once.
            await asyncio.gather(
                timer.start(0.05, on_expire),
                timer.start(0.05, on_expire),
                timer.start(0.05, on_expire),
                timer.start(0.05, on_expire),
            )
            await asyncio.wait_for(event.wait(), timeout=1.0)
            await asyncio.sleep(0.1)
            return calls

        self.assertEqual(asyncio.run(scenario()), 1)

    def test_cancel_racing_expiry_does_not_double_fire(self) -> None:
        async def scenario() -> int:
            calls = 0

            async def on_expire() -> None:
                nonlocal calls
                calls += 1

            timer = RoundTimer()
            await timer.start(0.01, on_expire)
            # Cancel arrives at roughly the same time the timer would
            # naturally expire. The lock must serialize so we get either
            # 0 (cancel won) or 1 (expiry won), but never 2.
            await asyncio.sleep(0.01)
            await timer.cancel()
            await asyncio.sleep(0.1)
            return calls

        self.assertLessEqual(asyncio.run(scenario()), 1)

    def test_start_rejects_negative_duration(self) -> None:
        async def scenario() -> None:
            timer = RoundTimer()

            async def noop() -> None:
                return None

            await timer.start(-1.0, noop)

        with self.assertRaises(ValueError):
            asyncio.run(scenario())

    def test_start_rejects_negative_start_delay(self) -> None:
        async def scenario() -> None:
            timer = RoundTimer()

            async def noop() -> None:
                return None

            await timer.start(1.0, noop, start_delay_secs=-1.0)

        with self.assertRaises(ValueError):
            asyncio.run(scenario())

    def test_warning_thresholds_match_cli_timer(self) -> None:
        warning = timer_warning_for_remaining(
            duration_secs=300.0,
            remaining_secs=149.9,
        )
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning.threshold_secs, 150)
        self.assertEqual(warning.message, "2.5 minutes left")

        warning = timer_warning_for_remaining(
            duration_secs=300.0,
            remaining_secs=59.9,
        )
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning.threshold_secs, 60)
        self.assertEqual(warning.message, "1 minute left")

        warning = timer_warning_for_remaining(
            duration_secs=300.0,
            remaining_secs=29.9,
        )
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning.threshold_secs, 30)
        self.assertEqual(warning.message, "30 seconds left")

    def test_warning_ignores_thresholds_longer_than_timer(self) -> None:
        warning = timer_warning_for_remaining(
            duration_secs=120.0,
            remaining_secs=119.0,
        )

        self.assertIsNone(warning)


class TestTimerSnapshot(unittest.TestCase):
    def test_idle_factory_marks_snapshot_not_running(self) -> None:
        snap = TimerSnapshot.idle(duration_secs=60.0)
        self.assertFalse(snap.running)
        self.assertEqual(snap.duration_secs, 60.0)
        self.assertEqual(snap.remaining_secs, 0.0)
        self.assertIsNone(snap.started_at)

    def test_active_factory_marks_snapshot_running(self) -> None:
        snap = TimerSnapshot.active(
            duration_secs=60.0, remaining_secs=42.0, started_at=100.0
        )
        self.assertTrue(snap.running)
        self.assertEqual(snap.remaining_secs, 42.0)
        self.assertEqual(snap.started_at, 100.0)
        self.assertEqual(snap.warning.threshold_secs, 60)

    def test_rejects_running_without_started_at(self) -> None:
        with self.assertRaises(ValueError):
            TimerSnapshot(
                running=True,
                duration_secs=60.0,
                remaining_secs=60.0,
                started_at=None,
            )

    def test_rejects_idle_with_started_at(self) -> None:
        with self.assertRaises(ValueError):
            TimerSnapshot(
                running=False,
                duration_secs=60.0,
                remaining_secs=60.0,
                started_at=100.0,
            )

    def test_is_immutable(self) -> None:
        snap = TimerSnapshot.idle(duration_secs=60.0)
        with self.assertRaises(Exception):  # FrozenInstanceError
            snap.running = True  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
