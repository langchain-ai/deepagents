import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from control_server import eval_runner


class TestEvalRunner(unittest.TestCase):
    def test_to_obs_score_clamps_and_scales(self) -> None:
        self.assertEqual(eval_runner.to_obs_score(0.0), 0.0)
        self.assertEqual(eval_runner.to_obs_score(1.0), 10.0)
        self.assertEqual(eval_runner.to_obs_score(0.5), 5.0)
        self.assertEqual(eval_runner.to_obs_score(-0.1), 0.0)
        self.assertEqual(eval_runner.to_obs_score(2.0), 10.0)

    def test_aggregate_weighted_mean(self) -> None:
        axes = dict.fromkeys(eval_runner.LLM_AXES, 0.5)
        self.assertAlmostEqual(eval_runner.aggregate(axes), 0.5, places=3)

    def test_aggregate_skips_none_axes(self) -> None:
        axes: dict[str, float | None] = dict.fromkeys(eval_runner.LLM_AXES, 0.6)
        axes["color"] = None
        result = eval_runner.aggregate(axes)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_sanitize_axes_coerces_strings_and_drops_unknowns(self) -> None:
        raw = {
            "color": "0.6",
            "typography": 0.7,
            "layout": "garbage",
            "creativity": None,
            "junk_axis": 0.9,
        }
        sanitized = eval_runner._sanitize_axes(raw)  # noqa: SLF001
        self.assertEqual(sanitized["color"], 0.6)
        self.assertEqual(sanitized["typography"], 0.7)
        self.assertIsNone(sanitized["layout"])
        self.assertIsNone(sanitized["creativity"])
        self.assertNotIn("junk_axis", sanitized)

    def test_run_eval_returns_fallback_when_judge_exits_nonzero(self) -> None:
        async def fake_spawn(**_: object) -> tuple[int, bytes, bytes]:
            return 17, b"", b"boom\n"

        with patch.object(eval_runner, "_spawn_judge", new=fake_spawn):
            with TemporaryDirectory() as tmp:
                result = asyncio.run(
                    eval_runner.run_eval(
                        url="http://x/",
                        site_name="Alice",
                        prompt="p",
                        round_num=1,
                        work_dir=Path(tmp),
                    )
                )

        self.assertTrue(result.fallback)
        self.assertIn("judge exit", result.fallback_reason or "")
        self.assertEqual(set(result.axes.keys()), set(eval_runner.LLM_AXES))
        self.assertIsNotNone(result.overall)

    def test_run_eval_reads_judge_json(self) -> None:
        async def fake_spawn(**kwargs: object) -> tuple[int, bytes, bytes]:
            work_dir = kwargs["work_dir"]
            round_num = kwargs["round_num"]
            name = kwargs["name"]
            path = Path(work_dir) / f"round-{round_num}-{name}.json"
            payload = {
                "axes": {
                    "color": 0.8,
                    "typography": 0.7,
                    "layout": 0.6,
                    "content_completeness": 0.5,
                    "creativity": 0.4,
                    "interpretation_quality": 0.3,
                    "accessibility": 0.9,
                },
            }
            path.write_text(json.dumps(payload))
            return 0, b"", b""

        with patch.object(eval_runner, "_spawn_judge", new=fake_spawn):
            with TemporaryDirectory() as tmp:
                result = asyncio.run(
                    eval_runner.run_eval(
                        url="http://x/",
                        site_name="Alice",
                        prompt="p",
                        round_num=1,
                        work_dir=Path(tmp),
                    )
                )

        self.assertFalse(result.fallback)
        self.assertEqual(result.axes["color"], 0.8)
        self.assertGreater(result.overall or 0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
