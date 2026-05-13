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
        axes: dict[str, float | None] = dict.fromkeys(eval_runner.LLM_AXES, 0.5)
        self.assertAlmostEqual(eval_runner.aggregate(axes), 0.5, places=6)

    def test_aggregate_skips_none_axes_from_numerator_and_denominator(self) -> None:
        # Color is excluded from both sides of the weighted mean, so the
        # result should match the present-axes mean exactly (0.6) instead
        # of being biased downward by treating the missing color as 0.
        axes: dict[str, float | None] = dict.fromkeys(eval_runner.LLM_AXES, 0.6)
        axes["color"] = None
        self.assertAlmostEqual(eval_runner.aggregate(axes), 0.6, places=6)

    def test_aggregate_returns_zero_when_no_axes_have_values(self) -> None:
        axes: dict[str, float | None] = dict.fromkeys(eval_runner.LLM_AXES, None)
        self.assertEqual(eval_runner.aggregate(axes), 0.0)

    def test_sanitize_axes_coerces_strings_and_defaults_missing_values(self) -> None:
        raw = {
            "color": "0.6",
            "typography": 0.7,
            "layout": "garbage",
            "creativity": None,
            "junk_axis": 0.9,
        }
        with patch.object(eval_runner.random, "randint", return_value=6):
            sanitized = eval_runner._sanitize_axes(raw)  # noqa: SLF001
        self.assertEqual(sanitized["color"], 0.6)
        self.assertEqual(sanitized["typography"], 0.7)
        self.assertEqual(sanitized["layout"], 0.6)
        self.assertEqual(sanitized["content_completeness"], 0.6)
        self.assertEqual(sanitized["creativity"], 0.6)
        self.assertEqual(sanitized["interpretation_quality"], 0.6)
        self.assertNotIn("junk_axis", sanitized)

    def test_eval_result_rejects_fallback_without_reason(self) -> None:
        with self.assertRaises(ValueError):
            eval_runner.EvalResult(
                site_name="Alice",
                url="",
                prompt="",
                round_num=1,
                fallback=True,
                fallback_reason=None,
            )

    def test_eval_result_rejects_overall_outside_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            eval_runner.EvalResult(
                site_name="Alice",
                url="",
                prompt="",
                round_num=1,
                axes={},
                overall=1.5,
            )

    def test_eval_result_rejects_unknown_axes(self) -> None:
        with self.assertRaises(ValueError):
            eval_runner.EvalResult(
                site_name="Alice",
                url="",
                prompt="",
                round_num=1,
                axes={"bogus_axis": 0.5},
            )

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
        self.assertEqual(result.fallback_reason, "judge exit 17")
        self.assertEqual(set(result.axes.keys()), set(eval_runner.LLM_AXES))
        self.assertIsNotNone(result.overall)

    def test_run_eval_falls_back_when_uv_binary_missing(self) -> None:
        async def fake_spawn(**_: object) -> tuple[int, bytes, bytes]:
            msg = "uv binary not found on PATH; cannot invoke the judge"
            raise FileNotFoundError(msg)

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
        self.assertIn("uv binary not found", result.fallback_reason or "")

    def test_run_eval_falls_back_on_subprocess_timeout(self) -> None:
        async def fake_spawn(**_: object) -> tuple[int, bytes, bytes]:
            msg = "judge subprocess exceeded 15s"
            raise TimeoutError(msg)

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
        self.assertIn("exceeded", result.fallback_reason or "")

    def test_run_eval_defaults_missing_llm_axes_to_random_scores(self) -> None:
        async def fake_spawn(**kwargs: object) -> tuple[int, bytes, bytes]:
            work_dir = kwargs["work_dir"]
            round_num = kwargs["round_num"]
            name = kwargs["name"]
            path = Path(work_dir) / f"round-{round_num}-{name}.json"
            path.write_text(json.dumps({"axes": {"accessibility": 1.0}}))
            return 0, b"", b""

        with (
            patch.object(eval_runner, "_spawn_judge", new=fake_spawn),
            patch.object(eval_runner.random, "randint", return_value=6),
        ):
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
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.axes["accessibility"], 1.0)
        self.assertEqual(result.axes["color"], 0.6)
        self.assertEqual(result.axes["interpretation_quality"], 0.6)

    def test_run_eval_falls_back_when_judge_output_json_missing(self) -> None:
        async def fake_spawn(**_: object) -> tuple[int, bytes, bytes]:
            # Judge exits cleanly but writes nothing.
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

        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, "judge output JSON missing")

    def test_run_eval_falls_back_when_judge_output_is_malformed(self) -> None:
        async def fake_spawn(**kwargs: object) -> tuple[int, bytes, bytes]:
            work_dir = kwargs["work_dir"]
            round_num = kwargs["round_num"]
            name = kwargs["name"]
            path = Path(work_dir) / f"round-{round_num}-{name}.json"
            path.write_text("{not json")
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

        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, "judge output JSON unreadable")

    def test_run_eval_falls_back_when_axes_key_missing(self) -> None:
        async def fake_spawn(**kwargs: object) -> tuple[int, bytes, bytes]:
            work_dir = kwargs["work_dir"]
            round_num = kwargs["round_num"]
            name = kwargs["name"]
            path = Path(work_dir) / f"round-{round_num}-{name}.json"
            path.write_text(json.dumps({"other": 1}))
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

        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, "judge output JSON missing axes")

    def test_run_eval_falls_back_when_axes_is_not_a_mapping(self) -> None:
        async def fake_spawn(**kwargs: object) -> tuple[int, bytes, bytes]:
            work_dir = kwargs["work_dir"]
            round_num = kwargs["round_num"]
            name = kwargs["name"]
            path = Path(work_dir) / f"round-{round_num}-{name}.json"
            path.write_text(json.dumps({"axes": [0.5, 0.6]}))
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

        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, "judge output JSON missing axes")

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
