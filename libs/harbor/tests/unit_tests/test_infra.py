"""Tests for infrastructure noise analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepagents_harbor.infra import (
    FailureCategory,
    InfraMetadata,
    SandboxLike,
    classify_failure,
    collect_host_metadata,
    collect_sandbox_metadata,
    extract_exit_codes,
    format_ci,
    min_detectable_effect,
    wilson_ci,
)


class TestClassifyFailure:
    """Tests for failure classification logic."""

    def test_oom_exit_code(self):
        result = classify_failure(exit_codes=[137])
        assert result == FailureCategory.INFRA_OOM

    def test_timeout_exit_code(self):
        result = classify_failure(exit_codes=[124])
        assert result == FailureCategory.INFRA_TIMEOUT

    def test_oom_pattern_in_exception(self):
        result = classify_failure(exception_text="Container was OOMKilled by the runtime")
        assert result == FailureCategory.INFRA_OOM

    def test_timeout_pattern_in_exception(self):
        result = classify_failure(exception_text="Command timed out after 300 seconds")
        assert result == FailureCategory.INFRA_TIMEOUT

    def test_sandbox_pattern_in_exception(self):
        result = classify_failure(exception_text="Connection refused to container endpoint")
        assert result == FailureCategory.INFRA_SANDBOX

    def test_oom_pattern_in_trajectory(self):
        result = classify_failure(trajectory_text='{"output": "Cannot allocate memory"}')
        assert result == FailureCategory.INFRA_OOM

    def test_exit_code_takes_precedence_over_text(self):
        # Exit code 137 (OOM) even though text says timeout
        result = classify_failure(exit_codes=[137], exception_text="timed out")
        assert result == FailureCategory.INFRA_OOM

    def test_capability_failure_default(self):
        result = classify_failure()
        assert result == FailureCategory.CAPABILITY

    def test_unknown_with_exception_no_infra_pattern(self):
        result = classify_failure(exception_text="AssertionError: expected 42 got 41")
        assert result == FailureCategory.UNKNOWN

    def test_multiple_exit_codes_first_match_wins(self):
        result = classify_failure(exit_codes=[1, 137])
        assert result == FailureCategory.INFRA_OOM

    def test_zero_exit_code_ignored(self):
        result = classify_failure(exit_codes=[0])
        assert result == FailureCategory.CAPABILITY

    def test_case_insensitive_patterns(self):
        result = classify_failure(exception_text="OUT OF MEMORY error occurred")
        assert result == FailureCategory.INFRA_OOM

    def test_signal_9_pattern(self):
        result = classify_failure(exception_text="Process was killed by signal 9")
        assert result == FailureCategory.INFRA_OOM

    def test_trajectory_only_no_infra_pattern_is_capability(self):
        result = classify_failure(trajectory_text='{"output": "test failed: expected 42"}')
        assert result == FailureCategory.CAPABILITY

    def test_bare_killed_does_not_false_positive(self):
        # "killed" alone should NOT trigger OOM (too broad)
        result = classify_failure(exception_text="Agent killed the background process")
        assert result == FailureCategory.UNKNOWN


class TestWilsonCI:
    """Tests for Wilson score confidence interval."""

    def test_zero_trials(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_all_successes(self):
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.95
        assert hi > 0.99

    def test_no_successes(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_half_success(self):
        lo, hi = wilson_ci(50, 100)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61

    def test_bounds_within_zero_one(self):
        for s in range(11):
            lo, hi = wilson_ci(s, 10)
            assert 0.0 <= lo <= hi <= 1.0

    def test_small_sample(self):
        lo, hi = wilson_ci(1, 3)
        # Wilson CI is wider than naive for small samples
        assert lo < 0.33
        assert hi > 0.33


class TestFormatCI:
    """Tests for confidence interval formatting."""

    def test_basic_format(self):
        result = format_ci(72, 100)
        assert "72.0%" in result
        assert "95% CI" in result
        assert "n=100" in result

    def test_zero_trials(self):
        result = format_ci(0, 0)
        assert "N/A" in result


class TestMinDetectableEffect:
    """Tests for minimum detectable effect size."""

    def test_zero_total(self):
        assert min_detectable_effect(0) == 1.0

    def test_larger_sample_smaller_mde(self):
        mde_small = min_detectable_effect(50)
        mde_large = min_detectable_effect(500)
        assert mde_large < mde_small

    def test_reasonable_range(self):
        # With 90 tasks, MDE should be roughly 10-15pp
        mde = min_detectable_effect(90)
        assert 0.05 < mde < 0.20


class TestInfraMetadata:
    """Tests for InfraMetadata serialization."""

    def test_to_dict_all_fields(self):
        meta = InfraMetadata(
            host_platform="Linux-6.1",
            host_python_version="3.12.0",
            sandbox_type="DockerEnvironment",
            sandbox_cpu_count=4,
            sandbox_memory_total_mb=8192,
            sandbox_memory_available_mb=4096,
            sandbox_os="Linux 6.1.0",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            concurrency_env="4",
            resource_config={"memory_limit": "16g"},
        )
        d = meta.to_dict()
        assert d["sandbox_cpu_count"] == 4
        assert d["sandbox_memory_total_mb"] == 8192
        assert d["resource_config"] == {"memory_limit": "16g"}

    def test_to_dict_defaults(self):
        meta = InfraMetadata()
        d = meta.to_dict()
        assert d["sandbox_cpu_count"] is None
        assert d["resource_config"] == {}


class TestFailureCategoryIsInfrastructure:
    """Tests for the is_infrastructure property."""

    def test_infra_oom(self):
        assert FailureCategory.INFRA_OOM.is_infrastructure is True

    def test_infra_timeout(self):
        assert FailureCategory.INFRA_TIMEOUT.is_infrastructure is True

    def test_infra_sandbox(self):
        assert FailureCategory.INFRA_SANDBOX.is_infrastructure is True

    def test_capability(self):
        assert FailureCategory.CAPABILITY.is_infrastructure is False

    def test_unknown(self):
        assert FailureCategory.UNKNOWN.is_infrastructure is False


class TestCollectHostMetadata:
    """Tests for host metadata collection."""

    def test_returns_expected_keys(self):
        result = collect_host_metadata()
        assert "host_platform" in result
        assert "host_python_version" in result
        assert isinstance(result["host_platform"], str)
        assert isinstance(result["host_python_version"], str)


@dataclass
class _FakeExecResult:
    output: str


class _FakeEnvironment:
    pass


class _FakeBackend(SandboxLike):
    """Minimal fake for HarborSandbox used by collect_sandbox_metadata."""

    environment: Any

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = responses
        self._idx = 0
        self.environment = _FakeEnvironment()

    async def aexecute(self, command: str, *, timeout: int | None = None) -> _FakeExecResult:  # noqa: ARG002, ASYNC109
        resp = self._responses[self._idx]
        self._idx += 1
        if isinstance(resp, Exception):
            raise resp
        return _FakeExecResult(output=resp)


class TestCollectSandboxMetadata:
    """Tests for sandbox metadata collection."""

    async def test_happy_path(self):
        backend = _FakeBackend(["4\n", "8192\n", "4096\n", "Linux 6.1.0\n"])
        meta = await collect_sandbox_metadata(backend)

        assert meta.sandbox_cpu_count == 4
        assert meta.sandbox_memory_total_mb == 8192
        assert meta.sandbox_memory_available_mb == 4096
        assert meta.sandbox_os == "Linux 6.1.0"
        assert meta.sandbox_type == "_FakeEnvironment"
        assert meta.timestamp_utc != ""

    async def test_non_numeric_output_sets_none(self):
        backend = _FakeBackend(
            [
                "nproc: command not found\n",
                "unknown\n",
                "unknown\n",
                "Linux 6.1.0\n",
            ]
        )
        meta = await collect_sandbox_metadata(backend)

        assert meta.sandbox_cpu_count is None
        assert meta.sandbox_memory_total_mb is None
        assert meta.sandbox_memory_available_mb is None
        assert meta.sandbox_os == "Linux 6.1.0"

    async def test_partial_failure_still_collects(self):
        backend = _FakeBackend(
            [
                RuntimeError("sandbox down"),  # CPU fails
                "8192\n",  # memory succeeds
                "4096\n",  # avail succeeds
                ConnectionError("reset"),  # OS fails
            ]
        )
        meta = await collect_sandbox_metadata(backend)

        assert meta.sandbox_cpu_count is None
        assert meta.sandbox_memory_total_mb == 8192
        assert meta.sandbox_memory_available_mb == 4096
        assert meta.sandbox_os == ""

    async def test_all_commands_fail(self):
        backend = _FakeBackend(
            [
                TimeoutError(),
                RuntimeError(),
                ConnectionError(),
                OSError(),
            ]
        )
        meta = await collect_sandbox_metadata(backend)

        assert meta.sandbox_cpu_count is None
        assert meta.sandbox_memory_total_mb is None
        assert meta.sandbox_memory_available_mb is None
        assert meta.sandbox_os == ""
        # Host info should still be populated
        assert meta.host_platform != ""


class TestExtractExitCodes:
    """Tests for exit code extraction from trajectory text."""

    def test_json_field(self):
        text = '{"exit_code": 137, "output": "killed"}'
        assert extract_exit_codes(text) == [137]

    def test_prose_format(self):
        text = "Command failed with exit code 124"
        assert extract_exit_codes(text) == [124]

    def test_zero_exit_code_filtered(self):
        assert extract_exit_codes('{"exit_code": 0}') == []

    def test_multiple_codes(self):
        text = '"exit_code": 1, later "exit_code": 137'
        codes = extract_exit_codes(text)
        assert 1 in codes
        assert 137 in codes

    def test_empty_string(self):
        assert extract_exit_codes("") == []

    def test_no_exit_codes(self):
        assert extract_exit_codes("some random output") == []
