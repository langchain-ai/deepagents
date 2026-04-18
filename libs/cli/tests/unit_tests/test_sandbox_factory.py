"""Tests for sandbox factory optional dependency handling."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from deepagents_cli.integrations.sandbox_factory import (
    _get_provider,
    get_default_working_dir,
    verify_sandbox_deps,
)


@pytest.mark.parametrize(
    ("provider", "package"),
    [
        ("daytona", "langchain-daytona"),
        ("modal", "langchain-modal"),
        ("runloop", "langchain-runloop"),
    ],
)
def test_get_provider_raises_helpful_error_for_missing_optional_dependency(
    provider: str,
    package: str,
) -> None:
    """Provider construction should explain which CLI extra to install."""
    error = (
        rf"The '{provider}' sandbox provider requires the "
        rf"'{package}' package"
    )
    with (
        patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.import_module",
            side_effect=ImportError("missing dependency"),
        ),
        pytest.raises(ImportError, match=error),
    ):
        _get_provider(provider)


def test_agentcore_get_or_create_raises_for_missing_dep() -> None:
    """AgentCore should explain which package to install."""
    error = (
        r"The 'agentcore' sandbox provider requires the "
        r"'langchain-agentcore-codeinterpreter' package"
    )

    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    with (
        patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.import_module",
            side_effect=ImportError("missing dependency"),
        ),
        pytest.raises(ImportError, match=error),
    ):
        provider.get_or_create()


def test_agentcore_raises_on_missing_aws_credentials() -> None:
    """AgentCore should raise ValueError without AWS creds."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = None
    with (
        patch.dict(sys.modules, {"boto3": mock_boto3}),
        pytest.raises(ValueError, match="AWS credentials not found"),
    ):
        _get_provider("agentcore")


def test_agentcore_rejects_sandbox_id() -> None:
    """AgentCore should raise NotImplementedError for sandbox_id."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    with pytest.raises(NotImplementedError, match="does not support reconnecting"):
        provider.get_or_create(sandbox_id="some-id")


def test_agentcore_init_without_boto3_does_not_raise() -> None:
    """Provider construction should succeed when boto3 is not installed."""
    env_clear = dict.fromkeys(("AWS_REGION", "AWS_DEFAULT_REGION"), "")
    with (
        patch.dict(sys.modules, {"boto3": None}),
        patch.dict(os.environ, env_clear, clear=False),
    ):
        for k in env_clear:
            os.environ.pop(k, None)
        provider = _get_provider("agentcore")
    assert provider._region == "us-west-2"  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"AWS_REGION": "eu-west-1"}, "eu-west-1"),
        ({"AWS_DEFAULT_REGION": "ap-southeast-1"}, "ap-southeast-1"),
        (
            {"AWS_REGION": "us-east-1", "AWS_DEFAULT_REGION": "eu-west-1"},
            "us-east-1",
        ),
        ({}, "us-west-2"),
    ],
)
def test_agentcore_region_resolution(env: dict[str, str], expected: str) -> None:
    """Region should follow AWS_REGION > AWS_DEFAULT_REGION > us-west-2."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with (
        patch.dict(sys.modules, {"boto3": mock_boto3}),
        patch.dict(os.environ, env, clear=False),
        patch.dict(
            os.environ,
            {k: "" for k in ("AWS_REGION", "AWS_DEFAULT_REGION") if k not in env},
            clear=False,
        ),
    ):
        # Clear env vars not in this test case
        for k in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            if k not in env:
                os.environ.pop(k, None)
        provider = _get_provider("agentcore")
    assert provider._region == expected  # ty: ignore[unresolved-attribute]


def test_agentcore_get_or_create_happy_path() -> None:
    """Successful get_or_create should start interpreter and track it."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    mock_interpreter = MagicMock()
    mock_backend = MagicMock()
    mock_backend.id = "session-123"

    mock_ci_module = MagicMock()
    mock_ci_module.CodeInterpreter.return_value = mock_interpreter
    mock_backend_module = MagicMock()
    mock_backend_module.AgentCoreSandbox.return_value = mock_backend

    def fake_import(module_name: str, **_: object) -> MagicMock:
        if "code_interpreter_client" in module_name:
            return mock_ci_module
        return mock_backend_module

    with patch(
        "deepagents_cli.integrations.sandbox_factory._import_provider_module",
        side_effect=fake_import,
    ):
        result = provider.get_or_create()

    mock_interpreter.start.assert_called_once()
    assert result is mock_backend
    assert provider._active_interpreters["session-123"] is mock_interpreter  # ty: ignore[unresolved-attribute]


def test_agentcore_start_failure_cleans_up() -> None:
    """If interpreter.start() fails, interpreter.stop() should be called."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    mock_interpreter = MagicMock()
    mock_interpreter.start.side_effect = RuntimeError("connection failed")

    mock_ci_module = MagicMock()
    mock_ci_module.CodeInterpreter.return_value = mock_interpreter

    def fake_import(module_name: str, **_: object) -> MagicMock:
        if "code_interpreter_client" in module_name:
            return mock_ci_module
        return MagicMock()

    with (
        patch(
            "deepagents_cli.integrations.sandbox_factory._import_provider_module",
            side_effect=fake_import,
        ),
        pytest.raises(RuntimeError, match="connection failed"),
    ):
        provider.get_or_create()

    mock_interpreter.stop.assert_called_once()
    assert not provider._active_interpreters  # ty: ignore[unresolved-attribute]


def test_agentcore_delete_stops_tracked_interpreter() -> None:
    """delete() should call stop() on a tracked interpreter."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    mock_interpreter = MagicMock()
    provider._active_interpreters["sess-1"] = mock_interpreter  # ty: ignore[unresolved-attribute]

    provider.delete(sandbox_id="sess-1")

    mock_interpreter.stop.assert_called_once()
    assert "sess-1" not in provider._active_interpreters  # ty: ignore[unresolved-attribute]


def test_agentcore_delete_swallows_stop_exception() -> None:
    """delete() should not propagate if interpreter.stop() raises."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    mock_interpreter = MagicMock()
    mock_interpreter.stop.side_effect = RuntimeError("network error")
    provider._active_interpreters["sess-1"] = mock_interpreter  # ty: ignore[unresolved-attribute]

    provider.delete(sandbox_id="sess-1")  # should not raise
    assert "sess-1" not in provider._active_interpreters  # ty: ignore[unresolved-attribute]


def test_agentcore_delete_untracked_session() -> None:
    """delete() should not raise for an untracked session ID."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    provider.delete(sandbox_id="nonexistent")  # should not raise


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("agentcore", "/tmp"),
        ("daytona", "/home/daytona"),
        ("langsmith", "/tmp"),
        ("modal", "/workspace"),
        ("runloop", "/home/user"),
    ],
)
def test_get_default_working_dir(provider: str, expected: str) -> None:
    """Each provider should map to the correct default working directory."""
    assert get_default_working_dir(provider) == expected


class TestVerifySandboxDeps:
    """Tests for the early sandbox dependency check."""

    @pytest.mark.parametrize(
        ("provider", "expected_module"),
        [
            ("agentcore", "langchain_agentcore_codeinterpreter"),
            ("daytona", "langchain_daytona"),
            ("modal", "langchain_modal"),
            ("runloop", "langchain_runloop"),
        ],
    )
    def test_raises_import_error_when_backend_missing(
        self, provider: str, expected_module: str
    ) -> None:
        """Should raise ImportError with install instructions."""
        mock_find_spec = patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
            return_value=None,
        )
        with (
            mock_find_spec as find_spec,
            pytest.raises(
                ImportError,
                match=rf"Missing dependencies for '{provider}' sandbox.*"
                rf"pip install 'deepagents-cli\[{provider}\]'",
            ),
        ):
            verify_sandbox_deps(provider)

        find_spec.assert_called_once_with(expected_module)

    @pytest.mark.parametrize(
        "provider",
        ["agentcore", "daytona", "modal", "runloop"],
    )
    def test_passes_when_backend_installed(self, provider: str) -> None:
        """Should not raise when the backend module is found."""
        spec_sentinel = object()
        with patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
            return_value=spec_sentinel,
        ):
            verify_sandbox_deps(provider)  # should not raise

    @pytest.mark.parametrize(
        "exc_cls",
        [ImportError, ValueError],
    )
    def test_raises_when_find_spec_throws(self, exc_cls: type) -> None:
        """find_spec can raise ImportError/ValueError in corrupted envs."""
        with (
            patch(
                "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
                side_effect=exc_cls("broken"),
            ),
            pytest.raises(ImportError, match="Missing dependencies"),
        ):
            verify_sandbox_deps("daytona")

    @pytest.mark.parametrize("provider", ["none", "langsmith", "", None])
    def test_skips_builtin_and_empty_providers(self, provider: str | None) -> None:
        """Built-in and empty providers should be silently accepted."""
        verify_sandbox_deps(provider)  # type: ignore[arg-type]

    def test_skips_unknown_provider(self) -> None:
        """Unknown providers are passed through for downstream handling."""
        verify_sandbox_deps("unknown_provider")  # should not raise


class TestLangSmithSnapshotResolution:
    """Env-var-driven snapshot resolution in `_LangSmithProvider.get_or_create`."""

    @staticmethod
    def _make_ready_sandbox() -> MagicMock:
        """Mock Sandbox whose readiness poll succeeds immediately."""
        sandbox = MagicMock()
        sandbox.run.return_value = MagicMock(exit_code=0)
        return sandbox

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Mock SandboxClient that yields a ready sandbox from create_sandbox."""
        client = MagicMock()
        client.create_sandbox.return_value = self._make_ready_sandbox()
        return client

    @pytest.fixture
    def provider(self, mock_client: MagicMock, monkeypatch: pytest.MonkeyPatch):
        """Build `_LangSmithProvider` with its SandboxClient patched."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "fake")
        with patch("langsmith.sandbox.SandboxClient", return_value=mock_client):
            from deepagents_cli.integrations.sandbox_factory import (
                _LangSmithProvider,
            )

            return _LangSmithProvider()

    def test_snapshot_id_env_var_boots_directly_without_listing(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`LANGSMITH_SANDBOX_SNAPSHOT_ID` skips name lookup and auto-build."""
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", "snap-abc123")
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", raising=False)

        provider.get_or_create()

        mock_client.list_snapshots.assert_not_called()
        mock_client.create_snapshot.assert_not_called()
        mock_client.create_sandbox.assert_called_once()
        kwargs = mock_client.create_sandbox.call_args.kwargs
        assert kwargs["snapshot_id"] == "snap-abc123"

    def test_snapshot_name_env_var_overrides_default(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`LANGSMITH_SANDBOX_SNAPSHOT_NAME` is used as the lookup name."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", "custom-snap")

        existing = MagicMock(name="custom-snap", id="snap-xyz", status="ready")
        existing.name = "custom-snap"
        mock_client.list_snapshots.return_value = [existing]

        provider.get_or_create()

        mock_client.list_snapshots.assert_called_once()
        mock_client.create_snapshot.assert_not_called()
        assert mock_client.create_sandbox.call_args.kwargs["snapshot_id"] == "snap-xyz"

    def test_snapshot_name_env_var_triggers_build_when_missing(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unknown snapshot name triggers `create_snapshot` with that name."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", "built-snap")
        mock_client.list_snapshots.return_value = []
        built = MagicMock(id="snap-built")
        mock_client.create_snapshot.return_value = built

        provider.get_or_create()

        mock_client.create_snapshot.assert_called_once()
        assert mock_client.create_snapshot.call_args.kwargs["name"] == "built-snap"
        kwargs = mock_client.create_sandbox.call_args.kwargs
        assert kwargs["snapshot_id"] == "snap-built"

    def test_snapshot_id_wins_over_name(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`_ID` takes precedence — `_NAME` is ignored when both are set."""
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", "snap-id-wins")
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", "ignored-name")

        provider.get_or_create()

        mock_client.list_snapshots.assert_not_called()
        kwargs = mock_client.create_sandbox.call_args.kwargs
        assert kwargs["snapshot_id"] == "snap-id-wins"

    def test_defaults_when_no_env_vars(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no env vars, falls back to `deepagents-cli` snapshot name."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", raising=False)
        mock_client.list_snapshots.return_value = []
        mock_client.create_snapshot.return_value = MagicMock(id="snap-default")

        provider.get_or_create()

        assert mock_client.create_snapshot.call_args.kwargs["name"] == "deepagents-cli"
