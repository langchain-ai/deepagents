"""Tests for sandbox factory optional dependency handling."""

from __future__ import annotations

import contextlib
import os
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code.integrations.sandbox_config import SandboxConfig
from deepagents_code.integrations.sandbox_factory import (
    _VERCEL_SANDBOX_TIMEOUT,
    _AgentCoreProvider,
    _get_provider,
    _ModalProvider,
    _VercelProvider,
    create_sandbox,
    get_default_working_dir,
    verify_sandbox_deps,
)
from deepagents_code.integrations.sandbox_registry import SandboxRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

_FACTORY = "deepagents_code.integrations.sandbox_factory"


@contextlib.contextmanager
def _bind_environment(environment: Mapping[str, str]) -> Iterator[None]:
    """Bind one workspace environment everywhere the factory reads it.

    `sandbox_factory` aliases `active_environment` at import, while
    `resolve_env_var` calls it through `deepagents_code.config`. Both resolve
    to the same binding in production, so a test that patches only the alias
    lets the two disagree.

    Yields:
        `None`, with both references bound to `environment`.
    """
    with (
        patch(f"{_FACTORY}.active_environment", return_value=environment),
        patch(
            "deepagents_code.config.active_environment",
            return_value=environment,
        ),
    ):
        yield


def _registry_with(config: SandboxConfig) -> SandboxRegistry:
    """Build a deterministic registry (no entry-point discovery) from config."""
    return SandboxRegistry(config=config, include_entry_points=False)


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
            "deepagents_code.integrations.sandbox_factory.importlib.import_module",
            side_effect=ImportError("missing dependency"),
        ),
        pytest.raises(ImportError, match=error),
    ):
        _get_provider(provider)


def test_create_sandbox_rejects_snapshot_name_for_other_providers() -> None:
    """Snapshot names only apply to LangSmith and Runloop."""
    provider = MagicMock()

    with (
        patch(
            "deepagents_code.integrations.sandbox_factory._get_provider",
            return_value=provider,
        ),
        pytest.raises(
            ValueError,
            match="snapshot_name is not supported by provider 'modal'",
        ),
        create_sandbox("modal", snapshot_name="custom-snap"),
    ):
        pass

    provider.get_or_create.assert_not_called()


@pytest.mark.parametrize("provider_name", ["langsmith", "runloop"])
def test_create_sandbox_rejects_snapshot_name_with_sandbox_id(
    provider_name: str,
) -> None:
    """Snapshots are only meaningful for fresh sandboxes, not re-attach."""
    provider = MagicMock()

    with (
        patch(
            "deepagents_code.integrations.sandbox_factory._get_provider",
            return_value=provider,
        ),
        pytest.raises(ValueError, match="cannot be combined with sandbox_id"),
        create_sandbox(
            provider_name,
            sandbox_id="sb-existing",
            snapshot_name="custom-snap",
        ),
    ):
        pass

    provider.get_or_create.assert_not_called()


def test_runloop_provider_raises_sandbox_not_found_for_missing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing devbox ID (surfaced as `KeyError`) maps to `SandboxNotFoundError`.

    `RunloopProvider` translates the SDK's `NotFoundError` to a `KeyError`, so the
    factory only ever sees the builtin and stays free of an SDK import.
    """
    from deepagents_code.integrations.sandbox_factory import _RunloopProvider
    from deepagents_code.integrations.sandbox_provider import SandboxNotFoundError

    fake_provider = MagicMock()
    fake_provider.get_or_create.side_effect = KeyError("missing-dev")
    fake_module = MagicMock()
    fake_module.RunloopProvider.return_value = fake_provider

    monkeypatch.setenv("RUNLOOP_API_KEY", "test-key")
    with patch(
        "deepagents_code.integrations.sandbox_factory._import_provider_module",
        return_value=fake_module,
    ):
        provider = _RunloopProvider()
        with pytest.raises(SandboxNotFoundError, match="missing-dev"):
            provider.get_or_create(sandbox_id="missing-dev")


def test_runloop_provider_reraises_keyerror_without_sandbox_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `KeyError` with no `sandbox_id` is not mislabeled as `SandboxNotFoundError`."""
    from deepagents_code.integrations.sandbox_factory import _RunloopProvider

    fake_provider = MagicMock()
    fake_provider.get_or_create.side_effect = KeyError("unexpected")
    fake_module = MagicMock()
    fake_module.RunloopProvider.return_value = fake_provider

    monkeypatch.setenv("RUNLOOP_API_KEY", "test-key")
    with patch(
        "deepagents_code.integrations.sandbox_factory._import_provider_module",
        return_value=fake_module,
    ):
        provider = _RunloopProvider()
        with pytest.raises(KeyError):
            provider.get_or_create(sandbox_id=None)


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
            "deepagents_code.integrations.sandbox_factory.importlib.import_module",
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


def test_agentcore_uses_workspace_aws_session() -> None:
    """AgentCore receives a session built from workspace AWS settings."""
    environment = {
        "AWS_REGION": "us-test-1",
        "AWS_PROFILE": "workspace-profile",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_SESSION_TOKEN": "test-session-token",
    }
    session = MagicMock()
    session.get_credentials.return_value = MagicMock()
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value = session
    interpreter = MagicMock()
    client_module = MagicMock()
    client_module.CodeInterpreter.return_value = interpreter
    backend_module = MagicMock()
    backend_module.AgentCoreSandbox.return_value.id = "sandbox-id"

    with (
        _bind_environment(environment),
        patch.dict(sys.modules, {"boto3": mock_boto3}),
    ):
        provider = _AgentCoreProvider()

    with patch(
        f"{_FACTORY}._import_provider_module",
        side_effect=[client_module, backend_module],
    ):
        provider.get_or_create()

    mock_boto3.Session.assert_called_once_with(
        profile_name="workspace-profile",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
        region_name="us-test-1",
    )
    client_module.CodeInterpreter.assert_called_once_with(
        region="us-test-1",
        session=session,
        integration_source="deepagents-code",
    )


def test_agentcore_rejects_sandbox_id() -> None:
    """AgentCore should raise NotImplementedError for sandbox_id."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    with pytest.raises(NotImplementedError, match="does not support reconnecting"):
        provider.get_or_create(sandbox_id="some-id")


def test_agentcore_delete_untracked_session() -> None:
    """delete() should not raise for an untracked session ID."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    provider.delete(sandbox_id="nonexistent")  # should not raise


class TestVerifySandboxDeps:
    """Tests for the early sandbox dependency check."""

    @pytest.mark.parametrize(
        "provider",
        ["agentcore", "daytona", "modal", "runloop"],
    )
    def test_passes_when_backend_installed(self, provider: str) -> None:
        """Should not raise when the backend module is found."""
        spec_sentinel = object()
        with patch(
            "deepagents_code.integrations.sandbox_factory.importlib.util.find_spec",
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
                "deepagents_code.integrations.sandbox_factory.importlib.util.find_spec",
                side_effect=exc_cls("broken"),
            ),
            pytest.raises(ImportError, match="Missing dependencies"),
        ):
            verify_sandbox_deps("daytona")

    @pytest.mark.parametrize("provider", ["none", "langsmith", "", None])
    def test_skips_builtin_and_empty_providers(self, provider: str | None) -> None:
        """Built-in and empty providers should be silently accepted."""
        verify_sandbox_deps(provider)  # ty: ignore

    def test_skips_unknown_provider(self) -> None:
        """Unknown providers are passed through for downstream handling."""
        verify_sandbox_deps("unknown_provider")  # should not raise

    def test_config_override_of_builtin_uses_package_hint(self) -> None:
        """Overriding a built-in keeps its probe module and uses the package."""
        config = SandboxConfig(
            providers={"daytona": {"class_path": "x:Y", "package": "my-daytona"}}
        )
        with (
            patch(f"{_FACTORY}._get_registry", return_value=_registry_with(config)),
            patch(
                f"{_FACTORY}.importlib.util.find_spec",
                return_value=None,
            ),
            pytest.raises(
                ImportError,
                match=r"Missing dependencies for 'daytona'.*"
                r"/install my-daytona --package",
            ),
        ):
            verify_sandbox_deps("daytona")


class TestGetDefaultWorkingDirRegistry:
    """Tests for `get_default_working_dir` resolving through the registry."""

    def test_config_override(self) -> None:
        config = SandboxConfig(
            providers={"acme": {"class_path": "x:Y", "working_dir": "/cfg-wd"}}
        )
        with patch(f"{_FACTORY}._get_registry", return_value=_registry_with(config)):
            assert get_default_working_dir("acme") == "/cfg-wd"

    def test_unknown_provider_raises(self) -> None:
        with (
            patch(
                f"{_FACTORY}._get_registry",
                return_value=_registry_with(SandboxConfig()),
            ),
            pytest.raises(ValueError, match="Unknown sandbox provider: nope"),
        ):
            get_default_working_dir("nope")


class TestVercelProvider:
    """Tests for basic Vercel sandbox provider lifecycle."""

    @staticmethod
    def _clear_vercel_env(monkeypatch: pytest.MonkeyPatch) -> None:
        """Remove Vercel env vars that affect SDK kwargs."""
        for name in (
            "VERCEL_TOKEN",
            "DEEPAGENTS_CODE_VERCEL_TOKEN",
            "VERCEL_OIDC_TOKEN",
            "DEEPAGENTS_CODE_VERCEL_OIDC_TOKEN",
            "VERCEL_PROJECT_ID",
            "DEEPAGENTS_CODE_VERCEL_PROJECT_ID",
            "VERCEL_TEAM_ID",
            "DEEPAGENTS_CODE_VERCEL_TEAM_ID",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_get_provider_succeeds_without_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Vercel auth errors should be left to the SDK."""
        self._clear_vercel_env(monkeypatch)

        provider = _get_provider("vercel")

        assert isinstance(provider, _VercelProvider)

    def test_get_or_create_raises_helpful_error_for_missing_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Vercel should explain which package to install."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")

        with (
            patch(
                "deepagents_code.integrations.sandbox_factory.importlib.import_module",
                side_effect=ImportError("missing dependency"),
            ),
            pytest.raises(
                ImportError,
                match=(
                    r"The 'vercel' sandbox provider requires the "
                    r"'langchain-vercel-sandbox' package"
                ),
            ),
        ):
            provider.get_or_create()

    @pytest.mark.parametrize("sandbox_id", [None, "sb_existing"])
    def test_create_and_attach_sdk_errors_do_not_expose_secrets(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sandbox_id: str | None,
    ) -> None:
        """Create and attach failures use fixed messages."""
        secret = "sdk-leaked-secret"
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TOKEN", "runtime-token")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_PROJECT_ID", "runtime-project")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TEAM_ID", "runtime-team")
        provider = _VercelProvider()
        vercel_sdk = MagicMock()
        if sandbox_id is None:
            vercel_sdk.Sandbox.create.side_effect = RuntimeError(secret)
        else:
            vercel_sdk.Sandbox.get.side_effect = RuntimeError(secret)

        with (
            patch(
                f"{_FACTORY}._import_provider_module",
                return_value=vercel_sdk,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            provider.get_or_create(sandbox_id=sandbox_id)

        assert secret not in str(exc_info.value)
        assert "runtime-token" not in str(exc_info.value)
        # The original SDK error is preserved as the cause so developer
        # tracebacks retain root cause even though the message is redacted.
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == secret

    def test_delete_sdk_error_does_not_expose_secrets(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Delete failures use fixed messages."""
        secret = "sdk-leaked-secret"
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TOKEN", "runtime-token")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_PROJECT_ID", "runtime-project")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TEAM_ID", "runtime-team")
        provider = _VercelProvider()
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.get.side_effect = RuntimeError(secret)

        with (
            patch(
                f"{_FACTORY}._import_provider_module",
                return_value=vercel_sdk,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            provider.delete(sandbox_id="sb_123")

        assert str(exc_info.value) == "Failed to stop Vercel sandbox."
        assert secret not in str(exc_info.value)

    def test_wait_sdk_error_does_not_expose_secrets(self) -> None:
        """Readiness failures from the SDK use fixed messages."""
        secret = "sdk-leaked-secret"
        provider = _VercelProvider()
        sandbox = MagicMock(sandbox_id="sb_123", status="pending")
        sandbox.wait_for_status.side_effect = RuntimeError(secret)
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.create.return_value = sandbox
        vercel_backend = MagicMock()

        def fake_import(module_name: str, **_: object) -> MagicMock:
            if module_name == "vercel.sandbox":
                return vercel_sdk
            return vercel_backend

        with (
            patch(
                f"{_FACTORY}._import_provider_module",
                side_effect=fake_import,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            provider.get_or_create()

        assert str(exc_info.value) == "Failed while waiting for Vercel sandbox startup."
        assert secret not in str(exc_info.value)

    def test_readiness_failure_cleans_up_fresh_sandbox(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fresh Vercel sandboxes are stopped when readiness fails."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_123", status="failed")
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.create.return_value = sandbox
        vercel_backend = MagicMock()

        def fake_import(module_name: str, **_: object) -> MagicMock:
            if module_name == "vercel.sandbox":
                return vercel_sdk
            return vercel_backend

        with (
            patch(
                "deepagents_code.integrations.sandbox_factory._import_provider_module",
                side_effect=fake_import,
            ),
            pytest.raises(RuntimeError, match="terminal state"),
        ):
            provider.get_or_create()

        sandbox.stop.assert_called_once_with()

    def test_generic_readiness_failure_stops_fresh_sandbox(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-Timeout readiness error stops a freshly created sandbox."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_123", status="pending")
        sandbox.wait_for_status.side_effect = RuntimeError("boom")
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.create.return_value = sandbox
        vercel_backend = MagicMock()

        def fake_import(module_name: str, **_: object) -> MagicMock:
            if module_name == "vercel.sandbox":
                return vercel_sdk
            return vercel_backend

        with (
            patch(
                f"{_FACTORY}._import_provider_module",
                side_effect=fake_import,
            ),
            pytest.raises(RuntimeError, match="Failed while waiting"),
        ):
            provider.get_or_create()

        sandbox.stop.assert_called_once_with()

    def test_mixed_source_credentials_are_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single prefixed override completes via canonical values."""
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TOKEN", "token_prefixed")
        monkeypatch.setenv("VERCEL_PROJECT_ID", "project_canonical")
        monkeypatch.setenv("VERCEL_TEAM_ID", "team_canonical")
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_123", status="running")
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.create.return_value = sandbox
        vercel_backend = MagicMock()

        def fake_import(module_name: str, **_: object) -> MagicMock:
            if module_name == "vercel.sandbox":
                return vercel_sdk
            return vercel_backend

        with patch(
            f"{_FACTORY}._import_provider_module",
            side_effect=fake_import,
        ):
            provider.get_or_create()

        vercel_sdk.Sandbox.create.assert_called_once_with(
            runtime="python3.13",
            timeout=_VERCEL_SANDBOX_TIMEOUT,
            token="token_prefixed",
            project_id="project_canonical",
            team_id="team_canonical",
        )


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
            from deepagents_code.integrations.sandbox_factory import (
                _LangSmithProvider,
            )

            return _LangSmithProvider()


def test_setup_script_expands_workspace_environment() -> None:
    """The setup script sees the workspace `.env`, not the server process env."""
    from deepagents_code.integrations.sandbox_factory import _run_sandbox_setup

    backend = MagicMock()
    backend.execute.return_value = MagicMock(exit_code=0, output="")
    script = MagicMock()
    script.read_text.return_value = "echo ${WORKSPACE_ONLY} ${SERVER_ONLY}"

    with (
        patch(f"{_FACTORY}.Path", return_value=script),
        patch(
            f"{_FACTORY}.active_environment",
            return_value={"WORKSPACE_ONLY": "from-project-dotenv"},
        ),
        patch.dict(
            "os.environ",
            {"SERVER_ONLY": "server-secret", "WORKSPACE_ONLY": "server-value"},
            clear=False,
        ),
    ):
        script.exists.return_value = True
        _run_sandbox_setup(backend, "setup.sh")

    command = backend.execute.call_args[0][0]
    assert "from-project-dotenv" in command
    # The server process's values must not leak into the sandbox.
    assert "server-secret" not in command
    assert "server-value" not in command


def test_vercel_override_gate_reads_workspace_environment() -> None:
    """A prefixed override from the workspace `.env` still triggers the gate."""
    environment = {
        "DEEPAGENTS_CODE_VERCEL_TOKEN": "workspace-token",
        "DEEPAGENTS_CODE_VERCEL_PROJECT_ID": "workspace-project",
        "DEEPAGENTS_CODE_VERCEL_TEAM_ID": "workspace-team",
    }
    with (
        patch(f"{_FACTORY}.active_environment", return_value=environment),
        patch(
            "deepagents_code.model_config.resolve_env_var",
            side_effect=lambda name: environment.get(f"DEEPAGENTS_CODE_{name}"),
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        kwargs = _VercelProvider._resolve_sdk_kwargs()

    assert kwargs == {
        "token": "workspace-token",
        "project_id": "workspace-project",
        "team_id": "workspace-team",
    }


def test_vercel_uses_an_unprefixed_workspace_credential() -> None:
    """Canonical `VERCEL_*` names from a workspace `.env` must not be dropped.

    `_build_server_env` strips the client's project `.env` from the server
    process, so these never reach the SDK's own `os.environ` read.
    """
    environment = {
        "VERCEL_TOKEN": "workspace-token",
        "VERCEL_PROJECT_ID": "workspace-project",
        "VERCEL_TEAM_ID": "workspace-team",
    }
    with (
        patch(f"{_FACTORY}.active_environment", return_value=environment),
        patch(
            "deepagents_code.model_config.resolve_env_var",
            side_effect=environment.get,
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        kwargs = _VercelProvider._resolve_sdk_kwargs()

    assert kwargs == {
        "token": "workspace-token",
        "project_id": "workspace-project",
        "team_id": "workspace-team",
    }


@pytest.mark.parametrize(
    "server",
    [
        pytest.param({}, id="empty"),
        pytest.param(
            {
                "VERCEL_OIDC_TOKEN": "test-oidc-token",
                "VERCEL_PROJECT_ID": "server-project",
            },
            id="oidc-project",
        ),
        pytest.param(
            {"VERCEL_OIDC_TOKEN": "test-oidc-token", "VERCEL_TEAM_ID": "server-team"},
            id="oidc-team",
        ),
        pytest.param(
            {
                "VERCEL_OIDC_TOKEN": "test-oidc-token",
                "VERCEL_PROJECT_ID": "server-project",
                "VERCEL_TEAM_ID": "server-team",
            },
            id="oidc-project-and-team",
        ),
        pytest.param(
            {"VERCEL_TOKEN": "personal-token", "VERCEL_PROJECT_ID": "prj_1"},
            id="personal-token-and-project",
        ),
        pytest.param({"VERCEL_TOKEN": "personal-token"}, id="personal-token"),
        pytest.param(
            {"VERCEL_PROJECT_ID": "prj_1", "VERCEL_TEAM_ID": "team_1"},
            id="project-and-team",
        ),
    ],
)
def test_vercel_delegates_unchanged_server_credentials(server: dict[str, str]) -> None:
    """Inherited credentials remain SDK-managed, including partial sets and OIDC."""
    with (
        _bind_environment(server),
        patch.dict("os.environ", server, clear=True),
    ):
        assert _VercelProvider._resolve_sdk_kwargs() == {}


def test_vercel_fails_closed_when_the_workspace_overrides_part_of_the_set() -> None:
    """A workspace override differs from the server, so it must not delegate."""
    with (
        _bind_environment(
            {"VERCEL_TOKEN": "workspace-token", "VERCEL_PROJECT_ID": "prj_1"}
        ),
        patch.dict(
            "os.environ",
            {"VERCEL_TOKEN": "server-token", "VERCEL_PROJECT_ID": "prj_1"},
            clear=True,
        ),
        pytest.raises(ValueError, match="VERCEL_TEAM_ID not set"),
    ):
        _VercelProvider._resolve_sdk_kwargs()


@pytest.mark.parametrize("server_oidc", [False, True])
def test_vercel_oidc_does_not_discard_workspace_identifiers(
    server_oidc: bool,
) -> None:
    """Delegating cannot drop workspace settings the SDK cannot read."""
    environment = {
        "VERCEL_OIDC_TOKEN": "test-oidc-token",
        "VERCEL_PROJECT_ID": "workspace-project",
    }
    server = {"VERCEL_OIDC_TOKEN": "test-oidc-token"} if server_oidc else {}
    with (
        _bind_environment(environment),
        patch.dict("os.environ", server, clear=True),
        pytest.raises(ValueError, match="workspace Vercel configuration"),
    ):
        _VercelProvider._resolve_sdk_kwargs()


@pytest.mark.parametrize("oidc", [None, "test-oidc-token"])
@pytest.mark.parametrize("prefix", ["", "DEEPAGENTS_CODE_"])
def test_vercel_fails_closed_on_a_partial_workspace_credential_set(
    oidc: str | None,
    prefix: str,
) -> None:
    """A partial set must not fall back to the server's Vercel identity.

    An empty mapping hands auth back to the Vercel SDK, which resolves
    credentials from the server process (`VERCEL_*` in its own environment,
    or its OIDC identity). A workspace that pinned a restricted token would
    then silently run its sandbox under the server's broader identity.
    """
    environment = {
        f"{prefix}VERCEL_TOKEN": "workspace-token",
    }
    with (
        _bind_environment(environment),
        patch.dict(
            "os.environ", {"VERCEL_OIDC_TOKEN": oidc} if oidc else {}, clear=True
        ),
        pytest.raises(ValueError, match="VERCEL_PROJECT_ID, and VERCEL_TEAM_ID"),
    ):
        _VercelProvider._resolve_sdk_kwargs()


def test_vercel_get_provider_fails_closed_on_a_partial_set() -> None:
    """The constructor propagates the partial-set failure.

    `create_sandbox` surfaces `ValueError` as a startup error, so the server
    refuses the sandbox rather than running it under substituted credentials.
    """
    with (
        _bind_environment(
            {
                "DEEPAGENTS_CODE_VERCEL_TOKEN": "workspace-token",
                "DEEPAGENTS_CODE_VERCEL_TEAM_ID": "workspace-team",
            }
        ),
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="workspace Vercel configuration"),
    ):
        _get_provider("vercel")


@pytest.mark.parametrize("prefix", ["", "DEEPAGENTS_CODE_"])
@pytest.mark.parametrize(
    "overrides",
    [
        ("TOKEN",),
        ("PROJECT_ID",),
        ("TEAM_ID",),
        ("TOKEN", "PROJECT_ID"),
        ("TOKEN", "TEAM_ID"),
        ("PROJECT_ID", "TEAM_ID"),
    ],
)
def test_vercel_rejects_mixed_workspace_and_server_credentials(
    prefix: str, overrides: tuple[str, ...]
) -> None:
    """A complete merged mapping can still contain two credential identities."""
    server = {
        f"VERCEL_{key}": f"server-{key}" for key in ("TOKEN", "PROJECT_ID", "TEAM_ID")
    }
    environment = {
        **server,
        **{f"{prefix}VERCEL_{key}": f"workspace-{key}" for key in overrides},
    }
    with (
        _bind_environment(environment),
        patch.dict("os.environ", server, clear=True),
        pytest.raises(ValueError, match="mixes workspace and server"),
    ):
        _get_provider("vercel")


def test_vercel_empty_workspace_overrides_cannot_restore_server_auth() -> None:
    """Explicitly clearing every credential must not reactivate ambient auth."""
    server = {"VERCEL_TOKEN": "server-token"}
    with (
        _bind_environment({**server, "DEEPAGENTS_CODE_VERCEL_TOKEN": ""}),
        patch.dict("os.environ", server, clear=True),
        pytest.raises(ValueError, match="workspace Vercel configuration is incomplete"),
    ):
        _get_provider("vercel")


def test_vercel_complete_prefixed_set_can_share_server_identifiers() -> None:
    """Explicit workspace IDs remain scoped even when their values match."""
    server = {"VERCEL_PROJECT_ID": "shared-project", "VERCEL_TEAM_ID": "shared-team"}
    environment = {
        **server,
        "DEEPAGENTS_CODE_VERCEL_TOKEN": "workspace-token",
        "DEEPAGENTS_CODE_VERCEL_PROJECT_ID": "shared-project",
        "DEEPAGENTS_CODE_VERCEL_TEAM_ID": "shared-team",
    }
    with _bind_environment(environment), patch.dict("os.environ", server, clear=True):
        assert _VercelProvider._resolve_sdk_kwargs() == {
            "token": "workspace-token",
            "project_id": "shared-project",
            "team_id": "shared-team",
        }


def test_agentcore_omits_session_when_it_could_not_be_built() -> None:
    """A failed session must not masquerade as an applied workspace session."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.side_effect = RuntimeError("ProfileNotFound: typo")
    interpreter = MagicMock()
    client_module = MagicMock()
    client_module.CodeInterpreter.return_value = interpreter
    backend_module = MagicMock()
    backend_module.AgentCoreSandbox.return_value.id = "sandbox-id"

    with (
        _bind_environment({"AWS_REGION": "us-test-1"}),
        patch.dict(sys.modules, {"boto3": mock_boto3}),
    ):
        provider = _AgentCoreProvider()

    with patch(
        f"{_FACTORY}._import_provider_module",
        side_effect=[client_module, backend_module],
    ):
        provider.get_or_create()

    client_module.CodeInterpreter.assert_called_once_with(
        region="us-test-1",
        integration_source="deepagents-code",
    )
    assert "session" not in client_module.CodeInterpreter.call_args.kwargs


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"),
    ],
)
def test_agentcore_rejects_a_half_set_access_key_pair(
    present: str, missing: str
) -> None:
    """Either missing credential is named before any boto3 session is built."""
    mock_boto3 = MagicMock()

    with (
        _bind_environment({"AWS_REGION": "us-test-1", present: "workspace-value"}),
        patch.dict(sys.modules, {"boto3": mock_boto3}),
        pytest.raises(ValueError, match=f"{missing} is not set"),
    ):
        _AgentCoreProvider()

    mock_boto3.Session.assert_not_called()


def test_agentcore_rejects_a_session_token_without_the_key_pair() -> None:
    """Botocore declines the explicit provider and falls through to the server."""
    mock_boto3 = MagicMock()

    with (
        _bind_environment(
            {"AWS_REGION": "us-test-1", "AWS_SESSION_TOKEN": "only-the-token"}
        ),
        patch.dict(sys.modules, {"boto3": mock_boto3}),
        pytest.raises(ValueError, match="AWS_SESSION_TOKEN is set without"),
    ):
        _AgentCoreProvider()

    mock_boto3.Session.assert_not_called()


def test_agentcore_does_not_blame_the_workspace_for_a_server_level_profile() -> None:
    """A stale profile from the server's own shell must not be a startup failure."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.side_effect = RuntimeError("ProfileNotFound: stale")
    environment = {"AWS_REGION": "us-test-1", "AWS_PROFILE": "stale-server-profile"}

    with (
        _bind_environment(environment),
        patch.dict(os.environ, environment, clear=True),
        patch.dict(sys.modules, {"boto3": mock_boto3}),
    ):
        provider = _AgentCoreProvider()

    assert provider._session is None


def test_agentcore_blames_the_workspace_for_a_workspace_only_profile() -> None:
    """A profile the workspace pinned, and the server did not, still fails closed."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.side_effect = RuntimeError("ProfileNotFound: pinned")

    with (
        _bind_environment(
            {"AWS_REGION": "us-test-1", "AWS_PROFILE": "workspace-pinned"}
        ),
        patch.dict(os.environ, {"AWS_REGION": "us-test-1"}, clear=True),
        patch.dict(sys.modules, {"boto3": mock_boto3}),
        pytest.raises(ValueError, match="workspace scoped its sandbox"),
    ):
        _AgentCoreProvider()


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
        ("MODAL_TOKEN_SECRET", "MODAL_TOKEN_ID"),
    ],
)
def test_modal_rejects_a_half_set_token_pair(present: str, missing: str) -> None:
    """An incomplete pair cannot fall back to the server's Modal identity."""
    mock_modal = MagicMock()

    with (
        _bind_environment({present: "workspace-value"}),
        patch.dict(sys.modules, {"modal": mock_modal}),
        pytest.raises(ValueError, match=f"{missing} is not set"),
    ):
        _ModalProvider()

    mock_modal.App.lookup.assert_not_called()
    mock_modal.Client.from_credentials.assert_not_called()


def test_modal_delegates_when_no_token_resolves() -> None:
    """No workspace Modal credentials at all still uses default auth."""
    mock_modal = MagicMock()

    with (
        _bind_environment({}),
        patch.dict(sys.modules, {"modal": mock_modal}),
    ):
        _ModalProvider()

    assert "client" not in mock_modal.App.lookup.call_args.kwargs


def test_agentcore_honors_a_prefixed_aws_credential_override() -> None:
    """The sandbox path must read the prefix exactly as the model path does."""
    session = MagicMock()
    session.get_credentials.return_value = MagicMock()
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value = session

    with (
        _bind_environment(
            {
                "AWS_REGION": "us-test-1",
                "AWS_PROFILE": "canonical-profile",
                "DEEPAGENTS_CODE_AWS_PROFILE": "prefixed-profile",
            }
        ),
        patch.dict(sys.modules, {"boto3": mock_boto3}),
    ):
        _AgentCoreProvider()

    mock_boto3.Session.assert_called_once_with(
        profile_name="prefixed-profile",
        region_name="us-test-1",
    )
