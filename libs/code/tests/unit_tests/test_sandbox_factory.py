"""Tests for sandbox factory optional dependency handling."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code.integrations.sandbox_config import SandboxConfig
from deepagents_code.integrations.sandbox_factory import (
    _VERCEL_SANDBOX_TIMEOUT,
    _get_provider,
    _VercelProvider,
    create_sandbox,
    get_default_working_dir,
    verify_sandbox_deps,
)
from deepagents_code.integrations.sandbox_registry import SandboxRegistry

_FACTORY = "deepagents_code.integrations.sandbox_factory"


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


def test_create_sandbox_passes_langsmith_snapshot_name() -> None:
    """LangSmith snapshot names are forwarded to the provider."""
    backend = MagicMock(id="sandbox-1")
    provider = MagicMock()
    provider.get_or_create.return_value = backend

    with (
        patch(
            "deepagents_code.integrations.sandbox_factory._get_provider",
            return_value=provider,
        ),
        create_sandbox("langsmith", snapshot_name="custom-snap") as result,
    ):
        assert result is backend

    provider.get_or_create.assert_called_once_with(
        sandbox_id=None,
        snapshot="custom-snap",
    )
    provider.delete.assert_called_once_with(sandbox_id="sandbox-1")


def test_create_sandbox_passes_runloop_snapshot_name() -> None:
    """Runloop blueprint names are forwarded to the provider."""
    backend = MagicMock(id="sandbox-1")
    provider = MagicMock()
    provider.get_or_create.return_value = backend

    with (
        patch(
            "deepagents_code.integrations.sandbox_factory._get_provider",
            return_value=provider,
        ),
        create_sandbox("runloop", snapshot_name="custom-blueprint") as result,
    ):
        assert result is backend

    provider.get_or_create.assert_called_once_with(
        sandbox_id=None,
        snapshot="custom-blueprint",
    )
    provider.delete.assert_called_once_with(sandbox_id="sandbox-1")


def test_runloop_provider_delegates_to_langchain_runloop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_RunloopProvider` forwards snapshot kwargs to `RunloopProvider`."""
    from deepagents_code.integrations.sandbox_factory import _RunloopProvider

    fake_provider = MagicMock()
    fake_provider.get_or_create.return_value = MagicMock(id="dev-1")
    fake_module = MagicMock()
    fake_module.RunloopProvider.return_value = fake_provider

    monkeypatch.setenv("RUNLOOP_API_KEY", "test-key")
    with patch(
        "deepagents_code.integrations.sandbox_factory._import_provider_module",
        return_value=fake_module,
    ):
        provider = _RunloopProvider()
        provider.get_or_create(sandbox_id=None, snapshot="my-bp")
        provider.delete(sandbox_id="dev-1")

    fake_module.RunloopProvider.assert_called_once()
    fake_provider.get_or_create.assert_called_once_with(
        sandbox_id=None,
        timeout=180,
        snapshot="my-bp",
    )
    fake_provider.delete.assert_called_once_with(sandbox_id="dev-1")


def test_runloop_provider_forwards_blueprint_dockerfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`blueprint_dockerfile` passes through `**kwargs` to `RunloopProvider`."""
    from deepagents_code.integrations.sandbox_factory import _RunloopProvider

    fake_provider = MagicMock()
    fake_provider.get_or_create.return_value = MagicMock(id="dev-1")
    fake_module = MagicMock()
    fake_module.RunloopProvider.return_value = fake_provider

    monkeypatch.setenv("RUNLOOP_API_KEY", "test-key")
    with patch(
        "deepagents_code.integrations.sandbox_factory._import_provider_module",
        return_value=fake_module,
    ):
        provider = _RunloopProvider()
        provider.get_or_create(
            sandbox_id=None,
            snapshot="my-bp",
            blueprint_dockerfile="FROM ubuntu:24.04\n",
        )

    fake_provider.get_or_create.assert_called_once_with(
        sandbox_id=None,
        timeout=180,
        snapshot="my-bp",
        blueprint_dockerfile="FROM ubuntu:24.04\n",
    )


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
    assert provider._region == "us-west-2"  # ty: ignore


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
    assert provider._region == expected  # ty: ignore


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
        "deepagents_code.integrations.sandbox_factory._import_provider_module",
        side_effect=fake_import,
    ):
        result = provider.get_or_create()

    mock_interpreter.start.assert_called_once()
    assert result is mock_backend
    assert provider._active_interpreters["session-123"] is mock_interpreter  # ty: ignore


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
            "deepagents_code.integrations.sandbox_factory._import_provider_module",
            side_effect=fake_import,
        ),
        pytest.raises(RuntimeError, match="connection failed"),
    ):
        provider.get_or_create()

    mock_interpreter.stop.assert_called_once()
    assert not provider._active_interpreters  # ty: ignore


def test_agentcore_delete_stops_tracked_interpreter() -> None:
    """delete() should call stop() on a tracked interpreter."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    mock_interpreter = MagicMock()
    provider._active_interpreters["sess-1"] = mock_interpreter  # ty: ignore

    provider.delete(sandbox_id="sess-1")

    mock_interpreter.stop.assert_called_once()
    assert "sess-1" not in provider._active_interpreters  # ty: ignore


def test_agentcore_delete_swallows_stop_exception() -> None:
    """delete() should not propagate if interpreter.stop() raises."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    mock_interpreter = MagicMock()
    mock_interpreter.stop.side_effect = RuntimeError("network error")
    provider._active_interpreters["sess-1"] = mock_interpreter  # ty: ignore

    provider.delete(sandbox_id="sess-1")  # should not raise
    assert "sess-1" not in provider._active_interpreters  # ty: ignore


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
        ("langsmith", "/root"),
        ("modal", "/workspace"),
        ("runloop", "/home/user"),
        ("vercel", "/vercel/sandbox"),
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
            ("vercel", "langchain_vercel_sandbox"),
        ],
    )
    def test_raises_import_error_when_backend_missing(
        self, provider: str, expected_module: str
    ) -> None:
        """Should raise ImportError with install instructions."""
        mock_find_spec = patch(
            "deepagents_code.integrations.sandbox_factory.importlib.util.find_spec",
            return_value=None,
        )
        with (
            mock_find_spec as find_spec,
            pytest.raises(
                ImportError,
                match=rf"Missing dependencies for '{provider}' sandbox.*"
                rf"/install {provider}.*dcode --install {provider}",
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


class TestCreateSandboxParams:
    """Tests for forwarding `params` into `provider.get_or_create()`."""

    def test_config_and_call_params_merge_with_call_precedence(self) -> None:
        """Call-site params override config params; both reach get_or_create."""
        config = SandboxConfig(
            providers={
                "acme": {
                    "class_path": "x:Y",
                    "working_dir": "/w",
                    "params": {"region": "us-east-1", "shared": "from-config"},
                }
            }
        )
        registry = _registry_with(config)
        backend = MagicMock(id="sandbox-1")
        provider = MagicMock()
        provider.get_or_create.return_value = backend

        with (
            patch(f"{_FACTORY}._get_registry", return_value=registry),
            patch(f"{_FACTORY}._get_provider", return_value=provider),
            create_sandbox(
                "acme", params={"shared": "from-call", "extra": "call-only"}
            ) as result,
        ):
            assert result is backend

        provider.get_or_create.assert_called_once_with(
            sandbox_id=None,
            region="us-east-1",
            shared="from-call",
            extra="call-only",
        )


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

    def test_fresh_sandbox_calls_create(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fresh Vercel sandboxes are created and wrapped."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_123", status="running")
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.create.return_value = sandbox
        backend = MagicMock(id="sb_123")
        vercel_backend = MagicMock()
        vercel_backend.VercelSandbox.return_value = backend

        def fake_import(module_name: str, **_: object) -> MagicMock:
            if module_name == "vercel.sandbox":
                return vercel_sdk
            return vercel_backend

        with patch(
            "deepagents_code.integrations.sandbox_factory._import_provider_module",
            side_effect=fake_import,
        ):
            result = provider.get_or_create()

        vercel_sdk.Sandbox.create.assert_called_once_with(
            runtime="python3.13",
            timeout=_VERCEL_SANDBOX_TIMEOUT,
        )
        vercel_sdk.Sandbox.get.assert_not_called()
        vercel_backend.VercelSandbox.assert_called_once_with(sandbox=sandbox)
        assert result is backend

    def test_canonical_env_vars_are_left_to_sdk(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Canonical Vercel credentials remain SDK-managed."""
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("VERCEL_TOKEN", "token-123")
        monkeypatch.setenv("VERCEL_PROJECT_ID", "prj_123")
        monkeypatch.setenv("VERCEL_TEAM_ID", "team_123")
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
            "deepagents_code.integrations.sandbox_factory._import_provider_module",
            side_effect=fake_import,
        ):
            provider.get_or_create()

        vercel_sdk.Sandbox.create.assert_called_once_with(
            runtime="python3.13",
            timeout=_VERCEL_SANDBOX_TIMEOUT,
        )

    def test_existing_sandbox_calls_get(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Existing Vercel sandboxes are attached by id."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_existing", status="running")
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.get.return_value = sandbox
        backend = MagicMock(id="sb_existing")
        vercel_backend = MagicMock()
        vercel_backend.VercelSandbox.return_value = backend

        def fake_import(module_name: str, **_: object) -> MagicMock:
            if module_name == "vercel.sandbox":
                return vercel_sdk
            return vercel_backend

        with patch(
            "deepagents_code.integrations.sandbox_factory._import_provider_module",
            side_effect=fake_import,
        ):
            result = provider.get_or_create(sandbox_id="sb_existing")

        vercel_sdk.Sandbox.get.assert_called_once_with(sandbox_id="sb_existing")
        vercel_sdk.Sandbox.create.assert_not_called()
        assert result is backend

    def test_prefixed_values_are_passed_to_create(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-empty prefixed Vercel values are forwarded to the SDK."""
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TOKEN", "token_prefixed")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_PROJECT_ID", "project_prefixed")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TEAM_ID", "team_prefixed")
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
            project_id="project_prefixed",
            team_id="team_prefixed",
        )

    def test_prefixed_values_override_canonical_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Prefixed Vercel credentials take precedence over canonical values."""
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("VERCEL_TOKEN", "token_canonical")
        monkeypatch.setenv("VERCEL_PROJECT_ID", "project_canonical")
        monkeypatch.setenv("VERCEL_TEAM_ID", "team_canonical")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TOKEN", "token_prefixed")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_PROJECT_ID", "project_prefixed")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TEAM_ID", "team_prefixed")
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
            project_id="project_prefixed",
            team_id="team_prefixed",
        )

    @pytest.mark.parametrize(
        "prefixed",
        [
            {"VERCEL_TOKEN": "token"},
            {"VERCEL_PROJECT_ID": "project"},
            {"VERCEL_TEAM_ID": "team"},
            {"VERCEL_TOKEN": "token", "VERCEL_PROJECT_ID": "project"},
            {"VERCEL_TOKEN": "token", "VERCEL_TEAM_ID": "team"},
            {"VERCEL_PROJECT_ID": "project", "VERCEL_TEAM_ID": "team"},
        ],
    )
    def test_incomplete_prefixed_credentials_fall_back_to_sdk(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        prefixed: dict[str, str],
    ) -> None:
        """Incomplete explicit credentials are discarded as a unit."""
        self._clear_vercel_env(monkeypatch)
        for name, value in prefixed.items():
            monkeypatch.setenv(f"DEEPAGENTS_CODE_{name}", value)
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
        )
        assert "Incomplete explicit Vercel credentials" in caplog.text
        assert not any(value in caplog.text for value in prefixed.values())

    def test_oidc_with_canonical_scope_remains_sdk_managed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OIDC may expose project/team IDs without an access token."""
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setenv("VERCEL_PROJECT_ID", "project")
        monkeypatch.setenv("VERCEL_TEAM_ID", "team")
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
        )

    def test_prefixed_values_are_reused_for_delete(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cleanup uses the same prefixed credentials resolved at construction."""
        self._clear_vercel_env(monkeypatch)
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TOKEN", "token_prefixed")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_PROJECT_ID", "project_prefixed")
        monkeypatch.setenv("DEEPAGENTS_CODE_VERCEL_TEAM_ID", "team_prefixed")
        provider = _get_provider("vercel")
        sandbox = MagicMock()
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.get.return_value = sandbox

        with patch(
            f"{_FACTORY}._import_provider_module",
            return_value=vercel_sdk,
        ):
            provider.delete(sandbox_id="sb_123")

        vercel_sdk.Sandbox.get.assert_called_once_with(
            sandbox_id="sb_123",
            token="token_prefixed",
            project_id="project_prefixed",
            team_id="team_prefixed",
        )
        sandbox.stop.assert_called_once_with()

    def test_delete_stops_sandbox(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Deleting a Vercel sandbox stops it."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock()
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.get.return_value = sandbox

        with patch(
            "deepagents_code.integrations.sandbox_factory._import_provider_module",
            return_value=vercel_sdk,
        ):
            provider.delete(sandbox_id="sb_123")

        vercel_sdk.Sandbox.get.assert_called_once_with(
            sandbox_id="sb_123",
        )
        sandbox.stop.assert_called_once_with()

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

    def test_readiness_timeout_is_actionable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Startup timeouts include the sandbox id, timeout, and current state."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_123", status="pending")
        sandbox.wait_for_status.side_effect = TimeoutError
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
            pytest.raises(
                RuntimeError,
                match=(
                    "Vercel sandbox sb_123 failed to start within 12 seconds; "
                    "current status is 'pending'"
                ),
            ),
        ):
            provider.get_or_create(timeout=12)

        sandbox.wait_for_status.assert_called_once_with("running", timeout=12)
        sandbox.stop.assert_called_once_with()

    def test_attached_terminal_sandbox_is_not_stopped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed attached sandbox remains owned by the caller."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_existing", status="stopped")
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.get.return_value = sandbox
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
            pytest.raises(RuntimeError, match="terminal state 'stopped'"),
        ):
            provider.get_or_create(sandbox_id="sb_existing")

        sandbox.stop.assert_not_called()

    def test_pending_sandbox_waits_for_running(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pending sandbox that transitions to running yields a backend."""
        self._clear_vercel_env(monkeypatch)
        provider = _get_provider("vercel")
        sandbox = MagicMock(sandbox_id="sb_123", status="pending")
        vercel_sdk = MagicMock()
        vercel_sdk.Sandbox.create.return_value = sandbox
        backend = MagicMock(id="sb_123")
        vercel_backend = MagicMock()
        vercel_backend.VercelSandbox.return_value = backend

        def fake_import(module_name: str, **_: object) -> MagicMock:
            if module_name == "vercel.sandbox":
                return vercel_sdk
            return vercel_backend

        with patch(
            f"{_FACTORY}._import_provider_module",
            side_effect=fake_import,
        ):
            result = provider.get_or_create(timeout=30)

        sandbox.wait_for_status.assert_called_once_with("running", timeout=30)
        assert result is backend

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

    def test_snapshot_kwarg_overrides_env_var(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit snapshot kwarg wins over `LANGSMITH_SANDBOX_SNAPSHOT_NAME`."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", "env-snap")

        existing = MagicMock(id="snap-flag", status="ready")
        existing.name = "flag-snap"
        mock_client.list_snapshots.return_value = [existing]

        provider.get_or_create(snapshot="flag-snap")

        mock_client.create_snapshot.assert_not_called()
        assert mock_client.create_sandbox.call_args.kwargs["snapshot_id"] == "snap-flag"

    def test_snapshot_name_env_var_overrides_default(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`LANGSMITH_SANDBOX_SNAPSHOT_NAME` is used as the lookup name."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", "custom-snap")

        # `MagicMock(name=...)` sets the mock's repr, not `.name` — the
        # explicit assignment below is load-bearing for the filter to match.
        existing = MagicMock(id="snap-xyz", status="ready")
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
        """With no env vars, falls back to `deepagents-code` + 16 GiB."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", raising=False)
        mock_client.list_snapshots.return_value = []
        mock_client.create_snapshot.return_value = MagicMock(id="snap-default")

        provider.get_or_create()

        kwargs = mock_client.create_snapshot.call_args.kwargs
        assert kwargs["name"] == "deepagents-code"
        assert kwargs["docker_image"] == "python:3"
        assert kwargs["fs_capacity_bytes"] == 16 * 1024**3

    def test_list_snapshots_failure_raises_runtime_error(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SDK failure during `list_snapshots` is wrapped in `RuntimeError`."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", raising=False)
        mock_client.list_snapshots.side_effect = Exception("network down")

        with pytest.raises(RuntimeError, match="Failed to list snapshots"):
            provider.get_or_create()

        mock_client.create_snapshot.assert_not_called()
        mock_client.create_sandbox.assert_not_called()

    def test_create_snapshot_failure_raises_runtime_error(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SDK failure during `create_snapshot` is wrapped with name context."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", "broken-snap")
        mock_client.list_snapshots.return_value = []
        mock_client.create_snapshot.side_effect = Exception("quota exceeded")

        with pytest.raises(
            RuntimeError,
            match=r"Failed to build snapshot 'broken-snap'",
        ):
            provider.get_or_create()

        mock_client.create_sandbox.assert_not_called()

    def test_non_ready_matching_snapshot_raises_instead_of_rebuilding(
        self,
        provider,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Matching-name snapshot in non-ready state must not silently rebuild."""
        monkeypatch.delenv("LANGSMITH_SANDBOX_SNAPSHOT_ID", raising=False)
        monkeypatch.setenv("LANGSMITH_SANDBOX_SNAPSHOT_NAME", "in-flight")

        building = MagicMock(id="snap-build-1", status="building")
        building.name = "in-flight"
        mock_client.list_snapshots.return_value = [building]

        with pytest.raises(
            RuntimeError,
            match=r"Snapshot 'in-flight' exists but is in state 'building'",
        ):
            provider.get_or_create()

        mock_client.create_snapshot.assert_not_called()
        mock_client.create_sandbox.assert_not_called()
