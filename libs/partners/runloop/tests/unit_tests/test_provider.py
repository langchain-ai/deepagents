"""Unit tests for RunloopProvider blueprint bootstrapping."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from langchain_runloop.provider import (
    RunloopProvider,
    _default_resolve_env,
    _ensure_blueprint,
)


def _make_provider(*, env: dict[str, str] | None = None) -> RunloopProvider:
    env_map = env or {}

    def resolve_env(name: str) -> str | None:
        return env_map.get(name)

    with patch("langchain_runloop.provider.RunloopSDK") as mock_sdk_cls:
        mock_sdk = mock_sdk_cls.return_value
        mock_sdk.api = MagicMock()
        mock_sdk.devbox.create.return_value = MagicMock(id="dev-1")
        provider = RunloopProvider(
            api_key="test-key",
            resolve_env_var=resolve_env,
        )
    provider._sdk = mock_sdk  # noqa: SLF001
    provider._client = mock_sdk.api  # noqa: SLF001
    return provider


def test_get_or_create_uses_empty_devbox_without_blueprint_hints() -> None:
    """No snapshot/env → plain devbox.create (backward compatible)."""
    provider = _make_provider()
    sandbox = provider.get_or_create()
    provider._sdk.devbox.create.assert_called_once()  # noqa: SLF001
    assert sandbox.id == "dev-1"


def test_get_or_create_from_blueprint_name_triggers_ensure() -> None:
    """Snapshot kwarg resolves blueprint by name."""
    provider = _make_provider()
    mock_devbox = MagicMock(id="dev-bp")
    provider._sdk.devbox.create_from_blueprint_name.return_value = mock_devbox  # noqa: SLF001

    with patch(
        "langchain_runloop.provider._ensure_blueprint",
    ) as mock_ensure:
        sandbox = provider.get_or_create(snapshot="my-blueprint")

    mock_ensure.assert_called_once()
    provider._sdk.devbox.create_from_blueprint_name.assert_called_once_with(  # noqa: SLF001
        "my-blueprint"
    )
    assert sandbox.id == "dev-bp"


def test_get_or_create_from_blueprint_id_skips_ensure() -> None:
    """RUNLOOP_SANDBOX_BLUEPRINT_ID boots by ID without listing/building."""
    provider = _make_provider(env={"RUNLOOP_SANDBOX_BLUEPRINT_ID": "bp-99"})
    mock_devbox = MagicMock(id="dev-id")
    provider._sdk.devbox.create_from_blueprint_id.return_value = mock_devbox  # noqa: SLF001

    with patch("langchain_runloop.provider._ensure_blueprint") as mock_ensure:
        sandbox = provider.get_or_create()

    mock_ensure.assert_not_called()
    provider._sdk.devbox.create_from_blueprint_id.assert_called_once_with("bp-99")  # noqa: SLF001
    assert sandbox.id == "dev-id"


def test_blueprint_id_env_wins_over_kwarg() -> None:
    """`RUNLOOP_SANDBOX_BLUEPRINT_ID` overrides an explicit `snapshot` kwarg."""
    provider = _make_provider(env={"RUNLOOP_SANDBOX_BLUEPRINT_ID": "bp-id"})
    provider._sdk.devbox.create_from_blueprint_id.return_value = MagicMock(  # noqa: SLF001
        id="dev-by-id"
    )

    with patch("langchain_runloop.provider._ensure_blueprint") as mock_ensure:
        sandbox = provider.get_or_create(snapshot="ignored-name")

    mock_ensure.assert_not_called()
    provider._sdk.devbox.create_from_blueprint_id.assert_called_once_with("bp-id")  # noqa: SLF001
    provider._sdk.devbox.create_from_blueprint_name.assert_not_called()  # noqa: SLF001
    assert sandbox.id == "dev-by-id"


def test_kwarg_wins_over_blueprint_name_env() -> None:
    """Explicit `snapshot` kwarg overrides `RUNLOOP_SANDBOX_BLUEPRINT_NAME`."""
    provider = _make_provider(
        env={"RUNLOOP_SANDBOX_BLUEPRINT_NAME": "env-name"},
    )
    provider._sdk.devbox.create_from_blueprint_name.return_value = MagicMock(  # noqa: SLF001
        id="dev-kwarg"
    )

    with patch("langchain_runloop.provider._ensure_blueprint") as mock_ensure:
        sandbox = provider.get_or_create(snapshot="kwarg-name")

    mock_ensure.assert_called_once()
    assert mock_ensure.call_args.args[1] == "kwarg-name"
    provider._sdk.devbox.create_from_blueprint_name.assert_called_once_with(  # noqa: SLF001
        "kwarg-name"
    )
    assert sandbox.id == "dev-kwarg"


def test_blueprint_name_env_used_when_no_kwarg() -> None:
    """`RUNLOOP_SANDBOX_BLUEPRINT_NAME` is the lookup name without an explicit kwarg."""
    provider = _make_provider(
        env={"RUNLOOP_SANDBOX_BLUEPRINT_NAME": "env-bp"},
    )
    provider._sdk.devbox.create_from_blueprint_name.return_value = MagicMock(  # noqa: SLF001
        id="dev-env"
    )

    with patch("langchain_runloop.provider._ensure_blueprint") as mock_ensure:
        sandbox = provider.get_or_create()

    mock_ensure.assert_called_once()
    assert mock_ensure.call_args.args[1] == "env-bp"
    assert sandbox.id == "dev-env"


def test_sandbox_id_skips_blueprint_logic() -> None:
    """Attaching to an existing devbox bypasses blueprint resolution."""
    provider = _make_provider(env={"RUNLOOP_SANDBOX_BLUEPRINT_NAME": "env-bp"})
    provider._sdk.devbox.from_id.return_value = MagicMock(id="existing-dev")  # noqa: SLF001

    with patch("langchain_runloop.provider._ensure_blueprint") as mock_ensure:
        sandbox = provider.get_or_create(sandbox_id="existing-dev")

    mock_ensure.assert_not_called()
    provider._sdk.devbox.from_id.assert_called_once_with("existing-dev")  # noqa: SLF001
    provider._sdk.devbox.create.assert_not_called()  # noqa: SLF001
    assert sandbox.id == "existing-dev"


def test_attach_to_missing_devbox_translates_not_found_to_keyerror() -> None:
    """The SDK's `NotFoundError` on attach becomes a `KeyError(sandbox_id)`."""
    from runloop_api_client import NotFoundError  # noqa: I001, PLC0415  # optional SDK error type

    provider = _make_provider()
    response = httpx.Response(404, request=httpx.Request("POST", "http://x"))
    provider._sdk.devbox.from_id.side_effect = NotFoundError(  # noqa: SLF001
        "missing", response=response, body=None
    )

    with pytest.raises(KeyError, match="missing-dev"):
        provider.get_or_create(sandbox_id="missing-dev")


def test_get_or_create_rejects_unknown_kwargs() -> None:
    """Extra kwargs raise TypeError like LangSmith provider."""
    provider = _make_provider()
    with pytest.raises(TypeError, match="unsupported arguments"):
        provider.get_or_create(extra=True)


def test_blueprint_failure_wraps_in_runtime_error() -> None:
    """SDK errors during creation surface as RuntimeError with context."""
    provider = _make_provider()
    provider._sdk.devbox.create_from_blueprint_name.side_effect = RuntimeError(  # noqa: SLF001
        "boom"
    )

    with (
        patch("langchain_runloop.provider._ensure_blueprint"),
        pytest.raises(RuntimeError, match="Failed to create Runloop devbox"),
    ):
        provider.get_or_create(snapshot="bad-bp")


def test_auth_failure_wraps_with_credential_hint() -> None:
    """Authentication errors surface a credential-specific message."""
    from runloop_api_client import AuthenticationError  # noqa: I001, PLC0415  # optional SDK error type

    provider = _make_provider()
    response = httpx.Response(401, request=httpx.Request("POST", "http://x"))
    provider._sdk.devbox.create.side_effect = AuthenticationError(  # noqa: SLF001
        "bad key", response=response, body=None
    )

    with pytest.raises(RuntimeError, match="RUNLOOP_API_KEY"):
        provider.get_or_create()


def test_connection_failure_wraps_as_retryable() -> None:
    """Transient connection errors are labeled retryable, not a hard failure."""
    from runloop_api_client import APIConnectionError  # noqa: I001, PLC0415  # optional SDK error type

    provider = _make_provider()
    provider._sdk.devbox.create.side_effect = APIConnectionError(  # noqa: SLF001
        request=httpx.Request("POST", "http://x")
    )

    with pytest.raises(RuntimeError, match="transient"):
        provider.get_or_create()


def test_blueprint_dockerfile_forwarded_to_ensure() -> None:
    """A custom `blueprint_dockerfile` reaches `_ensure_blueprint`."""
    provider = _make_provider()
    provider._sdk.devbox.create_from_blueprint_name.return_value = MagicMock(  # noqa: SLF001
        id="dev-df"
    )

    with patch("langchain_runloop.provider._ensure_blueprint") as mock_ensure:
        provider.get_or_create(
            snapshot="my-bp",
            blueprint_dockerfile="FROM ubuntu:24.04\n",
        )

    assert mock_ensure.call_args.kwargs["dockerfile"] == "FROM ubuntu:24.04\n"


def test_blueprint_dockerfile_defaults_when_omitted() -> None:
    """Omitting `blueprint_dockerfile` falls back to the default Dockerfile."""
    provider = _make_provider()
    provider._sdk.devbox.create_from_blueprint_name.return_value = MagicMock(  # noqa: SLF001
        id="dev-default"
    )

    with patch("langchain_runloop.provider._ensure_blueprint") as mock_ensure:
        provider.get_or_create(snapshot="my-bp")

    assert mock_ensure.call_args.kwargs["dockerfile"] == "FROM python:3\n"


def test_delete_shuts_down_devbox() -> None:
    """`delete` shuts the devbox down by ID via the raw client."""
    provider = _make_provider()

    provider.delete(sandbox_id="dev-x")

    provider._client.devboxes.shutdown.assert_called_once_with(id="dev-x")  # noqa: SLF001


def test_ensure_blueprint_reuses_build_complete() -> None:
    """Existing build_complete blueprint is not rebuilt."""
    client = MagicMock()
    ready = MagicMock(status="build_complete")
    ready.name = "snap"
    page = MagicMock(blueprints=[ready], has_more=False)
    client.blueprints.list.return_value = page

    _ensure_blueprint(client, "snap", dockerfile="FROM python:3\n")

    client.blueprints.create_and_await_build_complete.assert_not_called()


def test_ensure_blueprint_builds_when_missing() -> None:
    """Missing blueprint triggers create_and_await_build_complete."""
    client = MagicMock()
    page = MagicMock(blueprints=[], has_more=False)
    client.blueprints.list.return_value = page

    _ensure_blueprint(client, "new-bp", dockerfile="FROM python:3\n")

    client.blueprints.create_and_await_build_complete.assert_called_once_with(
        name="new-bp",
        dockerfile="FROM python:3\n",
    )


def test_ensure_blueprint_raises_when_in_flight() -> None:
    """In-flight blueprint raises instead of starting a duplicate build."""
    client = MagicMock()
    building = MagicMock(status="building")
    building.name = "snap"
    page = MagicMock(blueprints=[building], has_more=False)
    client.blueprints.list.return_value = page

    with pytest.raises(RuntimeError, match="still building"):
        _ensure_blueprint(client, "snap", dockerfile="FROM python:3\n")


def test_ensure_blueprint_failed_status_advises_delete() -> None:
    """A blueprint whose last build failed gets a failure-specific message."""
    client = MagicMock()
    failed = MagicMock(status="failed")
    failed.name = "snap"
    page = MagicMock(blueprints=[failed], has_more=False)
    client.blueprints.list.return_value = page

    with pytest.raises(RuntimeError, match="last build failed"):
        _ensure_blueprint(client, "snap", dockerfile="FROM python:3\n")

    client.blueprints.create_and_await_build_complete.assert_not_called()


def test_ensure_blueprint_paginates_to_find_ready_match() -> None:
    """A ready match on a later page is reused without rebuilding."""
    client = MagicMock()
    # Page 1: a different blueprint, more pages to follow.
    other = MagicMock(id="bp-1", status="build_complete")
    other.name = "other"
    page1 = MagicMock(blueprints=[other], has_more=True)
    # Page 2: the ready match.
    ready = MagicMock(id="bp-2", status="build_complete")
    ready.name = "snap"
    page2 = MagicMock(blueprints=[ready], has_more=False)
    client.blueprints.list.side_effect = [page1, page2]

    _ensure_blueprint(client, "snap", dockerfile="FROM python:3\n")

    # Reaching page two proves pagination advanced; the cursor must be page
    # one's last blueprint ID.
    assert client.blueprints.list.call_args_list[1].kwargs["starting_after"] == "bp-1"
    client.blueprints.create_and_await_build_complete.assert_not_called()


def test_ensure_blueprint_paginates_then_raises_for_non_ready_match() -> None:
    """A non-ready match found only on a later page raises, never rebuilds."""
    client = MagicMock()
    other = MagicMock(id="bp-1", status="build_complete")
    other.name = "other"
    page1 = MagicMock(blueprints=[other], has_more=True)
    building = MagicMock(id="bp-2", status="building")
    building.name = "snap"
    page2 = MagicMock(blueprints=[building], has_more=False)
    client.blueprints.list.side_effect = [page1, page2]

    with pytest.raises(RuntimeError, match="still building"):
        _ensure_blueprint(client, "snap", dockerfile="FROM python:3\n")

    # The non-ready match was only discoverable by paginating to page two.
    assert client.blueprints.list.call_args_list[1].kwargs["starting_after"] == "bp-1"
    client.blueprints.create_and_await_build_complete.assert_not_called()


def test_ensure_blueprint_list_failure_raises_runtime_error() -> None:
    """SDK error during blueprint listing surfaces as RuntimeError."""
    client = MagicMock()
    client.blueprints.list.side_effect = Exception("network down")

    with pytest.raises(RuntimeError, match="Failed to list blueprints"):
        _ensure_blueprint(client, "snap", dockerfile="FROM python:3\n")

    client.blueprints.create_and_await_build_complete.assert_not_called()


def test_ensure_blueprint_create_failure_raises_runtime_error() -> None:
    """SDK error during blueprint creation surfaces as RuntimeError with context."""
    client = MagicMock()
    page = MagicMock(blueprints=[], has_more=False)
    client.blueprints.list.return_value = page
    client.blueprints.create_and_await_build_complete.side_effect = Exception(
        "quota exceeded"
    )

    with pytest.raises(RuntimeError, match="Failed to build blueprint 'new-bp'"):
        _ensure_blueprint(client, "new-bp", dockerfile="FROM python:3\n")


def test_default_resolve_env_returns_plain_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain env var is returned when no DEEPAGENTS_CODE_ override exists."""
    monkeypatch.setenv("RUNLOOP_API_KEY", "plain-key")
    monkeypatch.delenv("DEEPAGENTS_CODE_RUNLOOP_API_KEY", raising=False)

    assert _default_resolve_env("RUNLOOP_API_KEY") == "plain-key"


def test_default_resolve_env_prefix_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEEPAGENTS_CODE_ prefixed var overrides the plain one."""
    monkeypatch.setenv("RUNLOOP_API_KEY", "plain-key")
    monkeypatch.setenv("DEEPAGENTS_CODE_RUNLOOP_API_KEY", "override-key")

    assert _default_resolve_env("RUNLOOP_API_KEY") == "override-key"


def test_default_resolve_env_empty_prefix_treated_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty DEEPAGENTS_CODE_ override is treated as unset (returns None)."""
    monkeypatch.setenv("RUNLOOP_API_KEY", "plain-key")
    monkeypatch.setenv("DEEPAGENTS_CODE_RUNLOOP_API_KEY", "")

    assert _default_resolve_env("RUNLOOP_API_KEY") is None


def test_default_resolve_env_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing var with no prefix override returns None."""
    monkeypatch.delenv("RUNLOOP_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_RUNLOOP_API_KEY", raising=False)

    assert _default_resolve_env("RUNLOOP_API_KEY") is None
