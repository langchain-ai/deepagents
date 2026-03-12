"""Tests for the optional E2B CLI provider."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Never

import pytest


def _unexpected_fake_sandbox_call() -> Never:
    """Fail when a fake sandbox method is called unexpectedly."""
    msg = "unexpected fake sandbox method call"
    raise AssertionError(msg)


def _noop(*_args: object, **_kwargs: object) -> None:
    """Ignore monkeypatched provider helper arguments."""
    return


def test_provider_module_import_is_safe_without_e2b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the module should not require the optional dependency."""
    module = importlib.import_module("deepagents_cli.integrations.e2b")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(ImportError, match="e2b package is required"):
        module.E2BProvider(api_key="test-key")


def test_provider_forwards_template_on_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating a sandbox should forward the selected E2B template."""
    module = importlib.import_module("deepagents_cli.integrations.e2b")

    monkeypatch.setattr(module, "_require_e2b", lambda: None)

    class FakeSandbox:
        def __init__(self) -> None:
            self.sandbox_id = "sbx_test"
            self.commands = SimpleNamespace(
                run=lambda *_args, **_kwargs: SimpleNamespace(
                    exit_code=0,
                    stdout="ready\n",
                    stderr="",
                )
            )

    create_calls: list[dict[str, object]] = []

    class FakeSandboxClass:
        @staticmethod
        def create(*args: object, **kwargs: object) -> FakeSandbox:
            assert not args
            create_calls.append(kwargs)
            return FakeSandbox()

        @staticmethod
        def connect(*args: object, **kwargs: object) -> FakeSandbox:
            _ = args, kwargs
            _unexpected_fake_sandbox_call()

        @staticmethod
        def kill(*args: object, **kwargs: object) -> None:
            _ = args, kwargs
            _unexpected_fake_sandbox_call()

    monkeypatch.setattr(module, "_load_e2b_sandbox_class", lambda: FakeSandboxClass)
    monkeypatch.setattr(
        module,
        "_load_e2b_not_found_exception",
        lambda: RuntimeError,
    )
    monkeypatch.setattr(module.E2BProvider, "_wait_until_ready", _noop)
    monkeypatch.setattr(
        module.E2BProvider,
        "_validate_runtime_tools",
        _noop,
    )

    provider = module.E2BProvider(api_key="test-key")
    provider.get_or_create(template="custom-template")

    assert create_calls == [
        {
            "template": "custom-template",
            "timeout": module.DEFAULT_SANDBOX_LIFETIME,
            "request_timeout": module.DEFAULT_STARTUP_TIMEOUT,
            "api_key": "test-key",
        }
    ]
