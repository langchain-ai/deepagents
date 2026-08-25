"""Tests for parsed CLI configuration resolution."""

from __future__ import annotations

import argparse

import pytest

from deepagents_code.config_manifest import CliSpec, get_option
from deepagents_code.configuration.provider import CliProvider
from deepagents_code.configuration.resolver import CLI_RANK
from deepagents_code.configuration.types import Found, Invalid, ProviderHealth, Unset


@pytest.mark.parametrize(
    ("flag", "dest", "expected"),
    [
        ("--auto-classifier-model", None, "auto_classifier_model"),
        ("--auto-approve", "auto_approve", "auto_approve"),
    ],
)
def test_cli_spec_destination(flag: str, dest: str | None, expected: str) -> None:
    assert CliSpec(flag, dest).dest_name == expected


@pytest.mark.parametrize(("flag", "dest"), [("-x", None), ("", None), ("--x", "")])
def test_cli_spec_rejects_invalid_metadata(flag: str, dest: str | None) -> None:
    with pytest.raises(ValueError, match=r"CLI (flag|destination)"):
        CliSpec(flag, dest)


@pytest.mark.parametrize(
    ("key", "values", "expected"),
    [
        (
            "models.auto_classifier",
            {"auto_classifier_model": "openai:gpt-5"},
            "openai:gpt-5",
        ),
        ("shell.allow_list", {"shell_allow_list": "ls, cat"}, ["ls", "cat"]),
        ("interpreter.enable_interpreter", {"interpreter": False}, False),
        ("interpreter.ptc", {"interpreter_tools": "safe,task"}, ["safe", "task"]),
        ("threads.relative_time", {"relative": False}, False),
        ("threads.sort_order", {"sort": "created"}, "created_at"),
        ("runtime.recursion_limit", {"recursion_limit": 123}, 123),
        ("startup.mode", {"auto_approve": True, "yolo": False}, "auto"),
        ("startup.mode", {"auto_approve": False, "yolo": True}, "yolo"),
    ],
)
def test_cli_provider_resolves_manifest_options(
    key: str, values: dict[str, object], expected: object
) -> None:
    option = get_option(key)
    assert option is not None
    result = CliProvider(argparse.Namespace(**values)).get(option)

    assert result.rank == CLI_RANK
    assert result.durable is False
    assert result.result == Found(expected)


def test_cli_provider_missing_and_none_are_unset() -> None:
    option = get_option("runtime.recursion_limit")
    assert option is not None

    assert isinstance(CliProvider({}).get(option).result, Unset)
    assert isinstance(CliProvider({"recursion_limit": None}).get(option).result, Unset)


def test_cli_provider_invalid_value_is_structured() -> None:
    option = get_option("threads.sort_order")
    assert option is not None

    assert isinstance(CliProvider({"sort": "oldest"}).get(option).result, Invalid)


def test_cli_provider_status_reload_and_protocol() -> None:
    from deepagents_code.configuration.provider import ConfigProvider

    provider = CliProvider({"recursion_limit": 25})
    provider.reload()

    assert isinstance(provider, ConfigProvider)
    assert provider.name == "CLI argument"
    assert provider.status().name == "CLI argument"
    assert provider.status().health is ProviderHealth.OK


def test_persistent_action_flags_are_not_resolution_bindings() -> None:
    for key in ("models.default", "update.auto_update"):
        option = get_option(key)
        assert option is not None
        assert option.cli is None
        assert option.cli_flag is not None


def test_cli_provider_reads_any_mapping_not_just_dict() -> None:
    """A `Mapping` that is not a `dict` must resolve like one.

    Regression: discriminating on `hasattr(args, "__dict__")` sent every
    `Mapping` subclass down the `vars()` branch, snapshotting the object's
    attributes instead of its items, so every option read back `Unset`.
    """
    from collections.abc import Mapping
    from typing import Any

    class ReadOnlyArgs(Mapping):  # type: ignore[type-arg]
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> Any:  # noqa: ANN401
            return self._data[key]

        def __iter__(self) -> Any:  # noqa: ANN401
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    option = get_option("runtime.recursion_limit")
    assert option is not None

    mapping = ReadOnlyArgs({"recursion_limit": 42})
    assert CliProvider(mapping).get(option).result == Found(42)
    assert CliProvider({"recursion_limit": 42}).get(option).result == Found(42)
    assert CliProvider(argparse.Namespace(recursion_limit=42)).get(
        option
    ).result == Found(42)


def test_cli_spec_rejects_malformed_companion_flags() -> None:
    with pytest.raises(ValueError, match="long option"):
        CliSpec("--auto-approve", companion_flags=("-y",))
    with pytest.raises(ValueError, match="long option"):
        CliSpec("--auto-approve", companion_flags=("--",))


def test_startup_mode_names_both_approval_flags() -> None:
    """`--yolo` sets `startup.mode` too, so user-facing text must say so.

    Only `--auto-approve` carries the argparse destination the provider reads.
    Rendering `cli.flag` alone told a user who typed `--yolo` that a flag they
    never passed had been ignored.
    """
    option = get_option("startup.mode")
    assert option is not None
    assert option.cli is not None
    assert option.cli.companion_flags == ("--yolo",)
    assert option.cli.display_flags == "--auto-approve/--yolo"
