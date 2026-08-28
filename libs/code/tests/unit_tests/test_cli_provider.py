"""Tests for parsed CLI configuration resolution."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from deepagents_code.config_manifest import CliSpec, get_option
from deepagents_code.configuration.provider import CliProvider
from deepagents_code.configuration.resolver import CLI_RANK
from deepagents_code.configuration.types import Found, Invalid, ProviderHealth, Unset


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


def _all_parser_destinations() -> set[str]:
    """Collect every argparse destination, root parser and subcommands alike.

    `parse_args` builds its parser inline, so the instance is captured on the
    way through rather than rebuilt here -- a copy would drift from the real
    one, which is exactly the failure this guards.

    Returns:
        Every `dest` the CLI can populate.
    """
    import sys
    from unittest.mock import patch

    from deepagents_code.main import parse_args

    captured: list[argparse.ArgumentParser] = []
    real_parse_args = argparse.ArgumentParser.parse_args

    def _capture(
        self: argparse.ArgumentParser, *args: Any, **kwargs: Any
    ) -> argparse.Namespace:
        captured.append(self)
        return real_parse_args(self, *args, **kwargs)

    with (
        patch.object(sys, "argv", ["dcode", "-n", "task"]),
        patch.object(argparse.ArgumentParser, "parse_args", _capture),
    ):
        parse_args()

    assert captured, "parse_args did not build a parser"

    dests: set[str] = set()

    def _walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:  # no public accessor
            if action.dest != argparse.SUPPRESS:
                dests.add(action.dest)
            choices = getattr(action, "choices", None) or {}
            if isinstance(action, argparse._SubParsersAction):
                for sub in choices.values():
                    _walk(sub)

    _walk(captured[0])
    return dests
