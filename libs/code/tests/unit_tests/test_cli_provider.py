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


def test_every_cli_binding_names_a_real_argparse_destination() -> None:
    """Each `CliSpec` must resolve to a destination argparse actually sets.

    `CliProvider.get` maps a missing destination to `Unset`, which is
    indistinguishable from an absent flag. A typo, or an argparse `dest=`
    rename, therefore stops a bound flag working with no error at import, at
    construction, or at read -- the option silently reads `Unset` forever.

    Nothing else ties the two files together: the manifest declares the
    bindings and argparse defines the destinations, across a boundary no type
    check crosses.
    """
    from deepagents_code.config_manifest import get_config_options

    dests = _all_parser_destinations()
    missing = {
        option.key: spec.dest_name
        for option in get_config_options()
        if (spec := option.cli) is not None
        if spec.dest_name not in dests
    }
    assert not missing, f"CLI bindings name unknown argparse destinations: {missing}"


def test_every_companion_flag_names_a_real_argparse_destination() -> None:
    """Companion flags must resolve too: the provider reads `--yolo`'s dest."""
    from deepagents_code.config_manifest import get_config_options

    dests = _all_parser_destinations()
    missing: dict[str, str] = {}
    for option in get_config_options():
        spec = option.cli
        if spec is None:
            continue
        for flag in spec.companion_flags:
            dest_name = CliSpec(flag).dest_name
            if dest_name not in dests:
                missing[option.key] = dest_name
    assert not missing, f"companion flags name unknown destinations: {missing}"


def test_bound_flags_have_no_truthy_argparse_default() -> None:
    """A bound flag must default to `None`, or the CLI tier always declares.

    `CliProvider.get` treats any non-`None` value as `Found`, so an
    `action="store_true"` on a bound flag would make the CLI tier report
    `False` on every invocation. That permanently masks the user's
    `config.toml` for the option and fires a spurious "managed config takes
    precedence" warning. Every bound flag is safe today; nothing keeps it so.
    """
    import sys
    from unittest.mock import patch

    from deepagents_code.config_manifest import get_config_options
    from deepagents_code.main import parse_args

    with patch.object(sys, "argv", ["dcode", "-n", "task"]):
        args = parse_args()

    offenders = {
        option.key: value
        for option in get_config_options()
        if (spec := option.cli) is not None
        if (value := getattr(args, spec.dest_name, None)) is not None
    }
    assert not offenders, (
        f"bound flags declare a value with no flag passed: {offenders}"
    )
