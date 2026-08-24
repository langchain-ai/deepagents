"""Drift, resolution, and behavior tests for the configuration manifest.

These guard the contract that the manifest is the single source of truth for
the scalar config surface, that its resolver matches what the runtime reads,
and that secret-flagged options are never rendered by value.
"""

from __future__ import annotations

import argparse
import logging
import os
import tomllib
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from deepagents_code import _env_vars
from deepagents_code.client.commands.config import (
    _display_value,
    _missing_extra_hint,
    _resolve,
    _run_get,
    _source_label,
    run_config_command,
)
from deepagents_code.config_manifest import (
    CURSOR_STYLE_DEFAULT,
    NON_OPTION_ENV_VARS,
    ConfigOption,
    OptionKind,
    _emit_ranked_diagnostics,
    _ranked_source,
    get_config_options,
    get_option,
    is_provider_package_installed,
    load_bool_display_preference,
    option_keys,
    options_with_key_prefix,
    provider_install_extra,
    provider_package_name,
)
from deepagents_code.model_config import DEFAULT_STARTUP_MODE, PROVIDER_API_KEY_ENV
from unit_tests.conftest import redirect_managed_config, resolve_option_for_test

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# Most unit tests set `DEEPAGENTS_CODE_NO_UPDATE_CHECK=1` to avoid accidental
# PyPI/DNS work. This module checks whether update settings came from the env,
# config file, or built-in defaults; adding the env var here would hide the
# config/default cases these tests are trying to verify.
pytestmark = pytest.mark.self_managed_update_check


def _resolve_manifest_option(
    option: ConfigOption,
    *,
    toml_data: dict[str, Any],
    managed_toml_data: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Resolve `option` through production code as `(value, source)`.

    Thin alias for the shared `resolve_option_for_test`; see it for why these
    resolutions must not be rebuilt in the test suite.
    """
    return resolve_option_for_test(
        option, toml_data=toml_data, managed_toml_data=managed_toml_data
    )


def _declared_deepagents_env_vars() -> set[str]:
    """Every `DEEPAGENTS_CODE_*` constant declared in `_env_vars`."""
    return {
        value
        for name, value in vars(_env_vars).items()
        if not name.startswith("_")
        and isinstance(value, str)
        and value.startswith("DEEPAGENTS_CODE_")
    }


def test_show_diff_line_numbers_defaults_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diff hunk line numbers remain enabled when app config is unset."""
    from deepagents_code import config_manifest
    from deepagents_code.app import _load_show_diff_line_numbers

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    assert _load_show_diff_line_numbers() is True


def test_show_diff_line_numbers_reads_app_config(tmp_path: Path) -> None:
    """`[ui].show_diff_line_numbers` can disable the diff gutter."""
    from deepagents_code.app import _load_show_diff_line_numbers

    # Written to the real file rather than stubbing a loader: display
    # preferences resolve through the shared process generation, so the user
    # tier is the `DEFAULT_CONFIG_PATH` that `_isolate_state_dir` redirects
    # under `tmp_path`.
    (tmp_path / "config.toml").write_text(
        "[ui]\nshow_diff_line_numbers = false\n", encoding="utf-8"
    )

    assert _load_show_diff_line_numbers() is False


# --- Drift / coverage -------------------------------------------------------


def test_manifest_covers_every_deepagents_env_var() -> None:
    """Every `DEEPAGENTS_CODE_*` env var must have a manifest entry."""
    manifest_env_vars = {opt.env_var for opt in get_config_options() if opt.env_var}
    declared = _declared_deepagents_env_vars() - NON_OPTION_ENV_VARS
    missing = declared - manifest_env_vars
    assert not missing, (
        f"`DEEPAGENTS_CODE_*` env vars without a manifest entry: {sorted(missing)}. "
        "Add a ConfigOption in config_manifest.py or list it in NON_OPTION_ENV_VARS."
    )


def test_manifest_covers_every_provider_credential() -> None:
    """Every provider in `PROVIDER_API_KEY_ENV` must have a credential option."""
    manifest_env_vars = {opt.env_var for opt in get_config_options() if opt.env_var}
    missing = set(PROVIDER_API_KEY_ENV.values()) - manifest_env_vars
    assert not missing, (
        f"Provider credential env vars without a manifest entry: {sorted(missing)}."
    )


_UI_READER_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Terminal-default inspection and the theme provider retain bespoke
        # terminal-program mapping semantics.
        "app.py:_load_terminal_default",
        "providers.py:ranked_theme_toml_value",
        # Reads `[ui]` only to repair and rewrite it; not a value reader.
        "app.py:_replace_malformed_ui",
    }
)
"""Functions permitted to read the `[ui]` table without using the manifest."""


def test_no_hand_rolled_ui_config_readers() -> None:
    """`[ui]` scalars must be read through the manifest, not re-parsed by hand.

    A bespoke loader re-implements env parsing, the `[ui]` lookup, the type
    check, and the fallback — which is how the app came to read values that
    `dcode config` did not report, and vice versa.
    """
    import ast
    from pathlib import Path

    from deepagents_code import config_manifest

    package_root = Path(config_manifest.__file__).parent
    readers: set[str] = set()
    for source in sorted(package_root.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        if '.get("ui"' not in text:
            continue
        hits = [
            number
            for number, line in enumerate(text.splitlines(), start=1)
            if '.get("ui"' in line
        ]
        functions = [
            node
            for node in ast.walk(ast.parse(text))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for hit in hits:
            # Smallest enclosing span, so a nested helper is not attributed to
            # the function it happens to live in.
            enclosing = min(
                (
                    node
                    for node in functions
                    if node.lineno <= hit <= (node.end_lineno or node.lineno)
                ),
                key=lambda node: (node.end_lineno or node.lineno) - node.lineno,
                default=None,
            )
            name = enclosing.name if enclosing else "<module>"
            readers.add(f"{source.name}:{name}")

    assert readers == _UI_READER_ALLOWLIST, (
        f"Unexpected `[ui]` readers: {sorted(readers - _UI_READER_ALLOWLIST)}. "
        "Register the option in config_manifest.py and resolve it through the "
        "shared resolver instead of parsing config.toml by hand."
    )


# Readers that resolve an option but deliberately emit no ranked diagnostics.
# Each reports what it rejected through its own channel: the two shared
# primitives hand the `ResolvedValue` back for the caller to emit against,
# `dcode config` prints provenance per option, the managed validator inspects a
# candidate generation not yet in force, and the sandbox, theme, and update
# readers log their own rejections in terms of the setting they own.
#
# Adding to this set should be a deliberate act with a channel named here. It
# is not a place to park a reader that reports nothing: file health is not
# value health, and a rejected value in a file that parses is invisible to
# `dcode doctor`.
_SILENT_RESOLVER_READERS = frozenset(
    {
        # The shared resolution primitives, not readers: each returns the
        # `ResolvedValue` and every caller emits against it.
        "config_manifest.py:_resolve_option",
        "config_manifest.py:_resolve_option_without_managed",
        "client/commands/config.py:_option_provenance",
        "configuration/service.py:resolve_managed_option",
        "integrations/sandbox_config.py:load",
        "theme.py:_load_user_themes",
        "update_check.py:_resolve_update_setting",
    }
)


def test_resolver_reads_emit_ranked_diagnostics() -> None:
    """A function that resolves an option must also report what was rejected.

    `resolve_scalar` emitted diagnostics itself, so no caller could forget
    them. Retiring it turned that guarantee into a two-line convention repeated
    at every reader, and the convention was dropped once already
    (`_save_ui_bool_result`, fixed in f6c6c8196): a malformed managed entry
    stopped being reported anywhere, removing the only signal an administrator
    has that their policy is inert.

    The failure is invisible at runtime -- the value falls back to the default
    and nothing is logged -- so it needs a structural guard rather than a
    behavioral one. Readers that report health through another channel are
    listed in `_SILENT_RESOLVER_READERS`; adding to that set should be a
    deliberate act with a reason.

    Three blind spots it cannot close, because it matches a literal `.get`
    call in the same function body: a reader that goes through
    `resolve_options` or `resolve_all` instead (`config.py` does, and passes
    only because it happens to emit per option); one that receives a
    `ConfigResolver` as a parameter rather than building one; and one that
    resolves several options and emits for only some of them --
    `sandbox_config.load` was exactly that, and being on the allowlist meant
    the guard could not have flagged it either way.
    """
    import ast
    from pathlib import Path

    from deepagents_code import config_manifest

    package_root = Path(config_manifest.__file__).parent
    silent: set[str] = set()
    for source in sorted(package_root.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        if "get_config_resolver" not in text and "resolver_from_snapshots" not in text:
            continue
        for function in ast.walk(ast.parse(text)):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            builds_resolver = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"get_config_resolver", "resolver_from_snapshots"}
                for node in ast.walk(function)
            )
            # `.get(option)` on the resolver, whether chained onto the
            # constructor or reached through a local binding.
            reads_option = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                for node in ast.walk(function)
            )
            if not (builds_resolver and reads_option):
                continue
            # A *call*, not the name: these readers import the helper inside
            # the function, so a substring check passes even after the call
            # itself is deleted.
            emits = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_emit_ranked_diagnostics"
                for node in ast.walk(function)
            )
            if emits:
                continue
            relative = source.relative_to(package_root).as_posix()
            silent.add(f"{relative}:{function.name}")

    assert silent == _SILENT_RESOLVER_READERS, (
        "Resolver readers that emit no diagnostics: "
        f"{sorted(silent - _SILENT_RESOLVER_READERS)}. Call "
        "`_emit_ranked_diagnostics(option, resolved)` after resolving, or add "
        "the reader to `_SILENT_RESOLVER_READERS` with the channel it reports "
        "health through instead."
    )


# Six `app.py` display toggles plus `display.show_usage_stats` in
# `_session_stats.py`. The `app.py` wrapper forwards variables, not literals,
# so it is deliberately not counted. Exact, not a floor — see the test.
_EXPECTED_LITERAL_CALL_SITES = 7


def test_bool_display_preference_fallbacks_match_the_manifest() -> None:
    """Every `load_bool_display_preference` call site must agree with the manifest.

    The `fallback` argument duplicates the option's declared default, and a
    mistyped key silently resolves to that fallback — with both in agreement
    that produces no symptom at all until someone actually sets the option.

    This walks the call sites instead of listing them, because a hand-kept list
    is exactly how the drift gets in: the key can be added to the list while a
    second call site with a different fallback goes unnoticed, or a new caller
    can skip the list entirely.

    The walk only sees literal arguments, so a refactor that passed keys or
    fallbacks as variables would quietly shrink coverage while still passing on
    whatever literal sites remained. `_EXPECTED_LITERAL_CALL_SITES` is an exact
    count rather than a floor, so both a shrink and an unreviewed new literal
    caller are loud — bump it when you add one. The blind spot it cannot close
    is a *new* caller that passes a variable key: the count stays put and
    nothing checks that caller's pair.
    """
    import ast
    from pathlib import Path

    from deepagents_code import config_manifest

    package_root = Path(config_manifest.__file__).parent
    call_sites: list[tuple[str, str, object]] = []
    for source in sorted(package_root.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        if "load_bool_display_preference(" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name not in {
                "load_bool_display_preference",
                "_load_bool_display_preference",
            }:
                continue
            keys = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ] + [
                kw.value.value
                for kw in node.keywords
                if kw.arg == "key"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ]
            fallbacks = [
                kw.value.value
                for kw in node.keywords
                if kw.arg == "fallback" and isinstance(kw.value, ast.Constant)
            ]
            # The `app.py` wrapper forwards `key`/`fallback` variables rather
            # than literals; only literal pairs are checkable here.
            if keys and fallbacks:
                call_sites.append((source.name, keys[0], fallbacks[0]))

    assert len(call_sites) == _EXPECTED_LITERAL_CALL_SITES, (
        f"found {len(call_sites)} literal call sites, expected "
        f"{_EXPECTED_LITERAL_CALL_SITES}; a caller that stopped passing "
        "literals is no longer checked here, and a new one has not been "
        "reviewed — change the count deliberately"
    )
    for filename, key, fallback in call_sites:
        option = get_option(key)
        assert option is not None, f"{filename}: {key} is not in the manifest"
        assert option.kind is OptionKind.BOOL, (
            f"{filename}: {key} is {option.kind.value}, not a bool"
        )
        assert option.default is fallback, (
            f"{filename}: {key} default {option.default!r} disagrees with the "
            f"call-site fallback {fallback!r}; they must match or the fallback "
            "path changes behavior"
        )


def test_bool_display_preference_unknown_key_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmapped key returns the fallback and says so in the log.

    Nothing else exercises this branch: every real call site names a key that
    is in the manifest, so the guard only ever fires after a rename. The
    fallback is returned as given, not coerced, in both directions.
    """
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert (
            load_bool_display_preference("display.no_such_key", fallback=False) is False
        )
        assert (
            load_bool_display_preference("display.no_such_key", fallback=True) is True
        )

    assert "display.no_such_key" in caplog.text


def test_bool_display_preference_non_bool_key_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A key naming a non-bool option returns the fallback rather than coercing.

    Without the kind check, `display.charset` (a `STR` option defaulting to
    `"auto"`) would resolve to `bool("auto")` — permanently `True`, with no
    warning at all.
    """
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert load_bool_display_preference("display.charset", fallback=False) is False

    assert "display.charset" in caplog.text
    assert "not a bool" in caplog.text


def test_toml_only_bool_display_options_declare_no_env_var() -> None:
    """`display.show_diff_line_numbers` is `config.toml`-only, deliberately.

    It is TUI-only and toggled in-app with `/line-numbers`, so `config.toml` is
    the natural place to preset it — unlike `display.show_usage_stats`, which
    also gates headless output and therefore does declare an env var.

    The promise lives in `app._load_show_diff_line_numbers` ("There is no env
    var for this option."), which is what a failure here has to go fix;
    `load_bool_display_preference` only says that not every option declares
    one. Adding an `env_var` is a legitimate product change, but it should be a
    decision rather than a drive-by, so it has to come through this test.
    """
    option = get_option("display.show_diff_line_numbers")
    assert option is not None
    assert option.env_var is None, (
        "display.show_diff_line_numbers gained an env var; update the promise in "
        "`app._load_show_diff_line_numbers`"
    )
    assert not option.fallback_env_vars


@pytest.mark.parametrize(
    ("key", "toml_keys"),
    [
        ("display.show_message_timestamps", ("ui", "show_message_timestamps")),
        ("display.show_usage_stats", ("ui", "show_usage_stats")),
        ("display.themes", ("themes",)),
        ("display.terminal_themes", ("ui", "terminal_themes")),
        ("models.providers", ("models", "providers")),
        ("retries.max_retries", ("retries", "max_retries")),
        ("agents.default", ("agents", "default")),
        ("agents.recent", ("agents", "recent")),
        ("agents.async_subagents", ("async_subagents",)),
        ("startup.recent", ("startup", "recent")),
        ("sandboxes.default", ("sandboxes", "default")),
        ("sandboxes.providers", ("sandboxes", "providers")),
    ],
)
def test_config_toml_sections_are_discoverable(
    key: str, toml_keys: tuple[str, ...]
) -> None:
    """Every `config.toml` section the app reads must be listed by `dcode config`."""
    option = get_option(key)
    assert option is not None, f"{key} is missing from the manifest"
    assert option.toml_keys == toml_keys


def test_negative_retry_count_is_not_reported_as_effective(caplog) -> None:
    """A retry value the runtime rejects must not appear active in `config`."""
    import logging

    option = get_option("retries.max_retries")
    assert option is not None
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option, toml_data={"retries": {"max_retries": -1}}
        ) == (
            None,
            "default",
        )
    assert any(
        "[retries].max_retries" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize("key", ["agents.default", "agents.recent"])
def test_blank_saved_agent_is_not_reported_as_effective(key: str, caplog) -> None:
    """Blank agent preferences fall through just as the launch loader does."""
    import logging

    option = get_option(key)
    assert option is not None
    field = key.removeprefix("agents.")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option, toml_data={"agents": {field: "   "}}
        ) == (
            None,
            "default",
        )
    assert any(f"[agents].{field}" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("key", ["agents.default", "agents.recent"])
def test_saved_agent_is_trimmed_like_the_launch_loader(key: str) -> None:
    """The manifest reports the normalized saved-agent value used at launch."""
    option = get_option(key)
    assert option is not None
    field = key.removeprefix("agents.")
    assert _resolve_manifest_option(
        option, toml_data={"agents": {field: "  coder  "}}
    ) == (
        "coder",
        "config.toml",
    )


def test_show_message_timestamps_env_overrides_config(monkeypatch) -> None:
    """The env var must outrank a persisted `/timestamps` toggle."""
    option = get_option("display.show_message_timestamps")
    assert option is not None
    monkeypatch.setenv(_env_vars.SHOW_MESSAGE_TIMESTAMPS, "1")
    value, source = _resolve_manifest_option(
        option, toml_data={"ui": {"show_message_timestamps": False}}
    )
    assert value is True
    assert source == f"env ({_env_vars.SHOW_MESSAGE_TIMESTAMPS})"


def test_option_keys_unique() -> None:
    """Manifest keys must be unique so `config get` lookups are unambiguous."""
    keys = option_keys()
    assert len(keys) == len(set(keys))


def test_cold_cache_warning_threshold_default() -> None:
    """Cold-cache warning uses the material incremental-cost default."""
    option = get_option("warnings.cold_cache_min_delta_usd")
    assert option is not None
    assert _resolve_manifest_option(option, toml_data={}) == (0.50, "default")


def test_memory_auto_save_defaults_enabled(monkeypatch) -> None:
    """`memory.auto_save` resolves to enabled when nothing overrides it."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    assert _resolve_manifest_option(option, toml_data={}) == (True, "default")


def test_memory_auto_save_env_disables(monkeypatch) -> None:
    """A falsy `DEEPAGENTS_CODE_MEMORY_AUTO_SAVE` disables auto-save."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.setenv(_env_vars.MEMORY_AUTO_SAVE, "0")
    value, _ = _resolve_manifest_option(option, toml_data={})
    assert value is False


def test_memory_auto_save_empty_env_disables(monkeypatch) -> None:
    """An explicitly empty env override disables automatic memory saving."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.setenv(_env_vars.MEMORY_AUTO_SAVE, "")
    assert _resolve_manifest_option(option, toml_data={}) == (
        False,
        f"env ({_env_vars.MEMORY_AUTO_SAVE})",
    )


def test_memory_auto_save_toml_disables(monkeypatch) -> None:
    """`[memory].auto_save = false` in config.toml disables auto-save."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    value, _ = _resolve_manifest_option(
        option, toml_data={"memory": {"auto_save": False}}
    )
    assert value is False


def test_is_memory_auto_save_enabled_reads_env(monkeypatch) -> None:
    """The `config.is_memory_auto_save_enabled` helper honors the env override."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_memory_auto_save_enabled

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    assert is_memory_auto_save_enabled() is True

    monkeypatch.setenv(_env_vars.MEMORY_AUTO_SAVE, "false")
    assert is_memory_auto_save_enabled() is False


def test_is_memory_auto_save_enabled_reads_toml(monkeypatch, tmp_path: Path) -> None:
    """The helper honors `[memory].auto_save` from `config.toml` when env is unset."""
    from deepagents_code.config import is_memory_auto_save_enabled

    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    (tmp_path / "config.toml").write_text(
        "[memory]\nauto_save = false\n", encoding="utf-8"
    )
    assert is_memory_auto_save_enabled() is False


def test_is_openai_prompt_cache_key_enabled_reads_env(monkeypatch) -> None:
    """`is_openai_prompt_cache_key_enabled` honors the env override."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_openai_prompt_cache_key_enabled

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    monkeypatch.delenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, raising=False)
    assert is_openai_prompt_cache_key_enabled() is True

    monkeypatch.setenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, "false")
    assert is_openai_prompt_cache_key_enabled() is False


def test_is_openai_prompt_cache_key_enabled_reads_toml(
    monkeypatch, tmp_path: Path
) -> None:
    """The helper honors `[models].openai_prompt_cache_key` when env is unset."""
    from deepagents_code.config import is_openai_prompt_cache_key_enabled

    monkeypatch.delenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, raising=False)
    (tmp_path / "config.toml").write_text(
        "[models]\nopenai_prompt_cache_key = false\n", encoding="utf-8"
    )
    assert is_openai_prompt_cache_key_enabled() is False


def test_is_openai_prompt_cache_key_enabled_empty_env_opts_out(monkeypatch) -> None:
    """An explicitly empty env value opts out (via `empty_env_is_false`)."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_openai_prompt_cache_key_enabled

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    monkeypatch.setenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, "")
    assert is_openai_prompt_cache_key_enabled() is False


def test_is_openai_prompt_cache_key_enabled_unrecognized_env_falls_through(
    monkeypatch, tmp_path: Path
) -> None:
    """An unrecognized env token is ignored, so config.toml decides."""
    from deepagents_code.config import is_openai_prompt_cache_key_enabled

    monkeypatch.setenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, "banana")
    (tmp_path / "config.toml").write_text(
        "[models]\nopenai_prompt_cache_key = false\n", encoding="utf-8"
    )
    assert is_openai_prompt_cache_key_enabled() is False


def test_resolve_auto_classifier_model_defaults_to_inheriting(monkeypatch) -> None:
    """No preference leaves the Auto classifier on the main agent model."""
    from deepagents_code.config import resolve_auto_classifier_model

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    assert resolve_auto_classifier_model() is None


def test_resolve_auto_classifier_model_reads_toml(monkeypatch, tmp_path: Path) -> None:
    """`[models].auto_classifier` selects the classifier when env is unset."""
    from deepagents_code.config import resolve_auto_classifier_model

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "openai:gpt-5.5-mini"\n',
        encoding="utf-8",
    )
    assert resolve_auto_classifier_model() == "openai:gpt-5.5-mini"


def test_resolve_auto_classifier_model_keeps_shared_user_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    """A file edit cannot switch the classifier before config reload."""
    from deepagents_code.config import resolve_auto_classifier_model

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[models]\nauto_classifier = "openai:original"\n', encoding="utf-8"
    )
    assert resolve_auto_classifier_model() == "openai:original"

    config_path.write_text(
        '[models]\nauto_classifier = "openai:edited"\n', encoding="utf-8"
    )

    assert resolve_auto_classifier_model() == "openai:original"


def test_resolve_auto_classifier_model_env_overrides_toml(
    monkeypatch, tmp_path: Path
) -> None:
    """The env var wins over config.toml; a blank value means inherit."""
    from deepagents_code.config import resolve_auto_classifier_model

    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "openai:gpt-5.5-mini"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "anthropic:claude-haiku-4-5")
    assert resolve_auto_classifier_model() == "anthropic:claude-haiku-4-5"

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "   ")
    assert resolve_auto_classifier_model() is None


def test_resolve_auto_classifier_model_warns_on_blank_value(
    monkeypatch, caplog, tmp_path: Path
) -> None:
    """A blank configured classifier is reported, not dropped in silence.

    Reverting to the main agent model is a security control quietly changing
    behavior, so it gets the same audible treatment a malformed value gets.
    """
    from deepagents_code.config import resolve_auto_classifier_model

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "   "\n', encoding="utf-8"
    )

    with caplog.at_level("WARNING", logger="deepagents_code.config"):
        assert resolve_auto_classifier_model() is None

    assert any(
        "blank" in record.getMessage() and "auto_classifier" in record.getMessage()
        for record in caplog.records
    )


def test_resolve_auto_classifier_model_reports_blank_problem(
    monkeypatch, tmp_path: Path
) -> None:
    """A blank value is returned as a problem, not just logged.

    A `logger.warning` lands in the debug log, which is not a surface the user
    reads; the caller needs the description to show it at startup.
    """
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "   "\n', encoding="utf-8"
    )

    spec, problem = resolve_auto_classifier_model_with_problem()

    assert spec is None
    assert problem is not None
    assert "blank" in problem
    assert "main agent model" in problem


def test_resolve_auto_classifier_model_reports_malformed_problem(
    monkeypatch, tmp_path: Path
) -> None:
    """A wrong-typed value must not silently revert to self-review.

    TOML coercion drops a non-string to the option default, making a
    malformed entry indistinguishable from an absent one — so the agent resumed
    grading its own actions with nothing logged at all.
    """
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    (tmp_path / "config.toml").write_text(
        "[models]\nauto_classifier = 3\n", encoding="utf-8"
    )

    spec, problem = resolve_auto_classifier_model_with_problem()

    assert spec is None
    assert problem is not None
    # The value, not just the fact of rejection: this string is shown to the
    # user, and "your setting is malformed" without saying which value was
    # rejected gives them nothing to act on.
    assert "auto_classifier=3" in problem
    assert "config.toml" in problem
    assert "provider:model" in problem
    assert "main agent model" in problem


def test_resolve_auto_classifier_model_reports_no_problem_when_absent(
    monkeypatch,
) -> None:
    """An unconfigured classifier is the default, not a problem to report."""
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)

    assert resolve_auto_classifier_model_with_problem() == (None, None)


def test_goal_auto_accept_criteria_defaults_to_review(monkeypatch) -> None:
    """Auto mode reviews generated goal criteria when no preference is set."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.delenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, raising=False)

    assert _resolve_manifest_option(option, toml_data={}) == (False, "default")


def test_goal_auto_accept_criteria_reads_toml(monkeypatch) -> None:
    """The goals table can opt Auto into applying criteria automatically."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.delenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, raising=False)

    assert _resolve_manifest_option(
        option,
        toml_data={"goals": {"auto_accept_criteria": True}},
    ) == (True, "config.toml")


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("0", False)])
def test_goal_auto_accept_criteria_env_overrides_toml(
    monkeypatch,
    raw: str,
    expected: bool,
) -> None:
    """A recognized env value takes priority over config.toml."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.setenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, raw)

    assert _resolve_manifest_option(
        option,
        toml_data={"goals": {"auto_accept_criteria": not expected}},
    ) == (expected, f"env ({_env_vars.GOAL_AUTO_ACCEPT_CRITERIA})")


@pytest.mark.parametrize(
    ("toml_data", "expected"),
    [
        ({"goals": {"auto_accept_criteria": True}}, (True, "config.toml")),
        ({}, (False, "default")),
    ],
)
def test_invalid_goal_auto_accept_env_falls_through(
    monkeypatch,
    toml_data: dict[str, object],
    expected: tuple[bool, str],
) -> None:
    """An invalid env value should not mask TOML or the safe default."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.setenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, "maybe")

    assert _resolve_manifest_option(option, toml_data=toml_data) == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(None, False), (_env_vars.DEBUG, True), (_env_vars.EXPERIMENTAL, True)],
)
def test_resume_term_program_resolves_mode_default(
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
    expected: bool,
) -> None:
    """The resume prefix defaults on only in experimental or debug mode."""
    option = get_option("features.resume_term_program")
    assert option is not None
    monkeypatch.delenv(_env_vars.RESUME_TERM_PROGRAM, raising=False)
    monkeypatch.delenv(_env_vars.DEBUG, raising=False)
    monkeypatch.delenv(_env_vars.EXPERIMENTAL, raising=False)
    if mode is not None:
        monkeypatch.setenv(mode, "1")

    assert _resolve_manifest_option(option, toml_data={}) == (expected, "default")


@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("0", False), ("", False)])
def test_resume_term_program_env_overrides_mode_default(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    """An explicit feature env value wins over mode and TOML values."""
    option = get_option("features.resume_term_program")
    assert option is not None
    monkeypatch.setenv(_env_vars.DEBUG, "1")
    monkeypatch.setenv(_env_vars.EXPERIMENTAL, "1")
    monkeypatch.setenv(_env_vars.RESUME_TERM_PROGRAM, raw)

    assert _resolve_manifest_option(
        option,
        toml_data={"features": {"resume_term_program": not expected}},
    ) == (expected, f"env ({_env_vars.RESUME_TERM_PROGRAM})")


@pytest.mark.parametrize(
    ("configured", "mode", "expected"),
    [
        (True, None, True),
        (False, _env_vars.DEBUG, False),
        (False, _env_vars.EXPERIMENTAL, False),
    ],
)
def test_resume_term_program_toml_overrides_mode_default(
    monkeypatch: pytest.MonkeyPatch,
    configured: bool,
    mode: str | None,
    expected: bool,
) -> None:
    """An explicit config.toml value wins over the mode-dependent default."""
    option = get_option("features.resume_term_program")
    assert option is not None
    monkeypatch.delenv(_env_vars.RESUME_TERM_PROGRAM, raising=False)
    monkeypatch.delenv(_env_vars.DEBUG, raising=False)
    monkeypatch.delenv(_env_vars.EXPERIMENTAL, raising=False)
    if mode is not None:
        monkeypatch.setenv(mode, "1")

    assert _resolve_manifest_option(
        option,
        toml_data={"features": {"resume_term_program": configured}},
    ) == (expected, "config.toml")


def test_resume_term_program_unrecognized_env_falls_through_to_mode_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd feature flag must not silently defeat debug mode."""
    option = get_option("features.resume_term_program")
    assert option is not None
    monkeypatch.delenv(_env_vars.EXPERIMENTAL, raising=False)
    monkeypatch.setenv(_env_vars.DEBUG, "1")
    monkeypatch.setenv(_env_vars.RESUME_TERM_PROGRAM, "maybe")

    assert _resolve_manifest_option(option, toml_data={}) == (True, "default")


def test_bool_mode_default_rejects_declared_default() -> None:
    """A declared default is dead for this kind, so it is rejected up front.

    Resolution derives the default from debug/experimental mode and
    returns before reading `default` -- but `dcode config` still renders the
    declared value, advertising a default that contradicts the real one.
    """
    option = get_option("features.resume_term_program")
    assert option is not None
    assert option.default is None

    with pytest.raises(TypeError, match="must not declare a default"):
        ConfigOption(
            key="features.example",
            group="Tools",
            summary="Example.",
            kind=OptionKind.BOOL_MODE_DEFAULT,
            default=False,
        )


def test_debug_log_level_resolves_dynamic_default(monkeypatch) -> None:
    """The effective log level follows debug mode when no level is explicit."""
    option = get_option("debug.log_level")
    assert option is not None
    monkeypatch.delenv(_env_vars.LOG_LEVEL, raising=False)

    monkeypatch.delenv(_env_vars.DEBUG, raising=False)
    assert _resolve_manifest_option(option, toml_data={}) == ("INFO", "default")

    monkeypatch.setenv(_env_vars.DEBUG, "true")
    assert _resolve_manifest_option(option, toml_data={}) == ("DEBUG", "default")


def test_debug_log_level_validates_explicit_value(monkeypatch) -> None:
    """Explicit log levels are normalized and invalid values use the runtime default."""
    option = get_option("debug.log_level")
    assert option is not None
    monkeypatch.setenv(_env_vars.DEBUG, "true")

    monkeypatch.setenv(_env_vars.LOG_LEVEL, " warning ")
    assert _resolve_manifest_option(option, toml_data={}) == (
        "WARNING",
        f"env ({_env_vars.LOG_LEVEL})",
    )

    monkeypatch.setenv(_env_vars.LOG_LEVEL, "TRACE")
    assert _resolve_manifest_option(option, toml_data={}) == ("DEBUG", "default")


@pytest.mark.parametrize(("debug", "expected"), [(None, "INFO"), ("1", "DEBUG")])
def test_config_get_and_aggregate_report_dynamic_log_level(
    monkeypatch, capsys, debug: str | None, expected: str
) -> None:
    """Both config command paths report the runtime's effective default."""
    import json

    monkeypatch.delenv(_env_vars.LOG_LEVEL, raising=False)
    if debug is None:
        monkeypatch.delenv(_env_vars.DEBUG, raising=False)
    else:
        monkeypatch.setenv(_env_vars.DEBUG, debug)

    get_args = argparse.Namespace(
        config_command="get",
        key="debug.log_level",
        output_format="json",
    )
    assert run_config_command(get_args) == 0
    get_payload = json.loads(capsys.readouterr().out)
    assert get_payload["data"]["value"] == expected

    config_args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(config_args) == 0
    config_payload = json.loads(capsys.readouterr().out)
    row = next(
        item for item in config_payload["data"] if item["key"] == "debug.log_level"
    )
    assert row["value"] == expected


# --- Provider install helpers ----------------------------------------------


def test_provider_install_extra_known_provider() -> None:
    """Known providers resolve to their installing extra."""
    assert provider_install_extra("baseten") == "baseten"
    # Provider name uses underscores; the extra uses hyphens.
    assert provider_install_extra("google_genai") == "google-genai"


def test_provider_install_extra_extra_only_provider() -> None:
    """Providers without required API keys can still resolve to extras."""
    assert provider_install_extra("bedrock") == "bedrock"
    assert provider_install_extra("ollama") == "ollama"


def test_provider_install_extra_unknown_provider() -> None:
    """Providers without a curated extra resolve to `None`."""
    assert provider_install_extra("not-a-real-provider") is None


def test_provider_package_name_known_provider() -> None:
    """Known providers resolve to their PyPI distribution name."""
    assert provider_package_name("baseten") == "langchain-baseten"
    # Every underscore in the import module becomes a hyphen, not just the
    # first -- `nvidia` is the entry with the most.
    assert provider_package_name("nvidia") == "langchain-nvidia-ai-endpoints"
    assert provider_package_name("google_genai") == "langchain-google-genai"


def test_provider_package_name_differs_from_extra() -> None:
    """The distribution name is not the extra name.

    `provider_package_name` feeds a `pypi.org/project/...` link, so confusing
    it with `provider_install_extra` would link an unrelated real project.
    """
    assert provider_install_extra("google_anthropic_vertex") == "vertex"
    assert provider_install_extra("google_vertexai") == "vertex"
    assert (
        provider_package_name("google_anthropic_vertex") == "langchain-google-vertexai"
    )
    assert provider_package_name("google_vertexai") == "langchain-google-vertexai"


def test_provider_package_name_aliased_providers() -> None:
    """Providers sharing an integration resolve to the same distribution."""
    assert provider_package_name("azure_openai") == "langchain-openai"
    assert provider_package_name("openai") == "langchain-openai"


def test_provider_package_name_unknown_provider() -> None:
    """Providers without a curated extra resolve to `None`."""
    assert provider_package_name("not-a-real-provider") is None


def test_is_provider_package_installed_unknown_provider() -> None:
    """Providers without a curated extra are reported as installed."""
    assert is_provider_package_installed("not-a-real-provider") is True


def test_is_provider_package_installed_core_provider() -> None:
    """Core providers ship as base dependencies and are always importable."""
    assert is_provider_package_installed("openai") is True


def test_is_provider_package_installed_missing_extra() -> None:
    """A known provider whose package is absent is reported as not installed."""
    import importlib.util

    if importlib.util.find_spec("langchain_baseten") is not None:
        pytest.skip("langchain_baseten is installed in this environment")
    assert is_provider_package_installed("baseten") is False


# --- Secrets ----------------------------------------------------------------


def test_api_key_credentials_are_secret() -> None:
    """Credential options backed by key/token env vars must be secret-flagged."""
    for opt in get_config_options():
        if opt.group != "Credentials" or not opt.env_var:
            continue
        looks_secret = any(
            marker in opt.env_var for marker in ("KEY", "TOKEN", "APIKEY")
        )
        assert opt.redacted is looks_secret, (
            f"{opt.key} redacted={opt.redacted} but env_var {opt.env_var!r} "
            f"implies redacted={looks_secret}"
        )


@pytest.mark.parametrize(
    ("key", "env_var"),
    [
        ("credentials.google_cloud_location", "GOOGLE_CLOUD_LOCATION"),
        ("credentials.google_anthropic_vertex", "GOOGLE_CLOUD_PROJECT"),
    ],
)
def test_google_cloud_configuration_is_not_secret(key: str, env_var: str) -> None:
    """Google Cloud project and location identifiers are visible configuration."""
    opt = get_option(key)
    assert opt is not None
    assert opt.env_var == env_var
    assert opt.redacted is False


def test_display_value_redacts_secrets() -> None:
    """A secret option never renders its raw value, only configured state."""
    option = ConfigOption(
        key="x",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        redacted=True,
    )
    assert _display_value(option, is_set=True, value="sk-supersecret") == "configured"
    assert _display_value(option, is_set=False, value=None) == "not configured"


def test_display_value_uses_credential_language_for_non_secret_unset() -> None:
    """Non-secret credential identifiers still use configured-state language."""
    option = ConfigOption(
        key="credentials.example",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        redacted=False,
    )
    assert _display_value(option, is_set=False, value=None) == "not configured"


def test_display_value_redacts_structured_table() -> None:
    """A redacted table never renders its dict, only its presence."""
    option = ConfigOption(
        key="agents.async_subagents",
        group="Agents",
        summary="",
        kind=OptionKind.STRUCTURED,
        redacted=True,
    )
    secret = {"researcher": {"headers": {"Authorization": "Bearer sk-secret"}}}
    rendered = _display_value(option, is_set=True, value=secret)
    assert rendered == "configured"
    assert "sk-secret" not in rendered
    assert _display_value(option, is_set=False, value=None) == "(unset)"


def test_missing_extra_hint_checks_provider_dependency(monkeypatch) -> None:
    """Credential rows can show when their provider integration is unavailable."""
    option = ConfigOption(
        key="credentials.example",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        redacted=True,
        dependency_module="langchain_missing_provider",
        install_extra="missing-provider",
    )
    monkeypatch.setattr(
        "deepagents_code.client.commands.config.importlib.util.find_spec",
        lambda name: None if name == "langchain_missing_provider" else object(),
    )
    assert _missing_extra_hint(option) is True
    assert (
        _display_value(option, is_set=True, value="sk-secret")
        == "configured, unavailable"
    )
    assert _source_label("default") == "default"


def test_source_label_marks_prefixed_credential_env_as_session_override() -> None:
    """Only prefixed credential env sources get the session-override note."""
    credential = get_option("credentials.anthropic")
    assert credential is not None
    display = get_option("display.theme")
    assert display is not None

    assert (
        _source_label(
            "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY)",
            option=credential,
        )
        == "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY); session override"
    )
    assert (
        _source_label("env (ANTHROPIC_API_KEY)", option=credential)
        == "env (ANTHROPIC_API_KEY)"
    )
    assert (
        _source_label("env (DEEPAGENTS_CODE_THEME)", option=display)
        == "env (DEEPAGENTS_CODE_THEME)"
    )


def test_run_get_json_omits_secret_value(monkeypatch, capsys) -> None:
    """JSON output for a secret option reports presence but never the value."""
    import json

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-secret")
    assert _run_get("credentials.anthropic", "json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["set"] is True
    assert payload["data"]["value"] is None


@pytest.fixture
def stored_auth_dir(tmp_path, monkeypatch):
    """Redirect the credential store into a temp dir so `/auth` keys are isolated."""
    state_dir = tmp_path / ".deepagents" / ".state"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state_dir)
    return state_dir


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_prefers_stored_over_env(monkeypatch):
    """A `/auth`-stored key wins over an env var, matching runtime precedence."""
    from deepagents_code import auth_store

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "stored"
    assert value == "from-store"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_prefers_prefixed_env_over_stored(monkeypatch):
    """A prefixed credential env var stays authoritative over a stored key."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "from-prefix")
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY)"
    assert value == "from-prefix"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_empty_prefixed_credential_blocks_stored(monkeypatch):
    """An empty prefixed credential suppresses the stored key like the runtime."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "")
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is False
    assert source == "default"
    assert value is None


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_uses_stored_when_env_unset(monkeypatch):
    """A stored key is surfaced even with no env var set."""
    from deepagents_code import auth_store

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", raising=False)
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, _ = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "stored"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_falls_back_to_env_without_stored(monkeypatch):
    """With no stored key, resolution falls through to the env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "env (ANTHROPIC_API_KEY)"
    assert value == "from-env"


def test_resolve_credential_corrupt_store_falls_back_to_env(
    stored_auth_dir, monkeypatch, caplog
):
    """A corrupt `auth.json` is logged and treated as absent, not a hard error."""
    import logging

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    option = get_option("credentials.anthropic")
    assert option is not None
    with caplog.at_level(logging.WARNING):
        is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "env (ANTHROPIC_API_KEY)"
    assert value == "from-env"
    assert "treating as absent" in caplog.text


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_text_reports_stored_source(capsys):
    """`config get` shows a stored credential as configured from the store."""
    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    assert "configured" in out
    assert "stored" in out
    assert "from-store" not in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_text_reports_prefixed_env_as_session_override(monkeypatch, capsys):
    """`config get` labels a visible prefixed credential env var as session-scoped."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "from-prefix")
    auth_store.set_stored_key("anthropic", "from-store")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    compact = " ".join(out.split())
    assert "configured" in out
    assert "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY); session override" in compact
    assert "from-prefix" not in out
    assert "from-store" not in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_json_redacts_stored_secret_value(capsys):
    """`config get --json` reports a stored credential as set but never its value."""
    import json

    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    assert _run_get("credentials.anthropic", "json") == 0
    raw = capsys.readouterr().out
    payload = json.loads(raw)["data"]
    assert payload["set"] is True
    assert payload["source"] == "stored"
    assert payload["value"] is None
    assert "from-store" not in raw


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_config_json_redacts_stored_secret_value(capsys):
    """`config --json` redacts a stored secret on the aggregate path too."""
    import json

    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    raw = capsys.readouterr().out
    rows = json.loads(raw)["data"]
    row = next(r for r in rows if r["key"] == "credentials.anthropic")
    assert row["set"] is True
    assert row["source"] == "stored"
    assert row["value"] is None
    assert "from-store" not in raw


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_config_text_reports_stored_source(capsys):
    """`config` (aggregate text path) shows a stored credential as configured."""
    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    args = argparse.Namespace(config_command=None, output_format="text")
    assert run_config_command(args) == 0
    out = capsys.readouterr().out
    assert "configured" in out
    assert "stored" in out
    assert "from-store" not in out


def test_resolve_empty_stored_key_falls_back_to_env(stored_auth_dir, monkeypatch):
    """A stored entry with a blank key does not mask a working env var."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credentials": {
                    "anthropic": {"type": "api_key", "key": "", "added_at": ""}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "env (ANTHROPIC_API_KEY)"
    assert value == "from-env"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_non_credential_ignores_store():
    """Non-credential options never consult the `/auth` store."""
    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("display.show_header")
    assert option is not None
    _, source, _ = _resolve(option, {}, managed_toml_data={})
    assert source != "stored"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_tavily_service_prefers_stored(monkeypatch):
    """A stored key for the tavily *service* resolves with a stored source.

    Guards that the credential branch keys on group membership (and the
    `provider` field), not on model-provider-registry membership.
    """
    from deepagents_code import auth_store

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_TAVILY_API_KEY", raising=False)
    auth_store.set_stored_key("tavily", "from-store")
    option = get_option("credentials.tavily")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "stored"
    assert value == "from-store"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_tavily_prefixed_env_overrides_stored(monkeypatch):
    """A prefixed tavily env var wins over the stored key, matching runtime."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_TAVILY_API_KEY", "from-prefix")
    auth_store.set_stored_key("tavily", "from-store")
    option = get_option("credentials.tavily")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "env (DEEPAGENTS_CODE_TAVILY_API_KEY)"
    assert value == "from-prefix"


def test_run_get_json_flags_unreadable_store(stored_auth_dir, capsys):
    """`config get --json` for a credential surfaces a store-read failure in-band."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert _run_get("credentials.anthropic", "json") == 0
    payload = json.loads(capsys.readouterr().out)["data"]
    assert "store_error" in payload
    # Redaction still holds even when the store is unreadable.
    assert payload["value"] is None


def test_run_get_json_non_credential_omits_store_error(stored_auth_dir, capsys):
    """A non-credential `config get --json` never carries a `store_error` key."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert _run_get("display.show_header", "json") == 0
    payload = json.loads(capsys.readouterr().out)["data"]
    assert "store_error" not in payload


def test_run_get_pairs_values_and_health_from_one_managed_snapshot(
    monkeypatch, capsys
) -> None:
    """A config invocation refreshes managed provider state exactly once."""
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import (
        ProviderHealth,
        ProviderStatus,
        TomlSnapshot,
    )

    snapshot = TomlSnapshot(
        {},
        ProviderStatus("managed config", None, ProviderHealth.MISSING),
    )
    calls: list[bool] = []

    def get_snapshot(*, refresh: bool = False, path: object = None) -> TomlSnapshot:
        assert path is None
        calls.append(refresh)
        return snapshot

    monkeypatch.setattr(service, "get_managed_snapshot", get_snapshot)

    assert _run_get("display.show_header", "json") == 0
    assert calls == [True]
    capsys.readouterr()


def test_run_config_json_flags_unreadable_store(stored_auth_dir, capsys):
    """`config --json` marks credential rows when the store is unreadable."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    cred_rows = [r for r in rows if r["group"] == "Credentials"]
    assert cred_rows
    assert all("store_error" in r for r in cred_rows)
    assert all("store_error" not in r for r in rows if r["group"] != "Credentials")


def test_run_config_text_warns_on_unreadable_store(stored_auth_dir, capsys):
    """`config` text output warns when the credential store is unreadable.

    Guards the `_print_store_warning` call in both text renderers: without it a
    corrupt store would look identical to an empty one in the interactive view —
    the silent failure this warning exists to prevent.
    """
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    for verbose in (False, True):
        args = argparse.Namespace(
            config_command=None, output_format="text", verbose=verbose
        )
        assert run_config_command(args) == 0
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "unreadable" in out


def test_run_get_text_warns_on_unreadable_store(stored_auth_dir, capsys):
    """`config get` text output shows a warning banner for an unreadable store."""
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "unreadable" in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_config_reads_store_once(monkeypatch):
    """`config` parses the credential store once, not once per option."""
    from deepagents_code import auth_store

    calls = 0
    real_load = auth_store.load_credentials

    def _counting_load() -> dict:
        nonlocal calls
        calls += 1
        return real_load()

    monkeypatch.setattr(auth_store, "load_credentials", _counting_load)
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    # One read for the whole command, regardless of how many credential options
    # exist — guards the single-snapshot design against a per-option regression.
    assert calls == 1


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_non_redacted_credential_shows_stored_value(monkeypatch):
    """A non-redacted stored credential (the Vertex project) shows its value."""
    from deepagents_code import auth_store

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_GOOGLE_CLOUD_PROJECT", raising=False)
    auth_store.set_stored_key("google_vertexai", "my-project")
    option = get_option("credentials.google_vertexai")
    assert option is not None
    assert option.redacted is False
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "stored"
    assert value == "my-project"


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_json_shows_non_redacted_stored_value(capsys):
    """`config get --json` surfaces a non-redacted stored value (not `None`)."""
    import json

    from deepagents_code import auth_store

    auth_store.set_stored_key("google_vertexai", "my-project")
    assert _run_get("credentials.google_vertexai", "json") == 0
    payload = json.loads(capsys.readouterr().out)["data"]
    assert payload["source"] == "stored"
    assert payload["redacted"] is False
    assert payload["value"] == "my-project"


def test_run_get_non_utf8_store_does_not_crash(stored_auth_dir, capsys):
    """A non-UTF-8 `auth.json` degrades to a warning banner, not a traceback."""
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_bytes(b"\xff\xfe not utf-8")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "unreadable" in out


def test_charset_auto_display_value_includes_effective_glyph_mode() -> None:
    """The charset auto value says which glyph mode is actually being used."""
    option = get_option("display.charset")
    assert option is not None
    value = _display_value(option, is_set=False, value="auto")
    assert value in {
        "auto (using Unicode glyphs)",
        "auto (using ASCII glyphs)",
    }
    assert _source_label("default") == "default"


# --- Single-source defaults -------------------------------------------------


def test_interpreter_defaults_match_settings() -> None:
    """Manifest interpreter defaults are the same objects `Settings` uses.

    This is what makes the manifest the single source of truth: the dataclass
    default and the manifest default cannot diverge because they are one value.
    """
    from deepagents_code.config import Settings

    settings = Settings.from_environment()
    for opt in get_config_options():
        if opt.group != "Interpreter" or opt.settings_field is None:
            continue
        assert getattr(settings, opt.settings_field) == opt.default


def test_every_settings_field_names_a_real_settings_attribute() -> None:
    """Catch a typo'd `settings_field` on any option, not just interpreter ones.

    `settings_field` is a free-form string with no compile-time link to the
    `Settings` dataclass, so a misspelling would only surface at runtime
    `getattr`. This locks the mapping across the whole catalog.
    """
    from dataclasses import fields

    from deepagents_code.config import Settings

    valid = {f.name for f in fields(Settings)}
    bad = {
        opt.key: opt.settings_field
        for opt in get_config_options()
        if opt.settings_field is not None and opt.settings_field not in valid
    }
    assert not bad, f"options reference unknown Settings fields: {bad}"


# --- Resolution -------------------------------------------------------------


def test_resolve_prefers_prefixed_env(monkeypatch) -> None:
    """A `DEEPAGENTS_CODE_`-prefixed env var wins over the canonical name."""
    opt = get_option("credentials.openai")
    assert opt is not None
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "prefixed")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert source == "env (DEEPAGENTS_CODE_OPENAI_API_KEY)"
    assert value == "prefixed"


def test_resolve_empty_env_is_unset_matching_resolve_env_var(monkeypatch) -> None:
    """An empty (prefixed) env var is unset for `config`, as the app sees it.

    The runtime `resolve_env_var` returns `None` for an empty prefixed var (and
    a prefixed empty suppresses the canonical). The resolver must agree, or
    `config` would report a credential as "set" that the app treats as
    unset — the exact drift this feature exists to prevent.
    """
    from deepagents_code.model_config import resolve_env_var

    opt = get_option("credentials.openai")
    assert opt is not None
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "")

    value, source = _resolve_manifest_option(opt, toml_data={})
    assert resolve_env_var("OPENAI_API_KEY") is None
    assert source == "default"
    assert value is None


def test_langsmith_project_prefers_prefixed_env(monkeypatch) -> None:
    """The prefixed project env var wins over a bare `LANGSMITH_PROJECT`."""
    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "prefixed")
    monkeypatch.setenv("LANGSMITH_PROJECT", "bare")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == ("prefixed", "env (DEEPAGENTS_CODE_LANGSMITH_PROJECT)")


def test_langsmith_project_falls_back_to_bare_env(monkeypatch) -> None:
    """A bare `LANGSMITH_PROJECT` resolves when the prefixed var is unset.

    Mirrors `get_langsmith_project_name`, so `config`/`config get` report the
    project agent traces actually route to.
    """
    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGSMITH_PROJECT", "bare")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == ("bare", "env (LANGSMITH_PROJECT)")


def test_langsmith_project_default_when_unset(monkeypatch) -> None:
    """With no project env var set, the default project name is rendered."""
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert _resolve_manifest_option(opt, toml_data={}) == (
        LANGSMITH_PROJECT_DEFAULT,
        "default",
    )


def test_langsmith_project_empty_prefixed_falls_through_to_bare(monkeypatch) -> None:
    """An empty prefixed var is skipped, so a set bare `LANGSMITH_PROJECT` wins.

    This is the opposite of the single-name credential path
    (`test_resolve_empty_env_is_unset_matching_resolve_env_var`): with a
    fallback declared, an empty prefixed var does not suppress resolution — it
    falls through to the next name, matching `get_langsmith_project_name`.
    """
    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "")
    monkeypatch.setenv("LANGSMITH_PROJECT", "bare")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == ("bare", "env (LANGSMITH_PROJECT)")


def test_langsmith_project_empty_bare_is_default(monkeypatch) -> None:
    """An empty bare `LANGSMITH_PROJECT` is unset, so the default applies."""
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGSMITH_PROJECT", "")
    assert _resolve_manifest_option(opt, toml_data={}) == (
        LANGSMITH_PROJECT_DEFAULT,
        "default",
    )


def test_onboarding_empty_env_reports_env_opt_out(monkeypatch) -> None:
    """An explicitly empty `DEEPAGENTS_CODE_ONBOARDING` resolves as an env opt-out.

    `should_run_onboarding` treats an empty value as falsy and suppresses the
    flow, so `config` must report the option as set by the environment rather
    than unset/default.
    """
    from deepagents_code.onboarding import should_run_onboarding

    opt = get_option("startup.onboarding")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_ONBOARDING", "")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == (False, "env (DEEPAGENTS_CODE_ONBOARDING)")
    assert should_run_onboarding() is False


def test_fallback_env_vars_yield_to_toml_when_env_unset(monkeypatch) -> None:
    """A synthetic option exercises the empty-fallback → `config.toml` path.

    No shipping option both declares `fallback_env_vars` and has `toml_keys`,
    so this guards the generic resolver: an empty fallback env var must fall
    through to `config.toml`, while a set one still wins over it.
    """
    opt = ConfigOption(
        key="synthetic.fallback_toml",
        group="Synthetic",
        summary="Synthetic option for fallback + TOML precedence.",
        kind=OptionKind.STR,
        env_var=_env_vars.LANGSMITH_PROJECT,
        fallback_env_vars=("SYNTHETIC_FALLBACK",),
        toml_keys=("synthetic", "value"),
    )
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    toml_data = {"synthetic": {"value": "from-toml"}}

    monkeypatch.setenv("SYNTHETIC_FALLBACK", "")
    assert _resolve_manifest_option(opt, toml_data=toml_data) == (
        "from-toml",
        "config.toml",
    )

    monkeypatch.setenv("SYNTHETIC_FALLBACK", "from-env")
    assert _resolve_manifest_option(opt, toml_data=toml_data) == (
        "from-env",
        "env (SYNTHETIC_FALLBACK)",
    )


@pytest.mark.parametrize(
    "bad_fallback",
    [
        ["LANGSMITH_PROJECT"],  # mutable list reintroduces the lru_cache hazard
        ("",),  # empty name never matches any env var
        ("LANGSMITH_PROJECT", ""),  # one valid, one empty
    ],
)
def test_fallback_env_vars_rejects_invalid(bad_fallback) -> None:
    """`__post_init__` rejects non-tuple or empty/non-str `fallback_env_vars`."""
    with pytest.raises(TypeError, match="fallback_env_vars must be a tuple"):
        ConfigOption(
            key="synthetic.bad_fallback",
            group="Synthetic",
            summary="Synthetic option with an invalid fallback.",
            kind=OptionKind.STR,
            fallback_env_vars=bad_fallback,
        )


def test_run_config_json_redacts_every_secret(monkeypatch, capsys) -> None:
    """The `config` aggregate (separate path from `get`) never leaks a secret."""
    import json

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-secret")
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    assert any(r["key"] == "credentials.anthropic" and r["set"] for r in rows)
    assert all(r["value"] is None for r in rows if r["redacted"])


def test_run_get_json_redacts_credential_bearing_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`config get --json` reports a credential-bearing table's presence only.

    `[async_subagents]` headers, `[models.providers]`, and
    `[sandboxes.providers]` can hold tokens, so their rows carry `redacted`
    and a `None` value instead of the raw table.
    """
    from deepagents_code import model_config

    secret = "Bearer sk-secret"
    config = tmp_path / "config.toml"
    config.write_text(
        "[async_subagents.researcher]\n"
        'description = "Research agent"\n'
        'graph_id = "agent"\n'
        "headers = { Authorization = '" + secret + "' }\n"
        "[models.providers.acme]\n"
        'class_path = "acme.Chat:AcmeChat"\n'
        'api_key = "sk-secret"\n'
        "[sandboxes.providers.acme]\n"
        'class_path = "acme.Sandbox:AcmeSandbox"\n'
        "params = { token = 'sk-secret' }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config)

    for key, source in (
        ("agents.async_subagents", "config.toml"),
        ("models.providers", "config.toml"),
        ("sandboxes.providers", "config.toml"),
    ):
        data = _get_json_object(key, capsys)
        assert data["redacted"] is True
        assert data["set"] is True
        assert data["source"] == source
        assert data["value"] is None
    assert "sk-secret" not in capsys.readouterr().out


def test_resolve_int_falls_back_to_toml_then_default() -> None:
    """config.toml is consulted when env is unset; default is the last resort."""
    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"interpreter": {"memory_limit_mb": 128}}
    ) == (
        128,
        "config.toml",
    )
    assert _resolve_manifest_option(opt, toml_data={}) == (64, "default")


def test_resolve_malformed_toml_int_falls_back_with_warning(caplog) -> None:
    """A bad TOML scalar is logged and falls back to the default, never raising."""
    import logging

    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            opt, toml_data={"interpreter": {"memory_limit_mb": "oops"}}
        )
    assert (value, source) == (64, "default")
    assert any("memory_limit_mb" in r.getMessage() for r in caplog.records)


def test_resolve_bool_env_uses_truthy_semantics(monkeypatch) -> None:
    """BOOL options honor is_env_truthy semantics ('0' is falsy, not 'set')."""
    opt = get_option("display.hide_cwd")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "1")
    assert _resolve_manifest_option(opt, toml_data={})[0] is True
    monkeypatch.setenv(opt.env_var, "0")
    assert _resolve_manifest_option(opt, toml_data={})[0] is False


def test_cursor_style_option_definition() -> None:
    """Cursor style supports env and config.toml and defaults to a block."""
    opt = get_option("display.cursor_style")
    assert opt is not None
    assert opt.kind is OptionKind.CURSOR_STYLE_DELEGATE
    assert opt.default == CURSOR_STYLE_DEFAULT
    assert opt.toml_keys == ("ui", "cursor_style")
    assert opt.env_var == _env_vars.CURSOR_STYLE


def test_resolve_cursor_style_from_env(monkeypatch, caplog) -> None:
    """A valid env value wins; an invalid value falls through to config.toml."""
    import logging

    opt = get_option("display.cursor_style")
    assert opt is not None
    toml_data = {"ui": {"cursor_style": "underline"}}

    monkeypatch.setenv(_env_vars.CURSOR_STYLE, "block")
    assert _resolve_manifest_option(opt, toml_data=toml_data) == (
        "block",
        f"env ({_env_vars.CURSOR_STYLE})",
    )

    monkeypatch.setenv(_env_vars.CURSOR_STYLE, "bar")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(opt, toml_data=toml_data) == (
            "underline",
            "config.toml",
        )
    assert any(
        _env_vars.CURSOR_STYLE in record.getMessage() for record in caplog.records
    )


def test_resolve_cursor_style_from_toml(caplog) -> None:
    """Only supported cursor styles are accepted from config.toml."""
    import logging

    opt = get_option("display.cursor_style")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"ui": {"cursor_style": "underline"}}
    ) == (
        "underline",
        "config.toml",
    )

    for raw in ("bar", 1):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
            value, source = _resolve_manifest_option(
                opt, toml_data={"ui": {"cursor_style": raw}}
            )
        assert (value, source) == (CURSOR_STYLE_DEFAULT, "default")
        assert any(
            "[ui].cursor_style" in record.getMessage() for record in caplog.records
        )

    assert _resolve_manifest_option(opt, toml_data={}) == (
        CURSOR_STYLE_DEFAULT,
        "default",
    )


def test_collapse_pastes_default_enabled() -> None:
    """Paste collapsing is enabled by default when unset."""
    opt = get_option("display.collapse_pastes")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL
    assert _resolve_manifest_option(opt, toml_data={}) == (True, "default")


def test_collapse_pastes_env_disables(monkeypatch) -> None:
    """A falsy env var disables paste collapsing."""
    opt = get_option("display.collapse_pastes")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "0")
    assert _resolve_manifest_option(opt, toml_data={})[0] is False


def test_collapse_pastes_toml_disables() -> None:
    """A `[ui].collapse_pastes = false` entry disables paste collapsing."""
    opt = get_option("display.collapse_pastes")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"ui": {"collapse_pastes": False}}
    ) == (
        False,
        "config.toml",
    )


def test_thread_relative_time_default_matches_runtime_loader() -> None:
    """Fresh thread config shows relative timestamps by default."""
    opt = get_option("threads.relative_time")
    assert opt is not None
    assert _resolve_manifest_option(opt, toml_data={}) == (True, "default")


def test_auto_update_resolves_persisted_config() -> None:
    """`set_auto_update()` writes the TOML path surfaced by the manifest."""
    opt = get_option("update.auto_update")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"update": {"auto_update": True}}
    ) == (
        True,
        "config.toml",
    )


def test_no_update_check_env_uses_presence_semantics(monkeypatch) -> None:
    """Any non-empty no-update-check env var disables checks, including '0'."""
    opt = get_option("update.no_update_check")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL_PRESENCE
    monkeypatch.setenv(_env_vars.NO_UPDATE_CHECK, "0")
    assert _resolve_manifest_option(opt, toml_data={}) == (
        True,
        f"env ({_env_vars.NO_UPDATE_CHECK})",
    )


def test_no_update_check_resolves_inverted_persisted_check() -> None:
    """`[update].check = false` means the effective no-check flag is enabled."""
    opt = get_option("update.no_update_check")
    assert opt is not None
    assert _resolve_manifest_option(opt, toml_data={"update": {"check": False}}) == (
        True,
        "config.toml",
    )
    assert _resolve_manifest_option(opt, toml_data={"update": {"check": True}}) == (
        False,
        "config.toml",
    )


def test_prices_auto_update_default_matches_runtime(monkeypatch) -> None:
    """The manifest default must agree with the one `_start_price_updater` uses.

    They are declared independently, so a drift makes `dcode config get
    update.prices_auto_update` report the opposite of what the updater does.
    """
    from deepagents_code import cost_tracking

    opt = get_option("update.prices_auto_update")
    assert opt is not None
    monkeypatch.delenv(_env_vars.PRICES_AUTO_UPDATE, raising=False)
    # Suite-wide fixtures set both; the updater honors OFFLINE as well.
    monkeypatch.delenv(_env_vars.OFFLINE, raising=False)
    # The runtime gate resolves the option, which reads the user's real
    # `config.toml` here; an empty table keeps this test on defaults.
    monkeypatch.setattr("deepagents_code.config_manifest.load_config_toml", dict)

    started: list[object] = []
    monkeypatch.setattr(cost_tracking, "_PRICE_UPDATER_ATTEMPTED", False)
    monkeypatch.setattr(cost_tracking, "_PRICE_UPDATER", None)
    monkeypatch.setattr(
        cost_tracking,
        "_build_price_updater",
        lambda cls: started.append(cls) or MagicMock(),
    )
    cost_tracking._start_price_updater()

    assert opt.default is True
    assert bool(started) is opt.default


def test_prices_auto_update_persists_in_toml(monkeypatch) -> None:
    """The dotted key resolves from `config.toml`, not env vars only."""
    opt = get_option("update.prices_auto_update")
    assert opt is not None
    monkeypatch.delenv(_env_vars.PRICES_AUTO_UPDATE, raising=False)
    assert opt.toml_keys == ("update", "prices_auto_update")
    assert _resolve_manifest_option(
        opt, toml_data={"update": {"prices_auto_update": False}}
    ) == (
        False,
        "config.toml",
    )


def test_prices_auto_update_empty_env_disables(monkeypatch) -> None:
    """An explicitly empty env value opts out, matching `_start_price_updater`."""
    opt = get_option("update.prices_auto_update")
    assert opt is not None
    monkeypatch.setenv(_env_vars.PRICES_AUTO_UPDATE, "")
    assert _resolve_manifest_option(opt, toml_data={}) == (
        False,
        f"env ({_env_vars.PRICES_AUTO_UPDATE})",
    )


def test_resolve_ptc_delegates_to_parser() -> None:
    """The PTC kind routes through the dedicated allowlist parser."""
    opt = get_option("interpreter.ptc")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"interpreter": {"ptc": "safe"}}
    ) == (
        "safe",
        "config.toml",
    )
    # Invalid PTC value is rejected by the parser and falls back to default.
    value, source = _resolve_manifest_option(
        opt, toml_data={"interpreter": {"ptc": "bogus"}}
    )
    assert (value, source) == (opt.default, "default")


def test_resolve_theme_uses_terminal_mapping_before_saved_theme(monkeypatch) -> None:
    """Theme resolution mirrors startup: terminal mapping wins over `[ui].theme`."""
    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    value, source = _resolve_manifest_option(
        opt,
        toml_data={
            "ui": {
                "theme": "atom-one-light",
                "terminal_themes": {"vscode": "ansi-dark"},
            }
        },
    )

    assert value == "ansi-dark"
    assert source == "config.toml [ui.terminal_themes.vscode]"


def test_resolve_theme_uses_saved_theme_without_terminal_match(monkeypatch) -> None:
    """A saved `[ui].theme` is reported when no terminal mapping applies."""
    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "unknown-terminal")

    value, source = _resolve_manifest_option(
        opt,
        toml_data={
            "ui": {
                "theme": "atom-one-light",
                "terminal_themes": {"vscode": "ansi-dark"},
            }
        },
    )

    assert value == "atom-one-light"
    assert source == "config.toml [ui.theme]"


def test_resolve_theme_env_wins_over_config(monkeypatch) -> None:
    """The explicit theme env var wins over saved config, matching startup."""
    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_THEME", "ansi-dark")
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    value, source = _resolve_manifest_option(
        opt,
        toml_data={
            "ui": {
                "theme": "atom-one-light",
                "terminal_themes": {"vscode": "langchain"},
            }
        },
    )

    assert value == "ansi-dark"
    assert source == "env (DEEPAGENTS_CODE_THEME)"


# --- Misc -------------------------------------------------------------------


def test_get_option_unknown_returns_none() -> None:
    assert get_option("does.not.exist") is None


def test_run_get_missing_key_hints_available_keys(capsys) -> None:
    """Bare `config get` explains it needs a key and where to find one."""
    args = argparse.Namespace(config_command="get", key=None, output_format="text")
    assert run_config_command(args) == 2
    err = capsys.readouterr().err
    assert "needs an option key" in err
    assert "dcode config" in err


def test_run_get_missing_key_json_lists_keys(capsys) -> None:
    """JSON output for a bare `config get` returns the full key list for tools."""
    import json

    args = argparse.Namespace(config_command="get", key=None, output_format="json")
    assert run_config_command(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["error"] == "missing key"
    assert payload["data"]["keys"] == list(option_keys())


def test_missing_key_example_is_a_real_option() -> None:
    """The hint's example key must stay a resolvable manifest key."""
    from deepagents_code.client.commands.config import _GET_KEY_EXAMPLE

    assert get_option(_GET_KEY_EXAMPLE) is not None


def test_config_registered_as_bare_action_group() -> None:
    """Bare `config` must run its action instead of startup-fast-path help."""
    from deepagents_code import ui
    from deepagents_code.main import _BARE_ACTION_GROUPS, _HELP_SPECS

    assert "config" in _BARE_ACTION_GROUPS
    assert "config" not in _HELP_SPECS
    assert callable(ui.show_config_help)


# --- ConfigOption validation ------------------------------------------------


def test_config_option_rejects_type_mismatched_default() -> None:
    """A default whose type contradicts `kind` fails at construction."""
    import pytest

    with pytest.raises(TypeError, match="not valid for kind int"):
        ConfigOption(key="x", group="g", summary="s", kind=OptionKind.INT, default="5")


def test_config_option_rejects_bool_default_for_int() -> None:
    """`bool` is an `int` subclass but must not pass as an INT/FLOAT default."""
    import pytest

    with pytest.raises(TypeError, match="not valid for kind int"):
        ConfigOption(key="x", group="g", summary="s", kind=OptionKind.INT, default=True)


def test_config_option_rejects_mutable_default() -> None:
    """A mutable default would be shared by reference through the lru_cache."""
    import pytest

    with pytest.raises(TypeError, match="mutable default"):
        ConfigOption(
            key="x", group="g", summary="s", kind=OptionKind.STR, default=["a"]
        )


def test_config_option_rejects_default_on_structured() -> None:
    """STRUCTURED options are display-only pass-throughs and take no default."""
    import pytest

    with pytest.raises(TypeError, match="must not declare a default"):
        ConfigOption(
            key="x", group="g", summary="s", kind=OptionKind.STRUCTURED, default="x"
        )


def test_config_option_rejects_inverted_non_bool_toml() -> None:
    """Only boolean TOML options can use inverted config-file semantics."""
    import pytest

    with pytest.raises(TypeError, match="requires a boolean option kind"):
        ConfigOption(
            key="x",
            group="g",
            summary="s",
            kind=OptionKind.STR,
            default="x",
            toml_keys=("section", "key"),
            invert_toml_bool=True,
        )


# --- Coercion matrix --------------------------------------------------------


def test_resolve_bool_presence_enables_on_any_value(monkeypatch) -> None:
    """BOOL_PRESENCE treats any non-empty value as set, including '0'.

    This is the one branch whose semantics differ from BOOL, where '0' is
    falsy; here `bool(raw)` makes a literal '0' enable the flag.
    """
    opt = get_option("debug.notifications")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL_PRESENCE
    monkeypatch.setenv(opt.env_var, "0")
    assert _resolve_manifest_option(opt, toml_data={})[0] is True
    monkeypatch.setenv(opt.env_var, "")
    # An empty env value is unset, so resolution falls back to the default.
    assert _resolve_manifest_option(opt, toml_data={}) == (False, "default")


@pytest.mark.parametrize("value", ["0", "false"])
def test_debug_dep_floor_uses_boolean_semantics(monkeypatch, value: str) -> None:
    """Config inspection agrees with the runtime for explicitly falsy values."""
    opt = get_option("debug.dep_floor")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL
    monkeypatch.setenv(_env_vars.DEBUG_DEP_FLOOR, value)
    assert _resolve_manifest_option(opt, toml_data={})[0] is False


def test_resolve_malformed_int_env_falls_back_with_warning(monkeypatch, caplog) -> None:
    """A non-numeric env value for an INT option logs and falls back.

    Interpreter options are TOML-only, so the int env-coercion branch is
    exercised through a synthetic option with an env var.
    """
    import logging

    int_opt = ConfigOption(
        key="t.int",
        group="g",
        summary="s",
        kind=OptionKind.INT,
        default=7,
        env_var="DEEPAGENTS_CODE_TEST_INT",
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_TEST_INT", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(int_opt, toml_data={})
    assert (value, source) == (7, "default")
    assert any("TEST_INT" in r.getMessage() for r in caplog.records)


def test_resolve_toml_int_rejects_bool() -> None:
    """A TOML boolean must not coerce to an INT (bool is an int subclass)."""
    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"interpreter": {"memory_limit_mb": True}}
    ) == (64, "default")


def test_resolve_toml_float_rejects_bool() -> None:
    """A TOML boolean must not coerce to a FLOAT."""
    opt = get_option("interpreter.timeout_seconds")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"interpreter": {"timeout_seconds": True}}
    ) == (5.0, "default")


def test_resolve_structured_passes_value_through() -> None:
    """STRUCTURED options return the raw table verbatim for display."""
    opt = get_option("threads.columns")
    assert opt is not None
    assert opt.kind is OptionKind.STRUCTURED
    table = {"created": True, "updated": False}
    assert _resolve_manifest_option(opt, toml_data={"threads": {"columns": table}}) == (
        table,
        "config.toml",
    )


def test_structured_fallback_preserves_invalid_tier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A defaultless table fallback retains the rejected provider metadata."""
    from deepagents_code.configuration.resolver import (
        USER_RANK,
        resolver_from_snapshots,
    )
    from deepagents_code.configuration.types import (
        Invalid,
        ProviderHealth,
        ProviderStatus,
        TomlSnapshot,
    )

    option = get_option("display.terminal_themes")
    assert option is not None
    toml_data = {"ui": "dark"}

    resolved = resolver_from_snapshots(
        managed=TomlSnapshot(
            {}, ProviderStatus("managed config", None, ProviderHealth.OK)
        ),
        user=TomlSnapshot(
            toml_data, ProviderStatus("config.toml", None, ProviderHealth.OK)
        ),
    ).get(option)
    assert resolved.value is None
    assert isinstance(resolved.tier_health[USER_RANK], Invalid)

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option,
            toml_data=toml_data,
            managed_toml_data={},
        ) == (None, "default")
    assert any("expected a table" in record.message for record in caplog.records)


def test_resolve_malformed_skills_dir_env_falls_back(monkeypatch, caplog) -> None:
    """An unresolvable skills-dir env path logs and falls back, never raising."""
    import logging

    opt = get_option("skills.extra_allowed_dirs")
    assert opt is not None
    # `~nobodyuser_xyz` cannot resolve to a home directory; `expanduser` raises
    # RuntimeError, which the resolver must catch.
    monkeypatch.setenv(opt.env_var, "~nobodyuser_xyz/skills")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == (None, "default")
    assert any("could not resolve" in r.getMessage() for r in caplog.records)


def test_resolve_malformed_skills_dir_toml_falls_back(caplog) -> None:
    """An unresolvable skills-dir in config.toml logs and falls back."""
    import logging

    opt = get_option("skills.extra_allowed_dirs")
    assert opt is not None
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            opt,
            toml_data={"skills": {"extra_allowed_dirs": ["~nobodyuser_xyz/skills"]}},
        )
    assert (value, source) == (None, "default")
    assert any("could not resolve" in r.getMessage() for r in caplog.records)


# --- load_config_toml -------------------------------------------------------


def test_load_config_toml_absent_returns_empty(monkeypatch, tmp_path) -> None:
    """An absent config file is not an error: returns {} silently."""
    from deepagents_code import config_manifest, model_config

    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "missing.toml")
    assert config_manifest.load_config_toml() == {}


def test_load_config_toml_corrupt_returns_empty_with_warning(
    monkeypatch, tmp_path, caplog
) -> None:
    """A corrupt config file logs a warning and falls back to {}."""
    import logging

    from deepagents_code import config_manifest, model_config

    bad = tmp_path / "config.toml"
    bad.write_text("this is = not valid = toml ][")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", bad)
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert config_manifest.load_config_toml() == {}
    assert any("Could not read config" in r.getMessage() for r in caplog.records)


def test_load_config_toml_valid_parses(monkeypatch, tmp_path) -> None:
    """A valid config file is parsed into a mapping."""
    from deepagents_code import config_manifest, model_config

    good = tmp_path / "config.toml"
    good.write_text("[interpreter]\nmemory_limit_mb = 128\n")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", good)
    assert config_manifest.load_config_toml() == {
        "interpreter": {"memory_limit_mb": 128}
    }


# --- Display rendering ------------------------------------------------------


def test_display_value_unset_renders_placeholder() -> None:
    """A non-secret option with no value renders the unset placeholder."""
    opt = ConfigOption(key="x", group="g", summary="s", kind=OptionKind.STR)
    assert _display_value(opt, is_set=False, value=None) == "(unset)"


def test_display_value_truncates_long_values() -> None:
    """A long value is truncated to 60 chars with a trailing ellipsis."""
    opt = ConfigOption(key="x", group="g", summary="s", kind=OptionKind.STR)
    rendered = _display_value(opt, is_set=True, value="a" * 100)
    assert len(rendered) == 60
    assert rendered.endswith("\N{HORIZONTAL ELLIPSIS}")


def test_config_text_survives_markup_in_value(monkeypatch) -> None:
    """A value containing Rich close-tag markup must not crash text rendering."""
    monkeypatch.setenv(
        _env_vars.EXTERNAL_EVENT_SOCKET_PATH,
        "/tmp/sock[/]oops",
    )
    args = argparse.Namespace(config_command=None, output_format="text")
    assert run_config_command(args) == 0


def test_config_verbose_text_survives_markup_in_value(monkeypatch) -> None:
    """The verbose text path escapes markup in values so rendering can't break.

    `_print_config_verbose` renders with markup enabled and relies on manual
    `escape()`; the compact table path uses `Text` cells, so it needs its own
    guard.
    """
    monkeypatch.setenv(
        _env_vars.EXTERNAL_EVENT_SOCKET_PATH,
        "/tmp/sock[/]oops",
    )
    args = argparse.Namespace(config_command=None, output_format="text", verbose=True)
    assert run_config_command(args) == 0


# --- Command smoke (text paths) ---------------------------------------------


def test_run_config_text_returns_zero() -> None:
    """The default (text) `config` rendering path runs without error."""
    args = argparse.Namespace(config_command=None, output_format="text")
    assert run_config_command(args) == 0


def test_run_config_verbose_text_shows_descriptions(capsys) -> None:
    """`config --verbose` folds in each option's description and how-to-set.

    A plain exit-code check would still pass if the verbose path silently
    regressed to the compact table, so assert the distinguishing content — an
    option's summary and its `set via` line — is actually rendered.
    """
    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    args = argparse.Namespace(config_command=None, output_format="text", verbose=True)
    assert run_config_command(args) == 0
    # Normalize whitespace so Rich soft-wrapping at the test console width can't
    # break the substring match.
    rendered = " ".join(capsys.readouterr().out.split())
    assert " ".join(opt.summary.split()) in rendered
    assert "set via" in rendered


def test_config_parser_wires_default_and_verbose_flag(monkeypatch) -> None:
    """Real argparse wiring parses bare `config --all --json` as verbose JSON."""
    import sys

    from deepagents_code.main import parse_args

    monkeypatch.setattr(sys, "argv", ["dcode", "config", "--all", "--json"])
    ns = parse_args()
    assert ns.config_command is None
    assert ns.verbose is True
    assert ns.output_format == "json"


@pytest.mark.parametrize(
    "argv",
    [
        ["dcode", "config", "get", "credentials", "--verbose", "--json"],
        ["dcode", "config", "--verbose", "--json", "get", "credentials"],
    ],
)
def test_config_get_parser_accepts_verbose_on_either_side(monkeypatch, argv) -> None:
    """`--verbose`/`--json` reach `config get` before or after the subcommand.

    The `get` subparser suppresses its own defaults, so a flag set on the parent
    `config` parser survives the subparser namespace merge.
    """
    import sys

    from deepagents_code.main import parse_args

    monkeypatch.setattr(sys, "argv", argv)
    ns = parse_args()
    assert ns.config_command == "get"
    assert ns.key == "credentials"
    assert ns.verbose is True
    assert ns.output_format == "json"


@pytest.mark.parametrize("removed_subcommand", ["show", "list", "ls"])
def test_config_parser_rejects_removed_subcommands(
    monkeypatch, removed_subcommand
) -> None:
    """Removed display subcommands are no longer registered on the parser."""
    import sys

    from deepagents_code.main import parse_args

    monkeypatch.setattr(sys, "argv", ["dcode", "config", removed_subcommand])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_run_get_text_returns_zero() -> None:
    """The default (text) `config get` rendering path runs without error."""
    args = argparse.Namespace(
        config_command="get", key="interpreter.memory_limit_mb", output_format="text"
    )
    assert run_config_command(args) == 0


# --- `config get` sections --------------------------------------------------


def _get_args(
    key: str, output_format: str = "text", *, verbose: bool = False
) -> argparse.Namespace:
    """Build a parsed `config get <key>` namespace."""
    return argparse.Namespace(
        config_command="get", key=key, output_format=output_format, verbose=verbose
    )


def _get_json_data(
    key: str, capsys: pytest.CaptureFixture[str], *, verbose: bool = False
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run `config get <key> --json` and return the decoded `data` payload."""
    import json

    assert run_config_command(_get_args(key, "json", verbose=verbose)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "config get"
    return payload["data"]


def _get_json_object(
    key: str, capsys: pytest.CaptureFixture[str], *, verbose: bool = False
) -> dict[str, Any]:
    """Return the single-option `config get --json` payload for `key`."""
    data = _get_json_data(key, capsys, verbose=verbose)
    assert isinstance(data, dict)
    return data


def _get_json_rows(
    key: str, capsys: pytest.CaptureFixture[str], *, verbose: bool = False
) -> list[dict[str, Any]]:
    """Return the section `config get --json` rows for `key`."""
    data = _get_json_data(key, capsys, verbose=verbose)
    assert isinstance(data, list)
    return data


def test_options_with_key_prefix_matches_whole_segments_only() -> None:
    """A key prefix matches on the dot boundary, never a truncated guess."""
    matched = options_with_key_prefix("credentials")
    assert len(matched) > 1
    assert all(opt.key.startswith("credentials.") for opt in matched)
    assert options_with_key_prefix("credential") == ()
    assert options_with_key_prefix("") == ()
    # No `tools.*` key exists, so the `Tools` heading has no prefix section.
    assert options_with_key_prefix("tools") == ()


def test_options_with_key_prefix_is_case_insensitive() -> None:
    """Prefix matching casefolds, so capitalization cannot change the result."""
    lower = options_with_key_prefix("models")
    assert len(lower) > 1
    assert options_with_key_prefix("Models") == lower
    assert options_with_key_prefix("MODELS") == lower
    # A truncated guess stays unmatched in every case.
    assert options_with_key_prefix("Model") == ()


def test_run_get_exact_key_keeps_single_line_text(capsys) -> None:
    """An exact key still renders the compact `key = value (source)` line."""
    assert run_config_command(_get_args("interpreter.memory_limit_mb")) == 0
    rendered = " ".join(capsys.readouterr().out.split())
    assert rendered.startswith("interpreter.memory_limit_mb = ")


def test_run_get_exact_key_json_stays_single_object(capsys) -> None:
    """Existing single-key JSON consumers keep their one-object payload."""
    data = _get_json_object("interpreter.memory_limit_mb", capsys)
    assert data["key"] == "interpreter.memory_limit_mb"
    assert "group" not in data


def test_run_get_credentials_section_lists_every_credential(
    capsys, monkeypatch
) -> None:
    """`config get credentials` renders the grouped section, not one option."""
    from deepagents_code.config import console

    monkeypatch.setattr(console, "width", 200)
    assert run_config_command(_get_args("credentials")) == 0
    rendered = capsys.readouterr().out
    assert "Credentials" in rendered
    expected = options_with_key_prefix("credentials")
    assert len(expected) > 1
    assert all(opt.key in rendered for opt in expected)
    # Credentials render a presence word, never a key value. Matches both
    # "configured" and "not configured", so this does not depend on which
    # providers the developer running the suite happens to have set.
    assert "configured" in rendered


@pytest.mark.parametrize("key", ["credentials", "credentials.", "CREDENTIALS"])
def test_run_get_credentials_section_accepts_prefix_dot_and_case(key, capsys) -> None:
    """Prefix, trailing dot, and any casing all select the same options."""
    rows = _get_json_rows(key, capsys)
    assert [row["key"] for row in rows] == [
        opt.key for opt in options_with_key_prefix("credentials")
    ]
    # Secret-flagged rows report presence only, exactly as bare `config` does.
    assert any(row["redacted"] for row in rows)
    assert all(row["value"] is None for row in rows if row["redacted"])


def test_run_get_section_json_matches_bare_config_row_shape(capsys) -> None:
    """Section rows carry the same fields as each bare `config --json` row."""
    rows = _get_json_rows("interpreter", capsys)
    assert len(rows) > 1
    assert all(
        {"key", "group", "source", "set", "redacted", "value"} <= set(row)
        for row in rows
    )
    assert all("type" not in row for row in rows)


def test_run_get_section_verbose_json_folds_in_catalog(capsys) -> None:
    """`config get <section> --verbose --json` adds the static catalog fields."""
    rows = _get_json_rows("interpreter", capsys, verbose=True)
    assert all(
        {"summary", "type", "default", "env_var", "toml_path", "cli_flag"} <= set(row)
        for row in rows
    )
    assert any(
        row["key"] == "interpreter.memory_limit_mb" and row["default"] == 64
        for row in rows
    )


def test_run_get_section_verbose_text_shows_descriptions(capsys) -> None:
    """`config get <section> --verbose` matches the verbose bare-`config` view."""
    opt = get_option("display.charset")
    assert opt is not None
    assert run_config_command(_get_args("display", verbose=True)) == 0
    rendered = " ".join(capsys.readouterr().out.split())
    assert " ".join(opt.summary.split()) in rendered
    assert "set via" in rendered


def test_run_get_section_selects_every_prefixed_key_across_headings(capsys) -> None:
    """`models` resolves to every `models.*` key, even across group headings."""
    rows = _get_json_rows("models", capsys)
    keys = [row["key"] for row in rows]
    assert keys == [opt.key for opt in options_with_key_prefix("models")]
    # The `Models` heading covers only some of the `models.*` keys; the
    # section is the full prefix set, regardless of display grouping.
    assert len(keys) > sum(opt.group == "Models" for opt in get_config_options())


@pytest.mark.parametrize("title", ["Tools", "Updates"])
def test_run_get_group_title_is_not_a_section(title, capsys) -> None:
    """Display group titles are rejected; only key prefixes name a section.

    `Tools` has no matching key prefix at all, and `Updates` differs from the
    `update` prefix by an English plural — matching is case-insensitive on the
    prefix but never pluralized or fuzzy, so a heading word that isn't a
    prefix stays an error.
    """
    assert run_config_command(_get_args(title)) == 1
    assert "Unknown config option or section" in capsys.readouterr().err


def test_run_get_single_option_section_still_renders_as_a_list(capsys) -> None:
    """A one-option section stays list-shaped so responses are unambiguous.

    This is what the `is_exact` flag on `_Selection` protects: a single-option
    section and an exact key both yield one option, so a length check alone
    would collapse this into the single-key object shape.
    """
    prefix = next(
        segment
        for segment in dict.fromkeys(
            key.split(".")[0] for key in option_keys() if "." in key
        )
        if len(options_with_key_prefix(segment)) == 1
    )
    rows = _get_json_rows(prefix, capsys)
    assert [row["key"] for row in rows] == [
        opt.key for opt in options_with_key_prefix(prefix)
    ]
    assert len(rows) == 1


def test_run_get_section_is_case_insensitive_end_to_end(capsys) -> None:
    """A capitalized section selects the same options as the lowercase form.

    `dcode config` prints `Models` as a heading; typing that word back must
    resolve to the same `models.*` prefix section, not to the narrower heading
    or to an error.
    """
    lower = _get_json_rows("models", capsys)
    upper = _get_json_rows("Models", capsys)
    assert [row["key"] for row in upper] == [row["key"] for row in lower]
    # The `Models` heading covers only some of the `models.*` keys, so this
    # equality holds only when both casings resolve via the prefix.
    assert len(lower) > sum(opt.group == "Models" for opt in get_config_options())


def test_run_get_exact_key_verbose_json_folds_in_catalog(capsys) -> None:
    """`config get <key> --verbose --json` adds catalog fields but no `group`.

    The absent `group` key is load-bearing: it is how a consumer tells the
    single-key object from a section list, so `--verbose` must not introduce it.
    """
    data = _get_json_object("interpreter.memory_limit_mb", capsys, verbose=True)
    assert {
        "summary",
        "type",
        "default",
        "env_var",
        "toml_path",
        "cli_flag",
    } <= set(data)
    assert data["default"] == 64
    assert "group" not in data


def test_run_get_exact_key_plain_json_omits_catalog(capsys) -> None:
    """Without `--verbose`, the single-key payload keeps its original fields."""
    data = _get_json_object("interpreter.memory_limit_mb", capsys)
    assert set(data) == {"key", "source", "set", "redacted", "value"}


def test_run_get_exact_key_verbose_text_shows_description(capsys) -> None:
    """`config get <key> --verbose` appends the description and how-to-set."""
    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    assert run_config_command(_get_args(opt.key, verbose=True)) == 0
    rendered = " ".join(capsys.readouterr().out.split())
    assert rendered.startswith(f"{opt.key} = ")
    assert " ".join(opt.summary.split()) in rendered
    assert "default 64" in rendered


def test_run_get_section_json_flags_unreadable_store(stored_auth_dir, capsys) -> None:
    """A corrupt store is reported in-band on every credential section row.

    Without this a corrupt `auth.json` is indistinguishable from one holding no
    keys — the silent failure `store_error` exists to prevent.
    """
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert run_config_command(_get_args("credentials", "json")) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    assert rows
    assert all("store_error" in row for row in rows)
    # Redaction holds even when the store could not be read.
    assert all(row["value"] is None for row in rows if row["redacted"])


def test_run_get_section_text_warns_on_unreadable_store(
    stored_auth_dir, capsys
) -> None:
    """Both section text renderers surface the unreadable-store banner."""
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    for verbose in (False, True):
        assert run_config_command(_get_args("credentials", verbose=verbose)) == 0
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "unreadable" in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_section_skips_store_when_no_credentials(monkeypatch) -> None:
    """A credential-free section never reads the store, and reads it once if not.

    Pins the documented skip in `_run_get_section`: always loading would emit a
    spurious store warning for sections that cannot consult it, and the JSON
    would look identical either way.

    Bootstrap is marked done first: `_ensure_bootstrap` applies stored
    LangSmith auth (two store reads) only on the first call in the process, so
    without the pin this test's counts depend on which tests ran earlier in
    the same worker.
    """
    from deepagents_code import auth_store, config as config_mod

    monkeypatch.setattr(config_mod._bootstrap_state, "done", True)

    calls = 0
    real_load = auth_store.load_credentials

    def _counting_load() -> dict:
        nonlocal calls
        calls += 1
        return real_load()

    monkeypatch.setattr(auth_store, "load_credentials", _counting_load)

    assert run_config_command(_get_args("display", "json")) == 0
    assert calls == 0

    assert run_config_command(_get_args("credentials", "json")) == 0
    assert calls == 1


@pytest.mark.parametrize("key", ["credential", "credentials.open", "nope", "", "."])
def test_run_get_rejects_partial_matches(key, capsys) -> None:
    """Truncated keys, partial leaves, typos, and degenerate input stay errors."""
    assert run_config_command(_get_args(key)) == 1
    err = capsys.readouterr().err
    assert "Unknown config option" in err
    assert "`dcode config` to list keys" in err
    assert "config get <section>" in err


def test_run_get_unknown_key_json_names_sections_too(capsys) -> None:
    """The JSON error string matches the text branch's wording.

    A consumer surfacing `error` verbatim would otherwise tell a user who
    mistyped a section that no such *option* exists, steering them away from the
    section form.
    """
    import json

    assert run_config_command(_get_args("nope", "json")) == 1
    payload = json.loads(capsys.readouterr().out)["data"]
    assert payload == {"key": "nope", "error": "unknown option or section"}


def test_run_path_text_returns_zero() -> None:
    """The `config path` rendering path runs without error."""
    args = argparse.Namespace(config_command="path", output_format="text")
    assert run_config_command(args) == 0


# --- BOOL env coercion ------------------------------------------------------


def test_resolve_bool_unrecognized_env_falls_back_with_warning(
    monkeypatch, caplog
) -> None:
    """An unrecognized boolean env token logs and falls through, not source=env.

    `is_env_truthy` would silently return the default for `maybe`, but the
    resolver must not then credit the env var with that value: doing so would
    make `config` report `source=env` for a variable the runtime ignored.
    """
    import logging

    opt = get_option("display.hide_cwd")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "maybe")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == (False, "default")
    assert any("expected bool" in r.getMessage() for r in caplog.records)


# --- FLOAT / shell-list env coercion ---------------------------------------


def test_resolve_float_env_coerces_and_falls_back(monkeypatch, caplog) -> None:
    """The FLOAT env branch coerces a number and logs+falls back on garbage.

    Interpreter floats are TOML-only, so — like the INT branch — a synthetic
    env-backed option exercises both arms of the FLOAT env-coercion path.
    """
    import logging

    float_opt = ConfigOption(
        key="t.float",
        group="g",
        summary="s",
        kind=OptionKind.FLOAT,
        default=1.5,
        env_var="DEEPAGENTS_CODE_TEST_FLOAT",
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_TEST_FLOAT", "2.5")
    value, source = _resolve_manifest_option(float_opt, toml_data={})
    assert value == pytest.approx(2.5)
    assert source == "env (DEEPAGENTS_CODE_TEST_FLOAT)"

    monkeypatch.setenv("DEEPAGENTS_CODE_TEST_FLOAT", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(float_opt, toml_data={})
    assert (value, source) == (1.5, "default")
    assert any("TEST_FLOAT" in r.getMessage() for r in caplog.records)


def test_resolve_shell_list_env_happy_and_invalid(monkeypatch, caplog) -> None:
    """The shell-list env delegate parses a valid list and rejects bad input."""
    import logging

    opt = get_option("shell.allow_list")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "git status,ls")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert source == f"env ({opt.env_var})"
    assert isinstance(value, list)
    assert "ls" in value

    # `'all'` cannot be combined with other commands; the parser raises and the
    # resolver logs + falls back rather than crashing.
    monkeypatch.setenv(opt.env_var, "all,ls")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(opt, toml_data={})
    assert source == "default"
    assert any("Ignoring invalid" in r.getMessage() for r in caplog.records)


def test_environment_coercion_delegate_returns_invalid_not_raw() -> None:
    """A delegate kind reaching env coercion returns `Invalid`, never raw.

    PTC/STRUCTURED options declare no env var, so this branch is unreachable in
    the live manifest. The guard exists so that if one ever gains an env var,
    an uncoerced raw string cannot leak into a typed `Settings` field.
    """
    from deepagents_code.configuration.providers import coerce_environment_value
    from deepagents_code.configuration.types import Invalid

    opt = get_option("interpreter.ptc")
    assert opt is not None
    result = coerce_environment_value(opt, "safe", "DEEPAGENTS_CODE_FAKE")
    assert isinstance(result, Invalid)
    assert "not env-backed" in result.reason


# --- TOML coercion (success + mismatch) ------------------------------------


def test_resolve_toml_str_success_and_type_mismatch(caplog) -> None:
    """A STR option reads a string from TOML and rejects a wrong-typed value."""
    import logging

    opt = get_option("threads.sort_order")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"threads": {"sort_order": "created_at"}}
    ) == (
        "created_at",
        "config.toml",
    )

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            opt, toml_data={"threads": {"sort_order": 123}}
        )
    assert (value, source) == ("updated_at", "default")
    assert any("sort_order" in r.getMessage() for r in caplog.records)


def test_startup_mode_option_definition() -> None:
    """`startup.mode` is config.toml-only and uses runtime mode validation."""
    opt = get_option("startup.mode")
    assert opt is not None
    assert opt.group == "Startup"
    assert opt.kind is OptionKind.STARTUP_MODE_DELEGATE
    assert opt.default == DEFAULT_STARTUP_MODE
    assert opt.toml_keys == ("startup", "mode")
    assert opt.env_var is None


def test_startup_yolo_switcher_defaults_enabled(monkeypatch) -> None:
    """`startup.yolo_switcher` resolves to enabled when nothing overrides it."""
    option = get_option("startup.yolo_switcher")
    assert option is not None
    assert option.group == "Startup"
    assert option.default is True
    assert option.env_var == _env_vars.YOLO_SWITCHER
    assert option.toml_keys == ("startup", "yolo_switcher")
    monkeypatch.delenv(_env_vars.YOLO_SWITCHER, raising=False)
    assert _resolve_manifest_option(option, toml_data={}) == (True, "default")


def test_startup_yolo_switcher_env_disables(monkeypatch) -> None:
    """A falsy `DEEPAGENTS_CODE_YOLO_SWITCHER` removes YOLO from the switcher."""
    option = get_option("startup.yolo_switcher")
    assert option is not None
    monkeypatch.setenv(_env_vars.YOLO_SWITCHER, "0")
    value, _ = _resolve_manifest_option(option, toml_data={})
    assert value is False


def test_startup_yolo_switcher_empty_env_disables(monkeypatch) -> None:
    """An explicitly empty env override disables the YOLO switcher entry."""
    option = get_option("startup.yolo_switcher")
    assert option is not None
    monkeypatch.setenv(_env_vars.YOLO_SWITCHER, "")
    assert _resolve_manifest_option(option, toml_data={}) == (
        False,
        f"env ({_env_vars.YOLO_SWITCHER})",
    )


def test_startup_yolo_switcher_toml_disables(monkeypatch) -> None:
    """`[startup].yolo_switcher = false` disables the YOLO switcher entry."""
    option = get_option("startup.yolo_switcher")
    assert option is not None
    monkeypatch.delenv(_env_vars.YOLO_SWITCHER, raising=False)
    value, _ = _resolve_manifest_option(
        option, toml_data={"startup": {"yolo_switcher": False}}
    )
    assert value is False


def test_is_yolo_switcher_enabled_reads_env(monkeypatch) -> None:
    """The `config.is_yolo_switcher_enabled` helper honors the env override."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_yolo_switcher_enabled

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    monkeypatch.delenv(_env_vars.YOLO_SWITCHER, raising=False)
    assert is_yolo_switcher_enabled() is True

    monkeypatch.setenv(_env_vars.YOLO_SWITCHER, "false")
    assert is_yolo_switcher_enabled() is False


def test_is_yolo_switcher_enabled_reads_toml(monkeypatch, tmp_path: Path) -> None:
    """The helper honors `[startup].yolo_switcher` when env is unset."""
    from deepagents_code.config import is_yolo_switcher_enabled

    monkeypatch.delenv(_env_vars.YOLO_SWITCHER, raising=False)
    (tmp_path / "config.toml").write_text(
        "[startup]\nyolo_switcher = false\n", encoding="utf-8"
    )
    assert is_yolo_switcher_enabled() is False


def test_resolve_startup_mode_from_toml(caplog) -> None:
    """`startup.mode` resolves only valid runtime modes from config.toml."""
    import logging

    opt = get_option("startup.mode")
    assert opt is not None
    for mode in ("auto", "yolo"):
        assert _resolve_manifest_option(opt, toml_data={"startup": {"mode": mode}}) == (
            mode,
            "config.toml",
        )
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            opt, toml_data={"startup": {"mode": "dangerously-auto"}}
        )
    assert (value, source) == (DEFAULT_STARTUP_MODE, "default")
    assert any(
        "[startup].mode='dangerously-auto'" in r.getMessage() for r in caplog.records
    )

    for raw in (["manual"], {"name": "manual"}):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
            value, source = _resolve_manifest_option(
                opt, toml_data={"startup": {"mode": raw}}
            )
        assert (value, source) == (DEFAULT_STARTUP_MODE, "default")
        assert any("[startup].mode" in r.getMessage() for r in caplog.records)

    assert _resolve_manifest_option(opt, toml_data={}) == (
        DEFAULT_STARTUP_MODE,
        "default",
    )


@pytest.mark.parametrize(
    ("toml_data", "managed_toml_data", "expected"),
    [
        # Only `recent` set: a bare launch runs Auto, so the display must agree.
        ({"startup": {"recent": "auto"}}, {}, ("auto", "config.toml")),
        # An explicit mode outranks `recent`.
        (
            {"startup": {"mode": "manual", "recent": "auto"}},
            {},
            ("manual", "config.toml"),
        ),
        # Nothing configured: the typed default stands.
        ({}, {}, (DEFAULT_STARTUP_MODE, "default")),
        # An unsafe `recent` fails closed rather than crediting config.toml.
        ({"startup": {"recent": "yolo"}}, {}, (DEFAULT_STARTUP_MODE, "default")),
        # A non-scalar `recent` cannot reach the membership test.
        ({"startup": {"recent": ["auto"]}}, {}, (DEFAULT_STARTUP_MODE, "default")),
        # An invalid explicit mode is fail-closed: `load_startup_mode` returns
        # Manual without consulting `recent`, so the display must too.
        (
            {"startup": {"mode": "hands-off", "recent": "auto"}},
            {},
            (DEFAULT_STARTUP_MODE, "default"),
        ),
        # Managed `recent` participates in the same precedence as runtime loading.
        ({}, {"startup": {"recent": "auto"}}, ("auto", "managed config")),
        (
            {"startup": {"recent": "auto"}},
            {"startup": {"recent": "manual"}},
            ("manual", "managed config"),
        ),
    ],
    ids=[
        "recent-only",
        "explicit-outranks-recent",
        "nothing-configured",
        "unsafe-recent",
        "non-scalar-recent",
        "invalid-explicit-mode",
        "managed-recent",
        "managed-recent-outranks-user",
    ],
)
def test_resolve_startup_mode_with_source_reports_recent_fallback(
    monkeypatch: pytest.MonkeyPatch,
    toml_data: dict,
    managed_toml_data: dict,
    expected: tuple[str, str],
) -> None:
    """`startup.mode` display reflects the `[startup].recent` restore."""
    from deepagents_code import approval_mode
    from deepagents_code.config_manifest import resolve_startup_mode_with_source

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: True)

    assert (
        resolve_startup_mode_with_source(
            toml_data=toml_data,
            managed_toml_data=managed_toml_data,
        )
        == expected
    )


@pytest.mark.parametrize(
    "config_text",
    [
        "",
        "[startup]\n",
        "[startup]\nrecent = 'auto'\n",
        "[startup]\nrecent = 'manual'\n",
        "[startup]\nrecent = 'yolo'\n",
        "[startup]\nrecent = ['auto']\n",
        # Whitespace and blanks: the display must not accept a value the
        # loader's exact match rejects.
        "[startup]\nrecent = ' auto '\n",
        "[startup]\nrecent = 'AUTO'\n",
        "[startup]\nrecent = ''\n",
        "[startup]\nmode = 'auto'\n",
        "[startup]\nmode = 'yolo'\nrecent = 'manual'\n",
        "[startup]\nmode = 'hands-off'\nrecent = 'auto'\n",
        "[startup]\nmode = ['auto']\nrecent = 'auto'\n",
        "startup = 'nonsense'\n",
    ],
)
def test_resolve_startup_mode_with_source_agrees_with_loader(
    tmp_path, monkeypatch: pytest.MonkeyPatch, config_text: str
) -> None:
    """The display resolver and the runtime loader must never disagree.

    Both consult `[startup].mode`, `[startup].recent`, and the Auto notice, in
    two separate implementations held together only by a docstring. Per-case
    assertions cannot catch drift between them; this can.
    """
    from deepagents_code import approval_mode
    from deepagents_code.config_manifest import resolve_startup_mode_with_source
    from deepagents_code.model_config import load_startup_mode

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: True)
    config = tmp_path / "config.toml"
    config.write_text(config_text)
    with config.open("rb") as file:
        toml_data = tomllib.load(file)

    displayed, _ = resolve_startup_mode_with_source(
        toml_data=toml_data,
        managed_toml_data={},
    )
    assert displayed == load_startup_mode(config)


def test_resolve_startup_mode_with_source_gates_recent_auto_on_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The displayed fallback matches a launch blocked by a stale notice."""
    from deepagents_code import approval_mode
    from deepagents_code.config_manifest import resolve_startup_mode_with_source

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: False)

    assert resolve_startup_mode_with_source(
        toml_data={},
        managed_toml_data={"startup": {"recent": "auto"}},
    ) == (DEFAULT_STARTUP_MODE, "default")


def test_resolve_toml_float_success_non_bool() -> None:
    """A FLOAT option reads a real number from TOML and coerces an int to float."""
    opt = get_option("interpreter.timeout_seconds")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"interpreter": {"timeout_seconds": 2.5}}
    ) == (
        2.5,
        "config.toml",
    )
    # A bare TOML integer is accepted and coerced to float.
    assert _resolve_manifest_option(
        opt, toml_data={"interpreter": {"timeout_seconds": 3}}
    ) == (
        3.0,
        "config.toml",
    )


# --- Theme resolution warnings ----------------------------------------------


def test_resolve_theme_unknown_env_warns(monkeypatch, caplog) -> None:
    """An unknown theme in the env var warns and falls back to the default."""
    import logging

    from deepagents_code import theme

    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_THEME", "no-such-theme")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == (theme.DEFAULT_THEME, "default")
    assert any("Unknown theme" in r.getMessage() for r in caplog.records)


def test_resolve_theme_non_table_ui_warns(monkeypatch, caplog) -> None:
    """A non-table `[ui]` value warns and falls back to the default theme."""
    import logging

    from deepagents_code import theme

    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(opt, toml_data={"ui": "oops"})
    assert (value, source) == (theme.DEFAULT_THEME, "default")
    assert any("should be a table" in r.getMessage() for r in caplog.records)


def test_resolve_theme_unknown_saved_warns(monkeypatch, caplog) -> None:
    """An unknown saved `[ui].theme` warns and falls back to the default."""
    import logging

    from deepagents_code import theme

    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "no-mapping-terminal")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            opt, toml_data={"ui": {"theme": "no-such-theme"}}
        )
    assert (value, source) == (theme.DEFAULT_THEME, "default")
    assert any("Unknown theme" in r.getMessage() for r in caplog.records)


# --- config path: existence + OSError ---------------------------------------


def test_config_paths_logs_and_reports_missing_on_oserror(monkeypatch, caplog) -> None:
    """An `OSError` from `path.stat()` is logged and reported as missing."""
    import logging
    from pathlib import Path

    from deepagents_code import model_config
    from deepagents_code.client.commands.config import _config_paths

    target = model_config.DEFAULT_CONFIG_PATH
    real_stat = Path.stat

    def fake_stat(self, *, follow_symlinks: bool = True) -> object:
        if self == target:
            msg = "boom"
            raise OSError(msg)
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fake_stat)
    # The OSError guard and its debug log now live in the shared `_paths`
    # classifier that `_config_paths` delegates to.
    with caplog.at_level(logging.DEBUG, logger="deepagents_code._paths"):
        rows = _config_paths()
    config_row = next(row for row in rows if row[0] == "config.toml")
    assert config_row[2] is False
    assert any("Could not stat" in r.getMessage() for r in caplog.records)


def test_run_path_json_reports_existence(monkeypatch, tmp_path, capsys) -> None:
    """`config path --json` reports each location's existence and path."""
    import json

    from deepagents_code import model_config

    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", cfg)
    args = argparse.Namespace(config_command="path", output_format="json")
    assert run_config_command(args) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    row = next(r for r in rows if r["label"] == "config.toml")
    assert row["exists"] is True
    assert row["path"] == str(cfg)


def test_run_config_json_reports_effective_values(capsys) -> None:
    """`config --json` reports effective values without catalog fields."""
    import json

    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "config"
    rows = payload["data"]
    assert all(
        {"key", "group", "source", "set", "redacted", "value"} <= set(r) for r in rows
    )
    assert all("type" not in r for r in rows)


def test_run_config_verbose_json_serializes_catalog(capsys) -> None:
    """`config --verbose --json` folds the catalog into each row."""
    import json

    args = argparse.Namespace(config_command=None, output_format="json", verbose=True)
    assert run_config_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "config"
    rows = payload["data"]
    assert any(
        r["key"] == "interpreter.memory_limit_mb" and r["default"] == 64 for r in rows
    )
    assert all(
        {"key", "type", "default", "redacted", "env_var", "toml_path", "cli_flag"}
        <= set(r)
        for r in rows
    )


# --- Provider/credential drift ----------------------------------------------


def test_new_provider_surfaces_after_cache_clear(monkeypatch) -> None:
    """A provider added to the registry surfaces once the option cache is cleared.

    Exercises the `cache_clear` caveat documented on `get_config_options`: the
    credential surface is regenerated from `PROVIDER_API_KEY_ENV`, so a new
    provider must produce a `credentials.<name>` option after the cache resets.
    """
    from deepagents_code import config_manifest, model_config
    from deepagents_code.configuration import service

    patched = {
        **model_config.PROVIDER_API_KEY_ENV,
        "synthetic_xyz": "SYNTHETIC_XYZ_API_KEY",
    }
    monkeypatch.setattr(model_config, "PROVIDER_API_KEY_ENV", patched)
    config_manifest.get_config_options.cache_clear()
    service._managed_table_paths.cache_clear()
    config_manifest._options_by_key.cache_clear()
    config_manifest._options_by_toml_path.cache_clear()
    try:
        opt = config_manifest.get_option("credentials.synthetic_xyz")
        assert opt is not None
        assert opt.env_var == "SYNTHETIC_XYZ_API_KEY"
        # A *_API_KEY env var is treated as secret material.
        assert opt.redacted is True
    finally:
        # Restore the cache so later tests rebuild against the real registry.
        config_manifest.get_config_options.cache_clear()
        service._managed_table_paths.cache_clear()
        config_manifest._options_by_key.cache_clear()
        config_manifest._options_by_toml_path.cache_clear()


def test_provider_dependency_metadata_is_exhaustive() -> None:
    """Provider dependency metadata must cover auth and install surfaces."""
    from deepagents_code.config_manifest import _PROVIDER_DEPENDENCIES
    from deepagents_code.extras_info import MODEL_PROVIDER_EXTRAS

    assert set(PROVIDER_API_KEY_ENV) <= set(_PROVIDER_DEPENDENCIES), (
        "_PROVIDER_DEPENDENCIES must include every provider credential so config "
        "availability hints stay complete"
    )
    assert {extra for _module, extra in _PROVIDER_DEPENDENCIES.values()} == set(
        MODEL_PROVIDER_EXTRAS
    ), (
        "_PROVIDER_DEPENDENCIES must include every model-provider extra so the "
        "model selector can surface install-required recommended models"
    )


# --- Auto classifier timeout -----------------------------------------------


def test_auto_classifier_timeout_option_metadata() -> None:
    """The classifier-timeout option exposes the env/TOML override surface."""
    opt = get_option("models.auto_classifier_timeout")
    assert opt is not None
    assert opt.kind is OptionKind.FLOAT
    assert opt.env_var == _env_vars.AUTO_CLASSIFIER_TIMEOUT
    assert opt.toml_keys == ("models", "auto_classifier_timeout")
    assert "models.auto_classifier_timeout" in option_keys()


def test_auto_classifier_timeout_default_matches_middleware_default() -> None:
    """The middleware constant and the manifest default stay single-sourced."""
    from deepagents_code.auto_mode import _CLASSIFIER_TIMEOUT_SECONDS
    from deepagents_code.config_manifest import (
        AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT,
        resolve_auto_classifier_timeout,
    )

    assert _CLASSIFIER_TIMEOUT_SECONDS == AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT
    assert (
        resolve_auto_classifier_timeout(toml_data={})
        == AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT
    )


def test_resolve_auto_classifier_timeout_env_wins(monkeypatch) -> None:
    """A valid env var overrides both TOML and the default."""
    from deepagents_code.config_manifest import resolve_auto_classifier_timeout

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "45")
    resolved = resolve_auto_classifier_timeout(
        toml_data={"models": {"auto_classifier_timeout": 30}}
    )
    assert resolved == pytest.approx(45.0)


def test_resolve_auto_classifier_timeout_toml_when_env_unset(monkeypatch) -> None:
    """`config.toml` is used when no env var is set, including bare integers."""
    from deepagents_code.config_manifest import resolve_auto_classifier_timeout

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, raising=False)
    resolved = resolve_auto_classifier_timeout(
        toml_data={"models": {"auto_classifier_timeout": 30}}
    )
    assert resolved == pytest.approx(30.0)


@pytest.mark.parametrize(
    "raw", ["0", "-5", "0.5", "100000", "inf", "-inf", "nan", "soon"]
)
def test_resolve_auto_classifier_timeout_out_of_range_falls_back(
    monkeypatch, raw
) -> None:
    """Non-positive, unbounded, non-finite, or malformed values fall back."""
    from deepagents_code.config_manifest import (
        AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT,
        resolve_auto_classifier_timeout,
    )

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, raw)
    assert (
        resolve_auto_classifier_timeout(toml_data={})
        == AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT
    )


def test_resolve_auto_classifier_timeout_invalid_env_falls_through_to_toml(
    monkeypatch,
) -> None:
    """An out-of-range env value does not mask a valid TOML override."""
    from deepagents_code.config_manifest import resolve_auto_classifier_timeout

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "0")
    resolved = resolve_auto_classifier_timeout(
        toml_data={"models": {"auto_classifier_timeout": 30}}
    )
    assert resolved == pytest.approx(30.0)


def test_resolve_auto_classifier_timeout_invalid_managed_and_env_reach_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected managed and env values do not mask a valid user value."""
    from deepagents_code.config_manifest import resolve_auto_classifier_timeout

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "0")
    resolved = resolve_auto_classifier_timeout(
        toml_data={"models": {"auto_classifier_timeout": 30}},
        managed_toml_data={"models": {"auto_classifier_timeout": 500}},
    )
    assert resolved == pytest.approx(30.0)


def test_resolve_auto_classifier_timeout_invalid_env_leaves_env_intact(
    monkeypatch,
) -> None:
    """Fall-through resolution must restore the rejected env var afterward."""
    from deepagents_code.config_manifest import resolve_auto_classifier_timeout

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "0")
    resolve_auto_classifier_timeout(
        toml_data={"models": {"auto_classifier_timeout": 30}}
    )
    assert os.environ[_env_vars.AUTO_CLASSIFIER_TIMEOUT] == "0"


def test_resolve_auto_classifier_timeout_invalid_env_warns(monkeypatch, caplog) -> None:
    """The rejection warning is a user's only signal; it must not be dropped.

    Nothing else surfaces a discarded timeout, so a refactor that removed the
    warning (or lowered it to `debug`) would make the fall-through silent.
    """
    import logging

    from deepagents_code.config_manifest import resolve_auto_classifier_timeout

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "0")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        resolve_auto_classifier_timeout(
            toml_data={"models": {"auto_classifier_timeout": 30}}
        )
    assert any(
        "auto_classifier_timeout" in r.getMessage()
        and _env_vars.AUTO_CLASSIFIER_TIMEOUT in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize("raw", [0, 0.5, 100000, True, "30"])
def test_resolve_auto_classifier_timeout_rejects_unusable_toml(
    monkeypatch, caplog, raw
) -> None:
    """An unusable `config.toml` value falls back to the default with a warning.

    TOML is the only non-env surface for this option (no CLI flag, no `/auto`
    subcommand), and the threat model claims it cannot remove the deadline — so
    a `= 0` that reached the middleware would deny every gated batch.
    """
    import logging

    from deepagents_code.config_manifest import (
        AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT,
        resolve_auto_classifier_timeout,
    )

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, raising=False)
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        resolved = resolve_auto_classifier_timeout(
            toml_data={"models": {"auto_classifier_timeout": raw}}
        )
    assert resolved == AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT
    assert any("auto_classifier_timeout" in r.getMessage() for r in caplog.records)


def test_resolve_auto_classifier_timeout_malformed_env_falls_through_to_toml(
    monkeypatch,
) -> None:
    """A malformed env value takes the other fall-through path than an out-of-range one.

    `"soon"` is dropped during env coercion, so the env layer never reports
    itself as the source and no `os.environ` pop happens; `"0"` coerces fine and
    is rejected later, taking the pop-and-re-resolve branch. Both must land on
    the valid TOML value.
    """
    from deepagents_code.config_manifest import resolve_auto_classifier_timeout

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "soon")
    resolved = resolve_auto_classifier_timeout(
        toml_data={"models": {"auto_classifier_timeout": 30}}
    )
    assert resolved == pytest.approx(30.0)
    assert os.environ[_env_vars.AUTO_CLASSIFIER_TIMEOUT] == "soon"


def test_resolve_auto_classifier_timeout_guards_unpoppable_env_source(
    monkeypatch,
) -> None:
    """A source label that is not an environ key must not recurse forever.

    Termination of the fall-through branch depends on the reconstructed env name
    actually being the key the resolver read. If it is not, the pop is a
    no-op and re-resolving would see the same rejected value again.
    """
    from deepagents_code import config_manifest
    from deepagents_code.config_manifest import (
        AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT,
        resolve_auto_classifier_timeout,
    )
    from deepagents_code.configuration.resolver import ENVIRONMENT_RANK, ResolvedValue
    from deepagents_code.configuration.types import ProviderHealth, ProviderStatus

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, raising=False)
    monkeypatch.setattr(
        config_manifest,
        "_resolve_option",
        lambda *_args, **_kwargs: ResolvedValue(
            0.0,
            {ENVIRONMENT_RANK: frozenset({()})},
            {},
            {
                ENVIRONMENT_RANK: ProviderStatus(
                    "env (NOT_AN_ENVIRON_KEY)", None, ProviderHealth.OK
                )
            },
            selected_ranks=(ENVIRONMENT_RANK,),
        ),
    )
    assert (
        resolve_auto_classifier_timeout(toml_data={})
        == AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT
    )


def test_resolve_auto_classifier_timeout_accepts_floor_and_ceiling(
    monkeypatch,
) -> None:
    """The inclusive floor and ceiling are accepted verbatim."""
    from deepagents_code.config_manifest import (
        AUTO_CLASSIFIER_TIMEOUT_CEILING,
        AUTO_CLASSIFIER_TIMEOUT_FLOOR,
        resolve_auto_classifier_timeout,
    )

    monkeypatch.setenv(
        _env_vars.AUTO_CLASSIFIER_TIMEOUT, str(AUTO_CLASSIFIER_TIMEOUT_FLOOR)
    )
    assert (
        resolve_auto_classifier_timeout(toml_data={}) == AUTO_CLASSIFIER_TIMEOUT_FLOOR
    )
    monkeypatch.setenv(
        _env_vars.AUTO_CLASSIFIER_TIMEOUT, str(AUTO_CLASSIFIER_TIMEOUT_CEILING)
    )
    assert (
        resolve_auto_classifier_timeout(toml_data={}) == AUTO_CLASSIFIER_TIMEOUT_CEILING
    )


def test_resolve_auto_classifier_timeout_with_source_reports_effective_layer(
    monkeypatch,
) -> None:
    """The source variant credits the layer the runtime actually enforces."""
    from deepagents_code.config_manifest import (
        resolve_auto_classifier_timeout_with_source,
    )

    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "0")
    value, source = resolve_auto_classifier_timeout_with_source(
        toml_data={"models": {"auto_classifier_timeout": 30}}
    )
    assert value == pytest.approx(30.0)
    assert source == "config.toml"


def test_config_resolve_reports_effective_auto_classifier_timeout(
    monkeypatch,
) -> None:
    """`config get` must not credit an out-of-range env value the runtime drops."""
    option = get_option("models.auto_classifier_timeout")
    assert option is not None
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "0")
    is_set, source, value = _resolve(
        option, {"models": {"auto_classifier_timeout": 30}}, managed_toml_data={}
    )
    assert is_set is True
    assert source == "config.toml"
    assert value == pytest.approx(30.0)


def test_config_resolve_discards_out_of_range_toml_auto_classifier_timeout(
    monkeypatch,
) -> None:
    """`config get` reports the default when TOML provides an unusable timeout."""
    from deepagents_code.config_manifest import AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT

    option = get_option("models.auto_classifier_timeout")
    assert option is not None
    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, raising=False)
    is_set, source, value = _resolve(
        option, {"models": {"auto_classifier_timeout": 500}}, managed_toml_data={}
    )
    assert is_set is False
    assert source == "default"
    assert value == AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT


@pytest.mark.parametrize(
    ("toml_data", "expected"),
    [
        # A file with no `mode` key still reports the mode the launch will use.
        ({"startup": {"recent": "auto"}}, (True, "config.toml", "auto")),
        ({}, (False, "default", DEFAULT_STARTUP_MODE)),
        (
            {"startup": {"mode": "hands-off", "recent": "auto"}},
            (False, "default", DEFAULT_STARTUP_MODE),
        ),
    ],
    ids=["recent-only", "nothing-configured", "invalid-explicit-mode"],
)
def test_config_resolve_reports_effective_startup_mode(
    monkeypatch: pytest.MonkeyPatch,
    toml_data: dict,
    expected: tuple[bool, str, str],
) -> None:
    """`config get startup.mode` must route through the recent-aware resolver."""
    from deepagents_code import approval_mode

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: True)
    option = get_option("startup.mode")
    assert option is not None

    assert _resolve(option, toml_data, managed_toml_data={}) == expected


def test_config_resolve_reports_valid_env_auto_classifier_timeout(
    monkeypatch,
) -> None:
    """An in-range env override still wins in the config reporting path."""
    option = get_option("models.auto_classifier_timeout")
    assert option is not None
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, "45")
    is_set, source, value = _resolve(
        option, {"models": {"auto_classifier_timeout": 30}}, managed_toml_data={}
    )
    assert is_set is True
    assert source == f"env ({_env_vars.AUTO_CLASSIFIER_TIMEOUT})"
    assert value == pytest.approx(45.0)


def test_config_resolve_reports_default_auto_classifier_timeout_as_unset(
    monkeypatch,
) -> None:
    """A default-resolved timeout reports the default source, not a set value."""
    option = get_option("models.auto_classifier_timeout")
    assert option is not None
    from deepagents_code.config_manifest import AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, raising=False)
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is False
    assert source == "default"
    assert value == AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT


# --- Recursion limit -------------------------------------------------------


def test_recursion_limit_option_metadata() -> None:
    """The recursion-limit option exposes the env/TOML/CLI override surface."""
    opt = get_option("runtime.recursion_limit")
    assert opt is not None
    assert opt.kind is OptionKind.INT
    assert opt.env_var == _env_vars.RECURSION_LIMIT
    assert opt.toml_keys == ("runtime", "recursion_limit")
    assert opt.cli_flag == "--recursion-limit"
    assert "runtime.recursion_limit" in option_keys()


def test_resolve_recursion_limit_default() -> None:
    """With no override, the resolver returns the manifest default."""
    from deepagents_code.config_manifest import (
        RECURSION_LIMIT_DEFAULT,
        resolve_recursion_limit,
    )

    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_DEFAULT


def test_resolve_recursion_limit_env_wins(monkeypatch) -> None:
    """A valid env var overrides both TOML and the default."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "3000")
    assert (
        resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
        == 3000
    )


def test_resolve_recursion_limit_toml_when_env_unset(monkeypatch) -> None:
    """`config.toml` is used when no env var is set."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.delenv(_env_vars.RECURSION_LIMIT, raising=False)
    assert (
        resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
        == 1500
    )


@pytest.mark.parametrize("raw", ["0", "-5", "10", "999999999", "notanint"])
def test_resolve_recursion_limit_out_of_range_falls_back(monkeypatch, raw) -> None:
    """Non-positive, sub-floor, above-ceiling, or malformed values fall back."""
    from deepagents_code.config_manifest import (
        RECURSION_LIMIT_DEFAULT,
        resolve_recursion_limit,
    )

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, raw)
    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_DEFAULT


def test_resolve_recursion_limit_invalid_env_falls_through_to_toml(
    monkeypatch,
) -> None:
    """An out-of-range env value does not mask a valid TOML override."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "10")
    assert (
        resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
        == 1500
    )


def test_resolve_recursion_limit_invalid_managed_and_env_reach_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected managed and env limits do not mask a valid user limit."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "10")
    assert (
        resolve_recursion_limit(
            toml_data={"runtime": {"recursion_limit": 1500}},
            managed_toml_data={"runtime": {"recursion_limit": 500_000}},
        )
        == 1500
    )


def test_resolve_recursion_limit_invalid_env_leaves_env_intact(monkeypatch) -> None:
    """Fall-through resolution must restore the rejected env var afterward."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "10")
    resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
    assert os.environ[_env_vars.RECURSION_LIMIT] == "10"


def test_resolve_recursion_limit_accepts_floor_and_ceiling(monkeypatch) -> None:
    """The inclusive floor and ceiling are accepted verbatim."""
    from deepagents_code.config_manifest import (
        RECURSION_LIMIT_CEILING,
        RECURSION_LIMIT_FLOOR,
        resolve_recursion_limit,
    )

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, str(RECURSION_LIMIT_FLOOR))
    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_FLOOR
    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, str(RECURSION_LIMIT_CEILING))
    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_CEILING


def test_recursion_limit_managed_fallthrough_keeps_shared_user_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A rejected managed limit falls through to the startup-generation file.

    The managed fall-through used to re-parse `config.toml` off disk, so a
    hand edit after startup changed a later-built agent's limit without
    `/reload` while every other reader stayed on the old generation.
    """
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    config = tmp_path / "config.toml"
    config.write_text("[runtime]\nrecursion_limit = 2000\n", encoding="utf-8")
    managed = tmp_path / "managed_config.toml"
    managed.write_text("[runtime]\nrecursion_limit = 99999999\n", encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    monkeypatch.delenv(_env_vars.RECURSION_LIMIT, raising=False)

    # Prime the shared resolver on the startup generation.
    assert resolve_recursion_limit() == 2000

    # A hand edit after startup is not a reload: the managed fall-through must
    # still read the generation the rest of the process observes.
    config.write_text("[runtime]\nrecursion_limit = 3000\n", encoding="utf-8")
    assert resolve_recursion_limit() == 2000


def test_auto_classifier_timeout_managed_fallthrough_keeps_shared_user_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The timeout's managed fall-through pins the same user generation."""
    from deepagents_code.config_manifest import (
        resolve_auto_classifier_timeout_with_source,
    )
    from deepagents_code.configuration import service

    config = tmp_path / "config.toml"
    config.write_text("[models]\nauto_classifier_timeout = 30\n", encoding="utf-8")
    managed = tmp_path / "managed_config.toml"
    managed.write_text(
        "[models]\nauto_classifier_timeout = 99999999\n", encoding="utf-8"
    )
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_TIMEOUT, raising=False)

    value, source = resolve_auto_classifier_timeout_with_source()
    assert value == pytest.approx(30.0)
    assert source == "config.toml"

    config.write_text("[models]\nauto_classifier_timeout = 45\n", encoding="utf-8")
    value, source = resolve_auto_classifier_timeout_with_source()
    assert value == pytest.approx(30.0)
    assert source == "config.toml"


def test_auto_classifier_model_managed_fallthrough_keeps_shared_user_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The model resolver's managed-blank re-read pins the user generation too."""
    from deepagents_code.config_manifest import (
        resolve_auto_classifier_model_with_source,
    )
    from deepagents_code.configuration import service

    config = tmp_path / "config.toml"
    config.write_text('[models]\nauto_classifier = "openai:gpt-5"\n', encoding="utf-8")
    # Managed participates (deciding `runtime.recursion_limit`) but leaves the
    # classifier key unset, so the resolver re-reads without the managed tier.
    managed = tmp_path / "managed_config.toml"
    managed.write_text("[runtime]\nrecursion_limit = 1500\n", encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)

    value, source = resolve_auto_classifier_model_with_source()
    assert value == "openai:gpt-5"
    assert source == "config.toml"

    config.write_text(
        '[models]\nauto_classifier = "anthropic:claude"\n', encoding="utf-8"
    )
    value, source = resolve_auto_classifier_model_with_source()
    assert value == "openai:gpt-5"
    assert source == "config.toml"


def test_delegate_static_defaults_are_parseable() -> None:
    """A delegate option's static default must satisfy its own parser.

    Delegate defaults bypass the resolver's coercion (they are returned verbatim
    on the default path), so `__post_init__` cannot type-check them. This guards
    the one class of typo it would otherwise miss (e.g. `ptc` default `'saef'`).
    """
    from deepagents_code.config import _parse_interpreter_ptc

    for opt in get_config_options():
        if opt.default is None:
            continue
        if opt.kind is OptionKind.PTC_DELEGATE:
            assert _parse_interpreter_ptc(opt.default) == opt.default


def test_load_config_toml_tolerates_an_undecodable_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A mis-encoded config falls back to defaults instead of raising.

    `UnicodeDecodeError` subclasses `ValueError`, so it is neither `OSError`
    nor `TOMLDecodeError`. Without explicit handling, a config saved as UTF-16
    raises out of every caller that reads an option.
    """
    from deepagents_code.config_manifest import load_config_toml

    config_path = tmp_path / "config.toml"
    config_path.write_bytes(b"\xff\xfe[warnings]\n")
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path)

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert load_config_toml() == {}

    assert "using defaults" in caplog.text


def test_whitespace_env_is_ignored_with_a_warning(monkeypatch, caplog) -> None:
    """A whitespace-only env value falls through and says so.

    Empty is a normal "unset" idiom, but whitespace-only is nearly always an
    accident (`export X="$UNSET "`). Discarding it without a word was the only
    unlogged rejection path in resolution, so a user could lose an
    override with no evidence anywhere.
    """
    option = get_option("models.auto_classifier")
    assert option is not None
    assert option.env_var is not None
    monkeypatch.setenv(option.env_var, "   ")

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            option, toml_data={"models": {"auto_classifier": "openai:gpt-5"}}
        )

    assert (value, source) == ("openai:gpt-5", "config.toml")
    assert any("whitespace-only" in r.getMessage() for r in caplog.records)


def test_whitespace_env_opts_out_when_empty_means_false(monkeypatch) -> None:
    """`empty_env_is_false` treats whitespace like empty, outranking config.toml.

    The asymmetry with the test above is the resolver's contract, so pin both
    sides: a blank value is an opt-out only where the option declares one.
    """
    option = get_option("display.cursor_blink")
    assert option is not None
    assert option.empty_env_is_false is True
    assert option.env_var is not None
    monkeypatch.setenv(option.env_var, "  \t ")

    assert _resolve_manifest_option(
        option, toml_data={"ui": {"cursor_blink": True}}
    ) == (
        False,
        f"env ({option.env_var})",
    )


def test_invalid_primary_env_warns_before_valid_fallback(monkeypatch, caplog) -> None:
    """Alias fall-through retains diagnostics from earlier env candidates."""
    option = get_option("tracing.langsmith_project")
    assert option is not None
    assert option.env_var is not None
    fallback = option.fallback_env_vars[0]
    monkeypatch.setenv(option.env_var, "  ")
    monkeypatch.setenv(fallback, "fallback-project")

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(option, toml_data={})

    assert (value, source) == ("fallback-project", f"env ({fallback})")
    assert [
        record.getMessage()
        for record in caplog.records
        if "whitespace-only" in record.getMessage()
    ] == [f"Ignoring {option.env_var}='  ' (whitespace-only; treated as unset)"]


def test_non_table_toml_section_is_reported_once(caplog) -> None:
    """A scalar shadowing a whole table is logged, not silently defaulted.

    `ui = "dark"` defaults every `[ui]` option at once; the pre-manifest loaders
    each warned about it, and dropping that left the user's edited value absent
    from the output with no explanation. Logged once per path per process
    because `config` resolves the whole manifest in one pass.
    """
    from deepagents_code import config_manifest

    config_manifest._warned_non_table_paths.clear()
    scrollbar = get_option("display.show_scrollbar")
    blink = get_option("display.cursor_blink")
    assert scrollbar is not None
    assert blink is not None

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(scrollbar, toml_data={"ui": "dark"}) == (
            False,
            "default",
        )
        assert _resolve_manifest_option(blink, toml_data={"ui": "dark"}) == (
            True,
            "default",
        )

    warnings = [r for r in caplog.records if "expected a table" in r.getMessage()]
    assert len(warnings) == 1
    assert "[ui]" in warnings[0].getMessage()
    config_manifest._warned_non_table_paths.clear()


def test_verbose_provenance_distinguishes_quoted_dotted_keys() -> None:
    """Display labels retain the resolver's tuple-path distinction."""
    from deepagents_code.client.commands.config import _option_provenance

    option = get_option("display.themes")
    assert option is not None

    assert _option_provenance(
        option,
        source="managed config + config.toml",
        toml_data={"themes": {"a": {"b": "user"}, "sibling": 1}},
        managed_toml_data={"themes": {"a.b": "managed"}},
    ) == {
        '"a.b"': "managed config",
        "a.b": "config.toml",
        "sibling": "config.toml",
    }


def test_blank_env_auto_classifier_reports_a_problem(
    monkeypatch, tmp_path: Path
) -> None:
    """A blank env classifier must be described, not just logged.

    The blank value reverts authorization review to the main agent model — the
    agent grading its own actions. A `logger.warning` lands in the debug log,
    which is not a surface the user reads, so the caller needs the description.
    """
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "openai:gpt-5"\n', encoding="utf-8"
    )
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "   ")

    spec, problem = resolve_auto_classifier_model_with_problem()

    assert spec is None
    assert problem is not None
    assert _env_vars.AUTO_CLASSIFIER_MODEL in problem
    # The message must name the value it overrode; without it the user checks
    # config.toml, still sees their setting, and learns nothing.
    assert "openai:gpt-5" in problem


@pytest.mark.parametrize("blank", ["", "   "])
def test_config_surface_agrees_with_runtime_on_blank_env_classifier(
    blank: str, monkeypatch, tmp_path: Path
) -> None:
    """`config` must not credit a classifier the runtime refuses to use.

    A blank env var vetoes `config.toml` for this option only, so resolving it
    with the generic scalar path would report the config.toml model while the
    runtime inherits the main agent model.
    """
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    toml_data = {"models": {"auto_classifier": "openai:gpt-5"}}
    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "openai:gpt-5"\n', encoding="utf-8"
    )
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, blank)

    option = get_option("models.auto_classifier")
    assert option is not None
    runtime_spec, _ = resolve_auto_classifier_model_with_problem()
    is_set, source, displayed = _resolve(
        option, toml_data=toml_data, managed_toml_data={}
    )

    assert runtime_spec is None
    assert displayed == runtime_spec
    assert source == f"env ({_env_vars.AUTO_CLASSIFIER_MODEL})"
    assert is_set is True
    assert _display_value(option, is_set=is_set, value=displayed) == "(unset)"


def test_usable_env_classifier_is_still_reported(monkeypatch) -> None:
    """The veto must not swallow a real env value."""
    from deepagents_code import config_manifest

    toml_data = {"models": {"auto_classifier": "openai:gpt-5"}}
    monkeypatch.setattr(config_manifest, "load_config_toml", lambda: toml_data)
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "anthropic:claude-haiku-4-5")

    option = get_option("models.auto_classifier")
    assert option is not None
    is_set, source, value = _resolve(option, toml_data=toml_data, managed_toml_data={})

    assert (is_set, value) == (True, "anthropic:claude-haiku-4-5")
    assert source == f"env ({_env_vars.AUTO_CLASSIFIER_MODEL})"


def test_config_text_output_redacts_credential_bearing_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The human-readable surfaces must not print a credential-bearing table.

    The JSON path redacts through `_config_json_row`; the table, `--verbose`,
    and `config get` text paths go through `_display_value` instead, so they
    need their own guard against a renderer that forgets.
    """
    from deepagents_code import model_config

    config = tmp_path / "config.toml"
    config.write_text(
        "[async_subagents.researcher]\n"
        'description = "Research agent"\n'
        'graph_id = "agent"\n'
        "headers = { Authorization = 'Bearer sk-secret' }\n"
        "[models.providers.acme]\n"
        'class_path = "acme.Chat:AcmeChat"\n'
        'api_key = "sk-secret"\n'
        "[sandboxes.providers.acme]\n"
        'class_path = "acme.Sandbox:AcmeSandbox"\n'
        "params = { token = 'sk-secret' }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config)

    for args in (
        argparse.Namespace(config_command=None, output_format="text", verbose=False),
        argparse.Namespace(config_command=None, output_format="text", verbose=True),
        argparse.Namespace(
            config_command="get",
            key="agents.async_subagents",
            output_format="text",
            verbose=False,
        ),
    ):
        assert run_config_command(args) == 0
        out = capsys.readouterr().out
        assert "sk-secret" not in out
        assert "Authorization" not in out
    assert "configured" in out


def test_empty_redacted_table_reads_as_unset() -> None:
    """A present-but-empty table has nothing configured, so say so."""
    option = get_option("agents.async_subagents")
    assert option is not None
    assert option.redacted is True
    assert _display_value(option, is_set=True, value={}) == "(unset)"
    assert _display_value(option, is_set=True, value={"a": {}}) == "configured"


def test_cursor_style_predicate_accepts_exactly_the_valid_styles() -> None:
    """`is_cursor_style` must admit every valid style and nothing else.

    Renamed from `test_invalid_cursor_style_falls_back_to_the_default`, which
    asserted only the predicate and the default's membership -- it never called
    the reader whose fallback the name claimed to cover, so reverting that
    fallback to the original unchecked `cast` left it green. See
    `test_invalid_cursor_style_reaches_the_readers_fallback` for that.
    """
    from deepagents_code.config_manifest import (
        CURSOR_STYLE_DEFAULT,
        VALID_CURSOR_STYLES,
        is_cursor_style,
    )

    assert not is_cursor_style("not-a-cursor")
    assert not is_cursor_style(None)
    assert not is_cursor_style(3)
    assert CURSOR_STYLE_DEFAULT in VALID_CURSOR_STYLES
    for style in VALID_CURSOR_STYLES:
        assert is_cursor_style(style)


def test_invalid_cursor_style_never_reaches_textual(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The reader must return a real style, and say what it rejected.

    Drives `_load_cursor_style_preference`, which the predicate test above
    does not. Note where the rejection actually happens: coercion rejects the
    string at the provider, so the reader receives the manifest default and its
    own `is_cursor_style` guard is defense in depth rather than the branch that
    fires here. Reverting that guard to an unchecked `cast` therefore still
    passes this test -- it is unreachable through a TOML string, and only a
    value that bypassed coercion would reach it.
    """
    import logging

    from deepagents_code.app import _load_cursor_style_preference
    from deepagents_code.config_manifest import (
        CURSOR_STYLE_DEFAULT,
        VALID_CURSOR_STYLES,
    )

    (tmp_path / "config.toml").write_text(
        '[ui]\ncursor_style = "not-a-cursor"\n', encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        style = _load_cursor_style_preference()

    assert style == CURSOR_STYLE_DEFAULT
    assert style in VALID_CURSOR_STYLES
    assert "cursor_style='not-a-cursor'" in caplog.text
