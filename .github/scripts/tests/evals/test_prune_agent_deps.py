"""Tests for the agent-dep pruner (`.github/scripts/evals/prune_agent_deps.py`).

Mirrors `test_shard_matrix.py`: import-by-path, stdlib + pytest only, so it runs
under CI's `pytest .github/scripts/tests` (see `.github/workflows/ci.yml`).

The committed Harbor `langgraph.json` is the source of truth for the multi-provider
dependency list. These tests load that file rather than keeping a hand-copied
fixture that can drift when pins change.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PRUNE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "evals" / "prune_agent_deps.py"


def _load_prune_script() -> ModuleType:
    """Load `.github/scripts/evals/prune_agent_deps.py` as a module.

    The script lives outside any importable package, so import-by-path is the
    only way to exercise its internals from a test.
    """
    spec = importlib.util.spec_from_file_location("gha_prune_agent_deps", PRUNE_SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {PRUNE_SCRIPT}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prune = _load_prune_script()


# The real config the CI step rewrites (see harbor.yml). Unit tests that need a
# full multi-provider dependency list read it directly so pin bumps only touch
# the committed file.
REAL_CONFIG = (
    REPO_ROOT
    / "libs"
    / "evals"
    / "deepagents_harbor"
    / "langgraph_project"
    / "langgraph.json"
)

# `langchain`-family deps in the real config that are deliberately NOT provider
# integrations (core + MCP adapters), so they belong in neither
# PROVIDER_TO_PACKAGE nor the pruned set. The reverse-drift guard allowlists
# them; any *other* langchain package must be a mapped provider.
NON_PROVIDER_LANGCHAIN_PACKAGES: frozenset[str] = frozenset(
    {"langchain", "langchain-mcp-adapters"}
)


def _real_dependencies() -> list[str]:
    """Return the `dependencies` array from the committed langgraph.json."""
    return json.loads(REAL_CONFIG.read_text())["dependencies"]


def _non_provider_dependencies(deps: list[str] | None = None) -> list[str]:
    """Return deps that pruning must never drop (path/core/infra packages)."""
    source = _real_dependencies() if deps is None else deps
    return [
        dep
        for dep in source
        if prune.dependency_package(dep) not in prune.PRUNABLE_PACKAGES
    ]


def _dep_for_package(package: str, deps: list[str] | None = None) -> str:
    """Return the requirement string for `package` from deps (or the real config)."""
    source = _real_dependencies() if deps is None else deps
    for dep in source:
        if prune.dependency_package(dep) == package:
            return dep
    msg = f"Package {package!r} not found in dependencies"
    raise AssertionError(msg)


@pytest.mark.parametrize("provider", sorted(prune.PROVIDER_TO_PACKAGE))
def test_keeps_exactly_one_provider(provider: str) -> None:
    """Every provider prunes to its own package plus all non-provider deps."""
    deps = _real_dependencies()
    kept = prune.prune_dependencies(deps, provider)
    kept_provider_pkgs = [
        prune.dependency_package(d)
        for d in kept
        if prune.dependency_package(d) in prune.PRUNABLE_PACKAGES
    ]
    assert kept_provider_pkgs == [prune.PROVIDER_TO_PACKAGE[provider]]
    # Non-provider deps are untouched.
    for dep in _non_provider_dependencies(deps):
        assert dep in kept


def test_order_is_preserved() -> None:
    """Pruning filters in place without reordering the surviving deps."""
    deps = _real_dependencies()
    kept = prune.prune_dependencies(deps, "fireworks")
    assert kept == [d for d in deps if d in kept]


def test_openai_openrouter_do_not_collide() -> None:
    """`langchain-openai` and `langchain-openrouter` share a prefix.

    Name-boundary matching must keep only the selected one — the whole reason
    for parsing the bare package name rather than prefix/substring matching on
    the shared `langchain-open` stem.
    """
    deps = _real_dependencies()
    openai_dep = _dep_for_package("langchain-openai", deps)
    openrouter_dep = _dep_for_package("langchain-openrouter", deps)

    openai_kept = prune.prune_dependencies(deps, "openai")
    assert openai_dep in openai_kept
    assert openrouter_dep not in openai_kept

    openrouter_kept = prune.prune_dependencies(deps, "openrouter")
    assert openrouter_dep in openrouter_kept
    assert openai_dep not in openrouter_kept


def test_nvidia_package_name_differs_from_prefix() -> None:
    """`nvidia` maps to `langchain-nvidia-ai-endpoints`, not `langchain-nvidia`."""
    deps = _real_dependencies()
    nvidia_dep = _dep_for_package("langchain-nvidia-ai-endpoints", deps)
    kept = prune.prune_dependencies(deps, "nvidia")
    assert nvidia_dep in kept


def test_unknown_provider_raises_key_error() -> None:
    """A provider absent from the map is a programming error, not a silent no-op."""
    with pytest.raises(KeyError):
        prune.prune_dependencies(_real_dependencies(), "mistral")


def test_missing_provider_package_raises_value_error() -> None:
    """Drift guard: the selected provider's package must be present in deps."""
    deps_without_fireworks = [
        d for d in _real_dependencies() if "fireworks" not in d
    ]
    with pytest.raises(ValueError, match="fireworks"):
        prune.prune_dependencies(deps_without_fireworks, "fireworks")


@pytest.mark.parametrize(
    ("dep", "expected"),
    [
        ("langchain-openai>=1.3.0,<1.4.0", "langchain-openai"),
        ("langchain-openrouter>=0.2.3,<0.3.0", "langchain-openrouter"),
        ("langchain-nvidia-ai-endpoints>=1.4.1,<1.5.0", "langchain-nvidia-ai-endpoints"),
        ("langchain>=1.3.9,<2.0.0", "langchain"),
        ("./.local_deps/deepagents", "./.local_deps/deepagents"),
        ("pkg[extra]>=1.0", "pkg"),
        ("pkg ; python_version < '3.13'", "pkg"),
        ("bare-package", "bare-package"),
        # Every other operator in _NAME_DELIMITERS as the leading specifier.
        ("langchain-openai==1.3.0", "langchain-openai"),
        ("pkg~=1.0", "pkg"),
        ("pkg!=1.0", "pkg"),
        ("pkg<2.0", "pkg"),
        ("  spaced-pkg >=1.0  ", "spaced-pkg"),
    ],
)
def test_dependency_package_parsing(dep: str, expected: str) -> None:
    """Package-name extraction handles specifiers, extras, markers, path deps."""
    assert prune.dependency_package(dep) == expected


def _write_config(path: Path, deps: list[str]) -> None:
    path.write_text(json.dumps({"dependencies": deps, "graphs": {"g": "x:y"}}))


def test_main_rewrites_file_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`main` prunes the file and leaves unrelated keys (`graphs`) intact."""
    config_path = tmp_path / "langgraph.json"
    _write_config(config_path, _real_dependencies())
    monkeypatch.setenv("HARBOR_MODEL", "fireworks:accounts/fireworks/models/glm-5p2")
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), str(config_path)])

    prune.main()

    raw = config_path.read_text()
    written = json.loads(raw)
    provider_pkgs = [
        prune.dependency_package(d)
        for d in written["dependencies"]
        if prune.dependency_package(d) in prune.PRUNABLE_PACKAGES
    ]
    assert provider_pkgs == ["langchain-fireworks"]
    assert written["graphs"] == {"g": "x:y"}
    # The committed file is 2-space-indented with a trailing newline; main()
    # writes json.dump(indent=2) + "\n". Assert the format at the byte level —
    # json.loads round-trips would hide a reformat that reads back identically.
    assert raw.endswith("\n")
    assert '\n  "graphs"' in raw


def test_main_unknown_provider_fails_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unmapped provider hard-fails (its plumbing isn't wired in harbor.yml).

    Failing fast beats the no-op fallback, which would retain langchain-fireworks
    — whose transitive fireworks-ai dep is a prerelease — and fail the agent-env
    install with a cryptic resolver error (no UV_PRERELEASE=allow off the
    fireworks arm). The file is left untouched because we fail before writing.
    """
    config_path = tmp_path / "langgraph.json"
    _write_config(config_path, _real_dependencies())
    original = config_path.read_text()
    monkeypatch.setenv("HARBOR_MODEL", "some-new-provider:whatever")
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), str(config_path)])

    with pytest.raises(SystemExit):
        prune.main()

    assert config_path.read_text() == original


@pytest.mark.parametrize("bad_model", ["", "no-colon-here"])
def test_main_rejects_malformed_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_model: str
) -> None:
    """A missing or colon-less HARBOR_MODEL fails loudly."""
    config_path = tmp_path / "langgraph.json"
    _write_config(config_path, _real_dependencies())
    monkeypatch.setenv("HARBOR_MODEL", bad_model)
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), str(config_path)])

    with pytest.raises(SystemExit):
        prune.main()


def test_main_handles_multi_colon_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`provider:model:tag` specs split on the first colon only.

    Several matrix models carry a trailing tag (e.g. `ollama:glm-5.2:cloud`); the
    provider must still resolve via `split(':', 1)`.
    """
    config_path = tmp_path / "langgraph.json"
    _write_config(config_path, _real_dependencies())
    monkeypatch.setenv("HARBOR_MODEL", "ollama:glm-5.2:cloud")
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), str(config_path)])

    prune.main()

    written = json.loads(config_path.read_text())
    provider_pkgs = [
        prune.dependency_package(d)
        for d in written["dependencies"]
        if prune.dependency_package(d) in prune.PRUNABLE_PACKAGES
    ]
    assert provider_pkgs == ["langchain-ollama"]


@pytest.mark.parametrize("argv_tail", [[], ["a", "b"]])
def test_main_rejects_wrong_arg_count(
    monkeypatch: pytest.MonkeyPatch, argv_tail: list[str]
) -> None:
    """Missing or extra CLI arguments produce a usage error before any work."""
    monkeypatch.setenv("HARBOR_MODEL", "fireworks:accounts/fireworks/models/glm-5p2")
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), *argv_tail])

    with pytest.raises(SystemExit):
        prune.main()


def test_main_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator-facing summary names the kept package and removed count."""
    config_path = tmp_path / "langgraph.json"
    _write_config(config_path, _real_dependencies())
    monkeypatch.setenv("HARBOR_MODEL", "fireworks:accounts/fireworks/models/glm-5p2")
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), str(config_path)])

    prune.main()

    out = capsys.readouterr().out
    assert "langchain-fireworks" in out
    # Every prunable provider but the selected one is removed.
    assert f"removed {len(prune.PRUNABLE_PACKAGES) - 1}" in out


@pytest.mark.parametrize(
    "dependencies",
    [
        pytest.param([], id="empty"),
        pytest.param(
            [d for d in _real_dependencies() if "openai" in d],
            id="openai-only",
        ),
    ],
)
def test_main_annotates_drift_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependencies: list[str]
) -> None:
    """A config missing the selected provider fails with a `::error::` annotation.

    Covers both an empty `dependencies` array and one that omits the selected
    provider; either way the drift guard fires, the message is annotated (not a
    raw traceback), and the file is left untouched.
    """
    config_path = tmp_path / "langgraph.json"
    _write_config(config_path, dependencies)
    original = config_path.read_text()
    monkeypatch.setenv("HARBOR_MODEL", "fireworks:accounts/fireworks/models/glm-5p2")
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), str(config_path)])

    with pytest.raises(SystemExit) as excinfo:
        prune.main()

    assert "::error::" in str(excinfo.value)
    assert config_path.read_text() == original


def test_main_annotates_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid JSON fails with a `::error::` annotation rather than a traceback."""
    config_path = tmp_path / "langgraph.json"
    config_path.write_text("{not valid json")
    monkeypatch.setenv("HARBOR_MODEL", "fireworks:accounts/fireworks/models/glm-5p2")
    monkeypatch.setattr("sys.argv", [str(PRUNE_SCRIPT), str(config_path)])

    with pytest.raises(SystemExit, match="::error::"):
        prune.main()


@pytest.mark.parametrize("provider", sorted(prune.PROVIDER_TO_PACKAGE))
def test_real_config_prunes_for_every_provider(provider: str) -> None:
    """Forward drift guard: every mapped package is present in the real config.

    Enforced at PR time. Without it, renaming or dropping a provider package in
    the committed langgraph.json only fails later — inside the one Harbor eval
    that runs that provider — as a bare `ValueError`.
    """
    prune.prune_dependencies(_real_dependencies(), provider)  # raises on drift


def test_real_config_has_no_unmapped_provider() -> None:
    """Reverse drift guard: no langchain provider package escapes the map.

    A new `langchain-<provider>` added to the committed config without a
    PROVIDER_TO_PACKAGE entry would never be pruned — it would ship to every
    job. Core/infra langchain packages are allowlisted; anything else must be a
    mapped, prunable provider.
    """
    for dep in _real_dependencies():
        pkg = prune.dependency_package(dep)
        if pkg.startswith("langchain") and pkg not in NON_PROVIDER_LANGCHAIN_PACKAGES:
            assert pkg in prune.PRUNABLE_PACKAGES, (
                f"{pkg!r} looks like a provider integration but is not in "
                "PROVIDER_TO_PACKAGE; add it (and wire harbor.yml) or add it to "
                "NON_PROVIDER_LANGCHAIN_PACKAGES if it is core/infra."
            )
