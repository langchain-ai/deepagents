"""Tests for built-wheel dependency freshness validation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from check_wheel_dep_freshness import (
    WheelMetadata,
    _escape_command,
    compare_wheel_with_pypi,
    load_sibling_python_metadata,
    main,
    parse_wheel_metadata,
    release_python_version,
    resolve_release_target,
    validate_wheel,
)
from packaging.requirements import Requirement
from packaging.version import Version


def _metadata(
    *,
    name: str = "deepagents-talon",
    version: str = "0.0.4",
    requires_python: str = ">=3.12",
    requirements: tuple[str, ...] = ("deepagents-code>=0.1.30,<1.0.0",),
    provides_extra: tuple[str, ...] = ("media",),
) -> WheelMetadata:
    return WheelMetadata(
        name=name,
        version=Version(version),
        requires_python=requires_python,
        requires_dist=tuple(Requirement(item) for item in requirements),
        provides_extra=provides_extra,
    )


def _payload(
    *releases: tuple[str, str | None],
) -> dict[str, object]:
    return {
        "releases": {
            version: [
                {
                    "requires_python": requires_python,
                    "yanked": False,
                    "packagetype": "bdist_wheel",
                }
            ]
            for version, requires_python in releases
        }
    }


def _write_wheel(
    path: Path,
    *,
    name: str = "deepagents-talon",
    version: str = "0.0.4",
    requires_python: str = ">=3.12",
    requirements: tuple[str, ...] = ("deepagents-code>=0.1.30,<1.0.0",),
    provides_extra: tuple[str, ...] = ("media",),
) -> None:
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"Requires-Python: {requires_python}",
    ]
    metadata.extend(f"Provides-Extra: {extra}" for extra in provides_extra)
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
            "\n".join(metadata) + "\n",
        )


def _write_release_config(path: Path, packages: dict[str, dict[str, str]]) -> None:
    path.write_text(json.dumps({"packages": packages}), encoding="utf-8")


def test_parse_wheel_metadata_reads_authoritative_fields(tmp_path: Path) -> None:
    wheel = tmp_path / "example.whl"
    _write_wheel(
        wheel,
        requirements=(
            "deepagents-code>=0.1.30,<1.0.0",
            'deepagents-code[media]>=0.1.30,<1.0.0; extra == "media"',
        ),
    )

    metadata = parse_wheel_metadata(wheel)

    assert metadata.name == "deepagents-talon"
    assert metadata.version == Version("0.0.4")
    assert metadata.requires_python == ">=3.12"
    assert metadata.provides_extra == ("media",)
    assert [requirement.name for requirement in metadata.requires_dist] == [
        "deepagents-code",
        "deepagents-code",
    ]


def test_compare_flags_talon_sibling_python_metadata_lag() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(),
        {"deepagents-code": _payload(("0.1.57", ">=3.11,<4.0"))},
        sibling_requires_python={"deepagents-code": ">=3.12,<4.0"},
    )

    assert len(checks) == 1
    assert not checks[0].passed
    assert (
        checks[0].message == "deepagents-code 0.1.57 (latest on PyPI) declares "
        "requires-python>=3.11,<4.0, but the current deepagents-code metadata "
        "requires >=3.12,<4.0. Constraint: <1.0.0,>=0.1.30. Release "
        "deepagents-code first, then re-run this check."
    )


def test_compare_passes_after_compatible_sibling_release() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(),
        {
            "deepagents-code": _payload(
                ("0.1.57", ">=3.11,<4.0"),
                ("0.1.58", "<4.0,>=3.12"),
            )
        },
        sibling_requires_python={"deepagents-code": ">=3.12,<4.0"},
    )

    assert checks[0].passed


def test_compare_requires_newest_eligible_sibling_metadata_to_match() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(),
        {
            "deepagents-code": _payload(
                ("0.1.56", ">=3.12,<4.0"),
                ("0.1.57", ">=3.11,<4.0"),
            )
        },
        sibling_requires_python={"deepagents-code": ">=3.12,<4.0"},
    )

    assert not checks[0].passed
    assert "deepagents-code 0.1.57 (latest on PyPI)" in checks[0].message


def test_compare_rejects_direct_url_dependency() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(requirements=("demo @ https://example.test/demo.whl",)),
        {},
    )

    assert not checks[0].passed
    assert "must resolve from PyPI" in checks[0].message
    assert "https://example.test/demo.whl" in checks[0].message


def test_compare_allows_broader_third_party_python_support() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(requirements=("third-party>=2,<3",)),
        {"third-party": _payload(("2.5", ">=3.9"))},
        sibling_requires_python={"deepagents-code": ">=3.12,<4.0"},
    )

    assert checks[0].passed


def test_compare_warns_when_third_party_caps_only_unreleased_minors(
    monkeypatch,
) -> None:
    monkeypatch.setattr("check_wheel_dep_freshness.CURRENT_CPYTHON_MINOR", 14)
    checks = compare_wheel_with_pypi(
        _metadata(
            name="deepagents-code",
            version="0.1.58",
            requires_python=">=3.12,<4.0",
            requirements=("langchain-ibm>=1.1.0,<2.0.0",),
        ),
        {"langchain-ibm": _payload(("1.1.0", ">=3.10,<3.15"))},
    )

    assert len(checks) == 1
    assert checks[0].passed
    assert checks[0].warning
    assert "langchain-ibm 1.1.0 (latest on PyPI) declares" in checks[0].message
    assert "deepagents-code 0.1.58 requires Python >=3.12,<4.0" in checks[0].message


def test_compare_fails_when_patch_exclusion_hides_unreleased_cap() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            name="deepagents-code",
            version="0.1.58",
            requires_python=">=3.12,<4.0",
            requirements=("demo>=1",),
        ),
        {"demo": _payload(("1.0", ">=3.12.1,<3.15"))},
    )

    assert len(checks) == 1
    assert not checks[0].passed
    assert not checks[0].warning


def test_compare_fails_when_third_party_excludes_existing_minor() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requires_python=">=3.12,<4.0",
            requirements=("demo>=1",),
        ),
        {"demo": _payload(("1.0", ">=3.13,<3.14"))},
    )

    assert len(checks) == 1
    assert not checks[0].passed
    assert not checks[0].warning


def test_compare_never_warns_for_sibling_python_metadata_lag() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(),
        {"deepagents-code": _payload(("0.1.57", ">=3.12,<3.15"))},
        sibling_requires_python={"deepagents-code": ">=3.12,<4.0"},
    )

    assert len(checks) == 1
    assert not checks[0].passed
    assert not checks[0].warning
    assert "Release deepagents-code first" in checks[0].message


def test_compare_checks_full_declared_python_range() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requires_python=">=3.11,<4.0",
            requirements=("demo>=1",),
        ),
        {"demo": _payload(("1.0", ">=3.11,<3.12"))},
    )

    assert not checks[0].passed
    assert "requires Python >=3.11,<4.0" in checks[0].message


def test_compare_detects_patch_level_python_exclusion() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requires_python=">=3.11,<3.12",
            requirements=("demo>=1",),
        ),
        {"demo": _payload(("1.0", ">=3.11,!=3.11.1,<3.12"))},
    )

    assert not checks[0].passed


def test_compare_combines_python_coverage_across_release_files() -> None:
    payload = {
        "releases": {
            "1.0": [
                {"requires_python": ">=3.11,<3.12"},
                {"requires_python": ">=3.12,<3.13"},
            ]
        }
    }
    checks = compare_wheel_with_pypi(
        _metadata(
            requires_python=">=3.11,<3.13",
            requirements=("demo>=1",),
        ),
        {"demo": payload},
    )

    assert checks[0].passed


def test_compare_skips_requirement_outside_supported_python_range() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requires_python=">=3.11,<4.0",
            requirements=('demo>=1; python_version < "3.10"',),
        ),
        {},
    )

    assert checks == ()


def test_compare_checks_platform_specific_requirement() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(requirements=('demo>=1; sys_platform == "win32"',)),
        {"demo": _payload(("1.0", ">=3.12"))},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_skips_impossible_platform_marker() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=('demo>=1; sys_platform == "win32" and os_name == "posix"',)
        ),
        {},
    )

    assert checks == ()


def test_compare_checks_other_platform_machine() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=('demo>=1; platform_machine not in "x86_64 arm64 AMD64"',)
        ),
        {"demo": _payload(("1.0", ">=3.12"))},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_checks_windows_arm64_marker() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=(
                'demo>=1; sys_platform == "win32" and platform_machine == "ARM64"',
            )
        ),
        {"demo": _payload(("1.0", ">=3.12"))},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_skips_non_cpython_marker() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(requirements=('demo>=1; implementation_name == "pypy"',)),
        {},
    )

    assert checks == ()


def test_compare_skips_unsupported_platform_identity() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(requirements=('demo>=1; sys_platform == "plan9"',)),
        {},
    )

    assert checks == ()


def test_compare_skips_unsupported_platform_combination() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=(
                'demo>=1; sys_platform == "win32" and platform_machine == "s390x"',
            )
        ),
        {},
    )

    assert checks == ()


def test_compare_checks_platform_release_and_version_literals_together() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=(
                (
                    'demo>=1; sys_platform == "linux" and platform_release == "10" '
                    'and platform_version == "20"'
                ),
            )
        ),
        {"demo": _payload(("1.0", ">=3.12"))},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_checks_platform_release_inequality_boundary() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=(
                'demo>=1; sys_platform == "linux" and platform_release > "10"',
            )
        ),
        {"demo": _payload(("1.0", ">=3.12"))},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_honors_python_full_version_marker() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requires_python=">=3.11,<3.12",
            requirements=('demo>=1; python_full_version >= "3.11.1"',),
        ),
        {"demo": _payload(("1.0", ">=3.11,<3.12"))},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_honors_patch_versions_above_minor_marker() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requires_python=">=3.11,<3.12",
            requirements=('demo>=1; python_full_version != "3.11"',),
        ),
        {"demo": _payload(("1.0", ">=3.11,<3.12"))},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_skips_requirement_for_undeclared_extra() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=('demo>=1; extra == "missing"',),
            provides_extra=("media",),
        ),
        {},
    )

    assert checks == ()


def test_compare_fails_when_no_version_satisfies_constraint() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(requirements=("demo>=2,<3",)),
        {"demo": _payload(("1.9", ">=3.12"), ("3.0", ">=3.12"))},
    )

    assert not checks[0].passed
    assert "No published demo release satisfies <3,>=2" in checks[0].message
    assert "latest on PyPI is 3.0" in checks[0].message


def test_compare_fails_when_dependency_rejects_release_minimum_python() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(requirements=("demo>=1",)),
        {"demo": _payload(("1.0", ">=3.13"))},
    )

    assert not checks[0].passed
    assert "deepagents-talon 0.0.4 requires Python >=3.12" in checks[0].message


def test_compare_checks_optional_metadata_without_duplicate_constraint() -> None:
    checks = compare_wheel_with_pypi(
        _metadata(
            requirements=(
                "deepagents-code>=0.1.30,<1.0.0",
                'deepagents-code[media]>=0.1.30,<1.0.0; extra == "media"',
            )
        ),
        {"deepagents-code": _payload(("0.1.58", ">=3.12,<4.0"))},
        sibling_requires_python={"deepagents-code": ">=3.12,<4.0"},
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_compare_leaves_yanked_file_handling_to_resolver_layer() -> None:
    payload = _payload(("1.0", ">=3.12"))
    releases = payload["releases"]
    assert isinstance(releases, dict)
    files = releases["1.0"]
    assert isinstance(files, list)
    files[0]["yanked"] = True

    checks = compare_wheel_with_pypi(
        _metadata(requirements=("demo>=1",)),
        {"demo": payload},
    )

    assert checks[0].passed


def test_load_sibling_python_metadata_reads_managed_manifests(tmp_path: Path) -> None:
    config = tmp_path / "release-please-config.json"
    _write_release_config(
        config,
        {
            "libs/code": {
                "package-name": "deepagents-code",
                "component": "deepagents-code",
            },
            "libs/talon": {
                "package-name": "deepagents-talon",
                "component": "deepagents-talon",
            },
        },
    )
    for package_path, name, constraint in (
        ("libs/code", "deepagents-code", ">=3.12,<4.0"),
        ("libs/talon", "deepagents-talon", ">=3.12"),
    ):
        manifest = tmp_path / package_path / "pyproject.toml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            f'[project]\nname = "{name}"\nrequires-python = "{constraint}"\n',
            encoding="utf-8",
        )

    assert load_sibling_python_metadata(
        repo_root=tmp_path,
        config_path=config,
    ) == {
        "deepagents-code": ">=3.12,<4.0",
        "deepagents-talon": ">=3.12",
    }


def test_resolve_release_target_matches_release_please_component(
    tmp_path: Path,
) -> None:
    config = tmp_path / "release-please-config.json"
    _write_release_config(
        config,
        {
            "libs/talon": {
                "package-name": "deepagents-talon",
                "component": "deepagents-talon",
            }
        },
    )

    target = resolve_release_target(
        "release(deepagents-talon): 0.0.4",
        config_path=config,
    )

    assert target.package_name == "deepagents-talon"
    assert target.package_path == "libs/talon"
    assert target.python_version == "3.12"


def test_release_python_version_matches_release_workflow() -> None:
    assert release_python_version("deepagents-code") == "3.12"
    assert release_python_version("deepagents-talon") == "3.12"
    assert release_python_version("deepagents") == "3.11"


def test_escape_command_neutralizes_workflow_command_injection() -> None:
    payload = "sibling pyproject\n::warning::spoofed\r::error::oops 100%"

    escaped = _escape_command(payload)

    assert "\n" not in escaped
    assert "\r" not in escaped
    assert escaped == "sibling pyproject%0A::warning::spoofed%0D::error::oops 100%25"


def test_validate_wheel_escapes_command_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    wheel = tmp_path / "talon.whl"
    _write_wheel(wheel, requires_python=">=3.12")
    config = tmp_path / "release-please-config.json"
    _write_release_config(
        config,
        {
            "libs/code": {
                "package-name": "deepagents-code",
                "component": "deepagents-code",
            }
        },
    )
    manifest = tmp_path / "libs/code/pyproject.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '[project]\nname = "deepagents-code"\n'
        'requires-python = ">=3.12\\n::warning::spoofed"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))

    def fetcher(_name: str) -> dict[str, object]:
        return _payload(("0.1.57", ">=3.11,<4.0"))

    assert (
        validate_wheel(
            wheel,
            repo_root=tmp_path,
            config_path=config,
            fetcher=fetcher,
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "\n::warning::" not in out
    assert "%0A::warning::spoofed" in out
    assert all(line.startswith("::error ") for line in out.splitlines())


def test_validate_wheel_fetches_each_project_once_and_fails_with_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wheel = tmp_path / "talon.whl"
    _write_wheel(
        wheel,
        requirements=(
            "deepagents-code>=0.1.30,<1.0.0",
            'deepagents-code[media]>=0.1.30,<1.0.0; extra == "media"',
        ),
    )
    config = tmp_path / "release-please-config.json"
    _write_release_config(
        config,
        {
            "libs/code": {
                "package-name": "deepagents-code",
                "component": "deepagents-code",
            }
        },
    )
    manifest = tmp_path / "libs/code/pyproject.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '[project]\nname = "deepagents-code"\nrequires-python = ">=3.12,<4.0"\n',
        encoding="utf-8",
    )
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    calls: list[str] = []

    def fetcher(name: str) -> dict[str, object]:
        calls.append(name)
        return _payload(("0.1.57", ">=3.11,<4.0"))

    assert (
        validate_wheel(
            wheel,
            repo_root=tmp_path,
            config_path=config,
            fetcher=fetcher,
        )
        == 1
    )
    assert calls == ["deepagents-code"]
    assert "Release deepagents-code first" in summary.read_text(encoding="utf-8")


def test_validate_wheel_passes_with_warning_for_unreleased_minor_cap(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("check_wheel_dep_freshness.CURRENT_CPYTHON_MINOR", 14)
    wheel = tmp_path / "code.whl"
    _write_wheel(
        wheel,
        name="deepagents-code",
        version="0.1.58",
        requires_python=">=3.12,<4.0",
        requirements=('langchain-ibm>=1.1.0,<2.0.0; extra == "ibm"',),
        provides_extra=("ibm",),
    )
    config = tmp_path / "release-please-config.json"
    _write_release_config(
        config,
        {
            "libs/code": {
                "package-name": "deepagents-code",
                "component": "deepagents-code",
            }
        },
    )
    manifest = tmp_path / "libs/code/pyproject.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '[project]\nname = "deepagents-code"\nrequires-python = ">=3.12,<4.0"\n',
        encoding="utf-8",
    )
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def fetcher(_name: str) -> dict[str, object]:
        return _payload(("1.1.0", ">=3.10,<3.15"))

    assert (
        validate_wheel(
            wheel,
            repo_root=tmp_path,
            config_path=config,
            fetcher=fetcher,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert any(line.startswith("::warning ") for line in out.splitlines())
    assert not any(line.startswith("::error ") for line in out.splitlines())
    assert "Dependency metadata warnings" in summary.read_text(encoding="utf-8")


def test_validate_wheel_does_not_fetch_inactive_requirement(tmp_path: Path) -> None:
    wheel = tmp_path / "example.whl"
    _write_wheel(
        wheel,
        requires_python=">=3.11,<4.0",
        requirements=('demo>=1; python_version < "3.10"',),
        provides_extra=(),
    )
    config = tmp_path / "release-please-config.json"
    _write_release_config(config, {})

    def fetcher(_name: str) -> dict[str, object]:
        msg = "inactive dependency should not be queried"
        raise AssertionError(msg)

    assert (
        validate_wheel(
            wheel,
            repo_root=tmp_path,
            config_path=config,
            fetcher=fetcher,
        )
        == 0
    )


def test_target_cli_writes_actions_outputs(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "release-please-config.json"
    _write_release_config(
        config,
        {
            "libs/talon": {
                "package-name": "deepagents-talon",
                "component": "deepagents-talon",
            }
        },
    )
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "target",
                "--title",
                "release(deepagents-talon): 0.0.4",
                "--config",
                str(config),
            ]
        )
        == 0
    )
    written = output.read_text(encoding="utf-8")
    assert "package_name" in written
    assert "deepagents-talon" in written
    assert "package_path" in written
    assert "libs/talon" in written
    assert "python_version" in written
    assert "3.12" in written


def test_workflow_mirrors_release_install_flags() -> None:
    workflows = Path(__file__).parents[3] / "workflows"
    freshness = (workflows / "check_dep_freshness.yml").read_text(encoding="utf-8")
    release = (workflows / "release.yml").read_text(encoding="utf-8")

    assert 'uv build --python "$PYTHON_VERSION"' in freshness
    assert "DEEPAGENTS_CODE_BUILD_COMMIT:" in freshness
    assert 'INSTALL_ARGS=(--index-url "https://pypi.org/simple"' in freshness
    assert 'if [ "$PACKAGE_NAME" = "deepagents-talon" ]; then' in freshness
    assert 'INSTALL_ARGS=(--prerelease allow "${INSTALL_ARGS[@]}")' in freshness
    assert (
        'env -u UV_PYTHON VIRTUAL_ENV="$VENV_PATH" uv pip install "${INSTALL_ARGS[@]}"'
        in freshness
    )

    assert "deepagents-code|deepagents-talon)" in release
    assert "DEEPAGENTS_CODE_BUILD_COMMIT:" in release
    assert 'if [ "$PKG_NAME" = "deepagents-talon" ]; then' in release
    assert 'INSTALL_ARGS=(--prerelease allow "${INSTALL_ARGS[@]}")' in release
    assert (
        'env -u UV_PYTHON VIRTUAL_ENV=.venv uv pip install "${INSTALL_ARGS[@]}"'
        in release
    )
 # pass if either one reverted.
    release_install = (
        'env -u UV_PYTHON VIRTUAL_ENV=.venv uv pip install "${INSTALL_ARGS[@]}"'
    )
    assert release.count(release_install) == 2
