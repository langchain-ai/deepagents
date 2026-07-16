import hashlib
import json
import os
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_product_packages as products  # noqa: E402


def _wheel(
    root: Path,
    distribution: str,
    version: str,
    content: bytes,
    *,
    requirements: tuple[str, ...] = (),
) -> Path:
    filename = f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
    path = root / filename
    metadata_dir = f"{distribution.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        metadata = (
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
        )
        archive.writestr(
            f"{metadata_dir}/METADATA",
            metadata,
        )
        archive.writestr(f"{distribution.replace('-', '_')}/data.txt", content)
    return path


def test_create_manifest_records_built_wheel_integrity(tmp_path: Path) -> None:
    sdk = _wheel(tmp_path, "deepagents", "1.2.3", b"sdk")
    code = _wheel(tmp_path, "deepagents-code", "4.5.6", b"code")
    manifest = products.create_manifest(
        tmp_path,
        ["deepagents", "deepagents-code"],
        version_id="v2",
        source_branch="feature/todos",
        source_sha="a" * 40,
    )
    assert [record["distribution"] for record in manifest["packages"]] == [
        "deepagents",
        "deepagents-code",
    ]
    assert (
        manifest["packages"][0]["sha256"]
        == hashlib.sha256(sdk.read_bytes()).hexdigest()
    )
    assert manifest["packages"][1]["filename"] == code.name


def test_create_manifest_rejects_missing_wheel(tmp_path: Path) -> None:
    _wheel(tmp_path, "deepagents", "1.2.3", b"sdk")
    with pytest.raises(ValueError, match="missing expected"):
        products.create_manifest(
            tmp_path,
            ["deepagents", "deepagents-code"],
            version_id="v1",
            source_branch="branch",
            source_sha="b" * 40,
        )


def test_dependency_overrides_select_runtime_packages_and_fixed_deps(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    wheels = project / ".branch_wheels"
    package_dir = wheels / "packages"
    package_dir.mkdir(parents=True)
    _wheel(package_dir, "deepagents", "1.2.3", b"sdk")
    _wheel(package_dir, "deepagents-code", "4.5.6", b"code")
    manifest = products.create_manifest(
        package_dir,
        ["deepagents", "deepagents-code"],
        version_id="v1",
        source_branch="branch",
        source_sha="c" * 40,
    )
    manifest_path = wheels / "package-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (project / "langgraph.json").write_text(
        json.dumps(
            {
                "dependencies": [
                    "./.local_deps/deepagents",
                    "./.local_deps/deepagents-code",
                    "langchain>=1",
                ]
            }
        )
    )

    bare = products.dependency_overrides(
        project, manifest_path, "bare", expected_sha="c" * 40
    )
    dcode = products.dependency_overrides(
        project, manifest_path, "dcode", expected_sha="c" * 40
    )
    assert bare == [
        "/installed-agent/langgraph-project/.branch_wheels/packages/"
        "deepagents-1.2.3-py3-none-any.whl",
        "langchain>=1",
    ]
    assert len(dcode) == 3
    assert dcode[0].endswith("deepagents-1.2.3-py3-none-any.whl")
    assert dcode[1].endswith("deepagents_code-4.5.6-py3-none-any.whl")


def test_build_packages_rebinds_temporary_code_pin_to_branch_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    sdk = source / "libs/deepagents"
    code = source / "libs/code"
    sdk.mkdir(parents=True)
    code.mkdir(parents=True)
    (sdk / "pyproject.toml").write_text(
        '[project]\nname="deepagents"\nversion="1.2.3"\n'
    )
    code_pyproject = code / "pyproject.toml"
    code_pyproject.write_text(
        '[project]\nname="deepagents-code"\nversion="4.5.6"\n'
        'dependencies = [\n    "deepagents==9.9.9",\n]\n'
    )
    out = tmp_path / "dist"

    def fake_build(command: list[str], *, check: bool, env: dict[str, str]) -> None:
        assert check is True
        build_dir = Path(command[-1])
        wheel_dir = Path(command[-2])
        project = tomllib.loads((build_dir / "pyproject.toml").read_text())["project"]
        if project["name"] == "deepagents":
            _wheel(wheel_dir, "deepagents", "1.2.3", b"sdk")
        else:
            assert project["dependencies"] == ["deepagents==1.2.3"]
            assert env["DEEPAGENTS_CODE_BUILD_COMMIT"] == "a" * 40
            _wheel(
                wheel_dir,
                "deepagents-code",
                "4.5.6",
                b"code",
                requirements=("deepagents==1.2.3",),
            )

    monkeypatch.setattr(products.subprocess, "run", fake_build)
    manifest = products.build_packages(
        source,
        out,
        ["deepagents", "deepagents-code"],
        version_id="v1",
        source_branch="branch",
        source_sha="a" * 40,
    )

    assert [record["version"] for record in manifest["packages"]] == [
        "1.2.3",
        "4.5.6",
    ]
    assert "deepagents==9.9.9" in code_pyproject.read_text()


def test_validate_workspace_pair_rejects_mismatched_sdk_pin(tmp_path: Path) -> None:
    _wheel(tmp_path, "deepagents", "1.2.3", b"sdk")
    _wheel(
        tmp_path,
        "deepagents-code",
        "4.5.6",
        b"code",
        requirements=("deepagents==9.9.9",),
    )
    with pytest.raises(ValueError, match="sibling SDK"):
        products._validate_workspace_pair(tmp_path)


def test_dependency_overrides_rejects_sha_or_hash_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    wheels = project / ".branch_wheels"
    package_dir = wheels / "packages"
    package_dir.mkdir(parents=True)
    wheel = _wheel(package_dir, "deepagents", "1.2.3", b"sdk")
    manifest = products.create_manifest(
        package_dir,
        ["deepagents"],
        version_id="v1",
        source_branch="branch",
        source_sha="d" * 40,
    )
    manifest_path = wheels / "package-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (project / "langgraph.json").write_text(json.dumps({"dependencies": []}))

    with pytest.raises(ValueError, match="SHA"):
        products.dependency_overrides(
            project, manifest_path, "bare", expected_sha="e" * 40
        )
    wheel.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        products.dependency_overrides(
            project, manifest_path, "bare", expected_sha="d" * 40
        )
