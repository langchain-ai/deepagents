"""Build, describe, and verify branch-pinned product wheels for Harbor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from email.message import Message
from email.parser import Parser
from pathlib import Path
from typing import TypedDict, cast

import eval_agent_configs as agent_configs


class PackageRecord(TypedDict):
    """Integrity and identity metadata for one wheel."""

    distribution: str
    version: str
    filename: str
    sha256: str


class PackageManifest(TypedDict):
    """Immutable product artifact manifest."""

    schema_version: int
    version_id: str
    source_branch: str
    source_sha: str
    packages: list[PackageRecord]


PACKAGE_PATHS = {
    "deepagents": Path("libs/deepagents"),
    "deepagents-code": Path("libs/code"),
    "langchain-quickjs": Path("libs/partners/quickjs"),
}
REMOTE_PROJECT_DIR = Path("/installed-agent/langgraph-project")
SDK_DISTRIBUTION = "deepagents"
CODE_DISTRIBUTION = "deepagents-code"
QUICKJS_DISTRIBUTION = "langchain-quickjs"
SDK_DEPENDENCY = re.compile(
    r'^(?P<indent>\s*)"deepagents==(?P<version>[^"\s]+)"(?P<suffix>\s*,.*)$',
    re.MULTILINE,
)


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _wheel_core_metadata(path: Path) -> Message:
    """Read and validate a wheel's core metadata."""
    if path.name != Path(path.name).name or path.suffix != ".whl":
        raise ValueError(f"unsafe wheel filename: {path.name!r}")
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"wheel must contain exactly one METADATA file: {path}")
        return Parser().parsestr(archive.read(metadata_names[0]).decode())


def _wheel_metadata(path: Path) -> tuple[str, str]:
    """Read distribution name and version from a wheel's core metadata."""
    metadata = _wheel_core_metadata(path)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"wheel metadata is missing Name or Version: {path}")
    return _normalized_distribution(name), version


def _built_wheel(wheel_dir: Path, distribution: str) -> Path:
    """Return the one built wheel for `distribution`."""
    normalized = _normalized_distribution(distribution)
    matches = [
        path
        for path in wheel_dir.glob("*.whl")
        if _wheel_metadata(path)[0] == normalized
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one built wheel for {normalized!r}, found {len(matches)}"
        )
    return matches[0]


def _rebind_code_sdk_dependency(pyproject: Path, sdk_version: str) -> str:
    """Bind dcode's release SDK pin to the evaluated branch's SDK version.

    The monorepo packages are developed together, while `deepagents-code` keeps an
    exact pin suitable for its next independent release. Comparison builds must
    instead install the SDK code from the same branch. This function operates only
    on a temporary package copy created by `build_packages`.

    Args:
        pyproject: Temporary dcode `pyproject.toml` to update.
        sdk_version: Version recorded by the branch-built SDK wheel.

    Returns:
        The original exact SDK version pin.

    Raises:
        ValueError: If the manifest does not contain exactly one exact SDK pin.
    """
    text = pyproject.read_text()
    matches = list(SDK_DEPENDENCY.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            "deepagents-code must contain exactly one quoted "
            f"deepagents==<version> dependency, found {len(matches)}"
        )
    original = matches[0].group("version")
    rebound = SDK_DEPENDENCY.sub(
        rf'\g<indent>"deepagents=={sdk_version}"\g<suffix>', text, count=1
    )
    pyproject.write_text(rebound)
    return original


def _validate_workspace_pair(wheel_dir: Path) -> None:
    """Ensure the dcode wheel requires the SDK wheel built from the same branch."""
    sdk = _built_wheel(wheel_dir, SDK_DISTRIBUTION)
    code = _built_wheel(wheel_dir, CODE_DISTRIBUTION)
    _, sdk_version = _wheel_metadata(sdk)
    requirements = _wheel_core_metadata(code).get_all("Requires-Dist", [])
    pins = [
        match.group(1)
        for requirement in requirements
        if (match := re.fullmatch(r"deepagents\s*==\s*([^\s;]+)", requirement))
    ]
    if pins != [sdk_version]:
        raise ValueError(
            "branch-built deepagents-code wheel must require its sibling SDK "
            f"version {sdk_version!r}, found exact pins {pins!r}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    wheel_dir: Path,
    expected_packages: list[str],
    *,
    version_id: str,
    source_branch: str,
    source_sha: str,
) -> PackageManifest:
    """Validate built wheels and return their immutable manifest."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source_sha must be a full lowercase commit SHA")
    expected = [_normalized_distribution(name) for name in expected_packages]
    records: list[PackageRecord] = []
    seen: set[str] = set()
    for path in sorted(wheel_dir.glob("*.whl")):
        distribution, version = _wheel_metadata(path)
        if distribution not in expected:
            continue
        if distribution in seen:
            raise ValueError(f"multiple wheels found for {distribution!r}")
        seen.add(distribution)
        records.append(
            {
                "distribution": distribution,
                "version": version,
                "filename": path.name,
                "sha256": _sha256(path),
            }
        )
    missing = [name for name in expected if name not in seen]
    if missing:
        raise ValueError(f"missing expected product wheels: {missing}")
    records.sort(key=lambda record: expected.index(record["distribution"]))
    return {
        "schema_version": 1,
        "version_id": version_id,
        "source_branch": source_branch,
        "source_sha": source_sha,
        "packages": records,
    }


def build_packages(
    source_root: Path,
    out_dir: Path,
    packages: list[str],
    *,
    version_id: str,
    source_branch: str,
    source_sha: str,
) -> PackageManifest:
    """Build selected distributions from one immutable repository checkout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = [_normalized_distribution(package) for package in packages]
    unknown = [package for package in normalized if package not in PACKAGE_PATHS]
    if unknown:
        raise ValueError(f"unsupported product packages: {unknown}")
    if CODE_DISTRIBUTION in normalized and SDK_DISTRIBUTION not in normalized:
        raise ValueError("deepagents-code comparison builds require the sibling SDK")

    # Build the SDK first so dcode's temporary workspace pin can be bound to the
    # exact version carried by the branch-built SDK wheel.
    build_order = [
        package
        for package in (SDK_DISTRIBUTION, QUICKJS_DISTRIBUTION, CODE_DISTRIBUTION)
        if package in normalized
    ]
    for package in build_order:
        package_dir = source_root / PACKAGE_PATHS[package]
        if not (package_dir / "pyproject.toml").is_file():
            raise FileNotFoundError(f"missing product package: {package_dir}")
        env = os.environ.copy()
        build_dir = package_dir
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if package == CODE_DISTRIBUTION:
            _, sdk_version = _wheel_metadata(_built_wheel(out_dir, SDK_DISTRIBUTION))
            temporary = tempfile.TemporaryDirectory(prefix="deepagents-eval-code-")
            build_dir = Path(temporary.name) / "code"
            shutil.copytree(package_dir, build_dir)
            _rebind_code_sdk_dependency(build_dir / "pyproject.toml", sdk_version)
            env["DEEPAGENTS_CODE_BUILD_COMMIT"] = source_sha
        try:
            subprocess.run(
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--no-sources",
                    "--out-dir",
                    str(out_dir),
                    str(build_dir),
                ],
                check=True,
                env=env,
            )
        finally:
            if temporary is not None:
                temporary.cleanup()
    if CODE_DISTRIBUTION in normalized:
        _validate_workspace_pair(out_dir)
    return create_manifest(
        out_dir,
        normalized,
        version_id=version_id,
        source_branch=source_branch,
        source_sha=source_sha,
    )


def load_manifest(path: Path) -> PackageManifest:
    """Load and minimally validate a package manifest."""
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or not isinstance(raw.get("packages"), list)
    ):
        raise ValueError(f"invalid product package manifest: {path}")
    return cast(PackageManifest, raw)


def dependency_overrides(
    project_dir: Path,
    manifest_path: Path,
    runtime: str,
    *,
    expected_sha: str,
) -> list[str]:
    """Build Harbor dependency overrides for one runtime implementation."""
    manifest = load_manifest(manifest_path)
    if manifest["source_sha"] != expected_sha:
        raise ValueError(
            "product artifact SHA does not match the resolved evaluation source"
        )
    by_distribution = {
        record["distribution"]: record for record in manifest["packages"]
    }
    wheels: list[str] = []
    local_package_dir = manifest_path.parent / "packages"
    # Harbor installs overrides one at a time. The comparison builder binds the
    # dcode wheel to this SDK version, so installing the SDK first guarantees the
    # dependent resolves against branch code rather than PyPI.
    runtime_packages = agent_configs.runtime_config(runtime)["packages"]
    for distribution in runtime_packages:
        normalized = _normalized_distribution(distribution)
        record = by_distribution.get(normalized)
        if record is None:
            raise ValueError(
                f"product artifact does not contain required wheel {normalized!r}"
            )
        local_wheel = local_package_dir / record["filename"]
        if not local_wheel.is_file() or _sha256(local_wheel) != record["sha256"]:
            raise ValueError(f"wheel integrity check failed: {record['filename']}")
        wheels.append(
            str(REMOTE_PROJECT_DIR / ".branch_wheels/packages" / local_wheel.name)
        )

    config = json.loads((project_dir / "langgraph.json").read_text())
    dependencies = config.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(dependency, str) for dependency in dependencies
    ):
        raise ValueError("langgraph.json dependencies must be a list of strings")
    fixed = [
        dependency
        for dependency in dependencies
        if not dependency.startswith("./.local_deps/deepagents")
    ]
    return [*wheels, *fixed]


def _json_list(raw: str, label: str) -> list[str]:
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a JSON list of strings")
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI for GitHub package-build and Harbor override steps."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--packages-json", required=True)
    build.add_argument("--version-id", required=True)
    build.add_argument("--source-branch", required=True)
    build.add_argument("--source-sha", required=True)
    build.add_argument("--manifest", type=Path, required=True)

    overrides = subparsers.add_parser("overrides")
    overrides.add_argument("--project-dir", type=Path, required=True)
    overrides.add_argument("--manifest", type=Path, required=True)
    overrides.add_argument("--runtime", required=True)
    overrides.add_argument("--source-sha", required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_packages(
            args.source_root,
            args.out_dir,
            _json_list(args.packages_json, "--packages-json"),
            version_id=args.version_id,
            source_branch=args.source_branch,
            source_sha=args.source_sha,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        return 0
    values = dependency_overrides(
        args.project_dir,
        args.manifest,
        args.runtime,
        expected_sha=args.source_sha,
    )
    print(json.dumps(values, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
