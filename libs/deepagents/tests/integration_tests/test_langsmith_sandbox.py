from __future__ import annotations

import os
import shlex
import time
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from langsmith.sandbox import SandboxClient

from deepagents.backends.langsmith import LangSmithSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


SNAPSHOT_NAME = "deepagents-cli"
DEFAULT_IMAGE = "python:3"
DEFAULT_FS_CAPACITY = 16 * 1024**3  # 16 GiB -- mirrors CLI _LangSmithProvider default.

ARTIFACT_FS_SNAPSHOT_NAME = "ubuntu-24-04-base"
ARTIFACT_FS_BASE_IMAGE = "ubuntu:24.04"
ARTIFACT_FS_REPO_URL = "https://github.com/langchain-ai/deepagents.git"
ARTIFACT_FS_REPO_NAME = "deepagents"
ARTIFACT_FS_MOUNT_ROOT = "/mnt"


class TestLangSmithSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("LANGSMITH_API_KEY")
        if not api_key:
            msg = "Missing secrets for LangSmith integration test: set LANGSMITH_API_KEY"
            raise RuntimeError(msg)

        client = SandboxClient(api_key=api_key)

        # Server-side filter keeps this quick even with many snapshots in the
        # workspace. name_contains is a case-insensitive substring match, so
        # match the exact name client-side.
        existing = client.list_snapshots(name_contains=SNAPSHOT_NAME)
        ready = any(snap.name == SNAPSHOT_NAME and snap.status == "ready" for snap in existing)
        if not ready:
            client.create_snapshot(
                name=SNAPSHOT_NAME,
                docker_image=DEFAULT_IMAGE,
                fs_capacity_bytes=DEFAULT_FS_CAPACITY,
            )

        ls_sandbox = client.create_sandbox(snapshot_name=SNAPSHOT_NAME)
        backend = LangSmithSandbox(sandbox=ls_sandbox)
        try:
            yield backend
        finally:
            # Never delete the snapshot -- it is shared across test runs.
            client.delete_sandbox(ls_sandbox.name)

    @pytest.mark.xfail(reason="LangSmith runs as root and ignores file permissions")
    def test_download_error_permission_denied(self, sandbox_backend: SandboxBackendProtocol) -> None:
        super().test_download_error_permission_denied(sandbox_backend)

    @pytest.mark.xfail(strict=True, reason="Upstream langchain_tests uses `in` on ReadResult dataclass")
    def test_read_basic_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        super().test_read_basic_file(sandbox_backend)

    @pytest.mark.xfail(strict=True, reason="Upstream langchain_tests uses `in` on ReadResult dataclass")
    def test_edit_single_occurrence(self, sandbox_backend: SandboxBackendProtocol) -> None:
        super().test_edit_single_occurrence(sandbox_backend)

    @pytest.mark.xfail(
        strict=True,
        reason="LangSmithSandbox.write() bypasses existence check; fix stashed",
    )
    def test_write_existing_file_fails(self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str) -> None:
        super().test_write_existing_file_fails(sandbox_backend, sandbox_test_root)

    @pytest.mark.xfail(
        strict=True,
        reason="BaseSandbox.read() via execute() hangs on large content over websocket; fix stashed",
    )
    async def test_awrite_aread_adownload_large_text_with_escaped_content(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        await super().test_awrite_aread_adownload_large_text_with_escaped_content(sandbox_backend, sandbox_test_root)


def _run(sandbox: object, command: str, *, timeout: int = 60) -> tuple[int, str, str]:
    """Run a shell command in the sandbox and return ``(exit_code, stdout, stderr)``."""
    result = sandbox.run(command, timeout=timeout)  # type: ignore[attr-defined]
    return result.exit_code, result.stdout or "", result.stderr or ""


def test_artifact_fs_mounts_deepagents_repo() -> None:
    """Mount langchain-ai/deepagents inside a LangSmith sandbox via cloudflare/artifact-fs.

    Ubuntu 24.04 ships Go 1.22, but artifact-fs requires Go 1.24+, so the toolchain is
    fetched from go.dev. FUSE in the sandbox needs ``/dev/fuse`` and ``CAP_SYS_ADMIN``;
    if either is missing, the daemon will exit and the mount-readiness loop fails fast.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY is required for the artifact-fs sandbox test")

    client = SandboxClient(api_key=api_key)

    existing = client.list_snapshots(name_contains=ARTIFACT_FS_SNAPSHOT_NAME)
    ready = any(snap.name == ARTIFACT_FS_SNAPSHOT_NAME and snap.status == "ready" for snap in existing)
    if not ready:
        client.create_snapshot(
            name=ARTIFACT_FS_SNAPSHOT_NAME,
            docker_image=ARTIFACT_FS_BASE_IMAGE,
            fs_capacity_bytes=DEFAULT_FS_CAPACITY,
        )

    # Bigger box than default: building artifact-fs pulls modernc.org/libc, which
    # the Go compiler OOMs on under the sandbox default memory.
    sandbox = client.create_sandbox(
        snapshot_name=ARTIFACT_FS_SNAPSHOT_NAME,
        vcpus=4,
        mem_bytes=8 * 1024**3,
    )
    try:
        # Resolve the latest stable Go version at runtime (artifact-fs requires 1.24+).
        install_script = """\
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends fuse3 git ca-certificates curl
rm -rf /var/lib/apt/lists/*
GO_VERSION="$(curl -fsSL https://go.dev/VERSION?m=text | head -n1)"
GO_VERSION="${GO_VERSION#go}"
GO_TARBALL="go${GO_VERSION}.linux-amd64.tar.gz"
curl -fsSL "https://go.dev/dl/${GO_TARBALL}" -o "/tmp/${GO_TARBALL}"
tar -C /usr/local -xzf "/tmp/${GO_TARBALL}"
rm "/tmp/${GO_TARBALL}"
export PATH=/usr/local/go/bin:$PATH
GOBIN=/usr/local/bin go install github.com/cloudflare/artifact-fs/cmd/artifact-fs@latest
artifact-fs --help >/dev/null
"""
        code, _stdout, stderr = _run(sandbox, install_script, timeout=900)
        assert code == 0, f"toolchain/artifact-fs install failed: {stderr}"

        repo_url = shlex.quote(ARTIFACT_FS_REPO_URL)
        repo_name = shlex.quote(ARTIFACT_FS_REPO_NAME)
        mount_root = shlex.quote(ARTIFACT_FS_MOUNT_ROOT)
        code, _stdout, stderr = _run(
            sandbox,
            f"mkdir -p {mount_root} && artifact-fs add-repo --name {repo_name} --remote {repo_url} --branch main --mount-root {mount_root}",
            timeout=180,
        )
        assert code == 0, f"artifact-fs add-repo failed: {stderr}"

        code, _stdout, stderr = _run(
            sandbox,
            f"nohup artifact-fs daemon --root {mount_root} >/tmp/afs.log 2>&1 & echo $! >/tmp/afs.pid",
            timeout=15,
        )
        assert code == 0, f"failed to launch artifact-fs daemon: {stderr}"

        mount_path = f"{ARTIFACT_FS_MOUNT_ROOT}/{ARTIFACT_FS_REPO_NAME}"
        mount_path_q = shlex.quote(mount_path)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            code, stdout, _stderr = _run(
                sandbox,
                f"mountpoint -q {mount_path_q} && echo ok || echo no",
                timeout=10,
            )
            if stdout.strip() == "ok":
                break
            time.sleep(1)
        else:
            _, log, _ = _run(sandbox, "cat /tmp/afs.log", timeout=10)
            pytest.fail(f"artifact-fs mount never became ready at {mount_path}:\n{log}")

        # Hydration is on-demand and can race the mount becoming ready;
        # retry until README content arrives.
        readme_deadline = time.monotonic() + 30
        readme = ""
        while time.monotonic() < readme_deadline:
            code, stdout, stderr = _run(sandbox, f"head -n 5 {mount_path_q}/README.md", timeout=30)
            assert code == 0, f"could not read README.md from mount: {stderr}"
            if stdout.strip():
                readme = stdout
                break
            time.sleep(1)
        assert "deepagents" in readme.lower(), f"unexpected README contents: {readme!r}"

        code, stdout, stderr = _run(
            sandbox,
            f"git -C {mount_path_q} log -1 --format=%H",
            timeout=60,
        )
        assert code == 0, f"git log failed inside mount: {stderr}"
        commit = stdout.strip()
        assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), f"unexpected HEAD oid: {commit!r}"
    finally:
        # Stop the daemon so unmount happens cleanly before sandbox deletion.
        try:
            _run(
                sandbox,
                'if [ -f /tmp/afs.pid ]; then kill "$(cat /tmp/afs.pid)" 2>/dev/null || true; fi',
                timeout=10,
            )
        finally:
            client.delete_sandbox(sandbox.name)
