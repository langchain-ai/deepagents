from __future__ import annotations

import os
import shlex
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

ARTIFACT_FS_SNAPSHOT_NAME = "artifact-fs-ready"
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
    r"""Mount langchain-ai/deepagents inside a LangSmith sandbox via cloudflare/artifact-fs.

    Expects a baked snapshot named ``artifact-fs-ready`` with ``fuse3``, ``git``, and the
    ``artifact-fs`` binary preinstalled. FUSE inside the sandbox requires ``/dev/fuse``
    and ``CAP_SYS_ADMIN``; if either is missing, the daemon exits and the mount-readiness
    loop fails fast.

    Building the snapshot (one-time, ~3 minutes; rerun whenever artifact-fs is bumped):

      # 1. Boot a temporary sandbox from ubuntu:24.04 with enough RAM for the Go build.
      langsmith sandbox snapshot build ubuntu-24-04-base \\
          --docker-image ubuntu:24.04 --capacity 16gb --wait
      langsmith sandbox create artifact-fs-bake \\
          --snapshot-id <ubuntu-24-04-base id> --vcpus 4 --memory 8gb --wait

      # 2. Install fuse3 + the latest Go toolchain, then build artifact-fs.
      langsmith sandbox exec artifact-fs-bake -- bash -c '
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
          export PATH=/usr/local/go/bin:$PATH
          GOBIN=/usr/local/bin go install github.com/cloudflare/artifact-fs/cmd/artifact-fs@latest
      '

      # 3. Capture the result and clean up. (Capture currently fails with a leaked
      #    "juicefs: not in PATH" error -- tracked in INF-2013.)
      langsmith sandbox snapshot capture artifact-fs-ready --box artifact-fs-bake --wait
      langsmith sandbox delete artifact-fs-bake
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY is required for the artifact-fs sandbox test")

    client = SandboxClient(api_key=api_key)

    existing = client.list_snapshots(name_contains=ARTIFACT_FS_SNAPSHOT_NAME)
    ready = any(snap.name == ARTIFACT_FS_SNAPSHOT_NAME and snap.status == "ready" for snap in existing)
    if not ready:
        pytest.skip(
            f"Snapshot '{ARTIFACT_FS_SNAPSHOT_NAME}' is missing. "
            f"See the docstring of {test_artifact_fs_mounts_deepagents_repo.__name__} for build steps.",
        )

    sandbox = client.create_sandbox(snapshot_name=ARTIFACT_FS_SNAPSHOT_NAME)
    try:
        # Mount + verify in a single exec: the daemon is a child of this shell,
        # and LangSmith terminates orphaned children when the originating run
        # exits, so subsequent execs would see a stale (or empty) mount.
        repo_url = shlex.quote(ARTIFACT_FS_REPO_URL)
        repo_name = shlex.quote(ARTIFACT_FS_REPO_NAME)
        mount_root = shlex.quote(ARTIFACT_FS_MOUNT_ROOT)
        mount_path = f"{ARTIFACT_FS_MOUNT_ROOT}/{ARTIFACT_FS_REPO_NAME}"
        mount_path_q = shlex.quote(mount_path)
        verify_script = f"""\
set -euo pipefail
mkdir -p {mount_root}
artifact-fs add-repo --name {repo_name} --remote {repo_url} --branch main --mount-root {mount_root}
artifact-fs daemon --root {mount_root} >/tmp/afs.log 2>&1 &
DAEMON_PID=$!
trap 'kill "$DAEMON_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if mountpoint -q {mount_path_q} 2>/dev/null; then break; fi
  sleep 1
done
if ! mountpoint -q {mount_path_q} 2>/dev/null; then
  echo "FAIL: mount never appeared" >&2
  cat /tmp/afs.log >&2
  exit 1
fi
# artifact-fs reports stat size 0 until hydration finishes; wait on -s,
# not on the read result, since head short-circuits on a zero-size stat.
for _ in $(seq 1 90); do
  if [ -s {mount_path_q}/README.md ]; then break; fi
  sleep 1
done
if ! [ -s {mount_path_q}/README.md ]; then
  echo "FAIL: README.md never hydrated" >&2
  echo "--- ls ---" >&2
  ls -la {mount_path_q} >&2 || true
  echo "--- daemon log ---" >&2
  cat /tmp/afs.log >&2 || true
  exit 1
fi
README="$(head -n 5 {mount_path_q}/README.md)"
COMMIT="$(git -C {mount_path_q} log -1 --format=%H)"
echo "===README==="
printf '%s\\n' "$README"
echo "===COMMIT==="
printf '%s\\n' "$COMMIT"
"""
        code, stdout, stderr = _run(sandbox, verify_script, timeout=240)
        assert code == 0, f"artifact-fs verification failed: {stderr}"
        assert "===README===" in stdout and "===COMMIT===" in stdout, f"unexpected output: {stdout!r}"
        readme_block = stdout.split("===README===", 1)[1].split("===COMMIT===", 1)[0]
        commit_block = stdout.split("===COMMIT===", 1)[1].strip()
        assert "deepagents" in readme_block.lower(), f"unexpected README contents: {readme_block!r}"
        assert len(commit_block) == 40 and all(c in "0123456789abcdef" for c in commit_block), f"unexpected HEAD oid: {commit_block!r}"
    finally:
        client.delete_sandbox(sandbox.name)
