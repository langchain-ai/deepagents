"""Harbor environment backed by LangSmith production sandboxes."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tarfile
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from langsmith import Client

if TYPE_CHECKING:
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    from requests import Response

_DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
_SANDBOX_API_PATH = "/v2/sandboxes"
_LANGSMITH_ENDPOINT_ENV = "LANGSMITH_ENDPOINT"
_LANGSMITH_SANDBOX_API_URL_ENV = "LANGSMITH_SANDBOX_API_URL"
_DEFAULT_DELETE_AFTER_STOP_SECONDS = 7200
_DEFAULT_IDLE_TTL_SECONDS = 0
_DEFAULT_WORKDIR = "/app"
_REMOTE_TMP_DIR = "/tmp"  # noqa: S108  # Remote sandbox scratch path.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
_DEFAULT_POLL_INTERVAL_SECONDS = 2.0
_DEFAULT_STARTUP_TIMEOUT_SECONDS = 900
_ONE_MIB = 1024 * 1024
_HTTP_TIMEOUT_MS_PER_SECOND = 1000

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class LangSmithSandboxEnvironment(BaseEnvironment):
    """Run Harbor trials on LangSmith production sandboxes.

    The adapter intentionally targets prebuilt image tasks first. Harbor tasks
    that only provide an `environment/Dockerfile` need an image build/push step
    before the LangSmith sandbox snapshot API can consume them.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        api_key: str | None = None,
        langsmith_api_key: str | None = None,
        langsmith_endpoint: str | None = None,
        sandbox_api_url: str | None = None,
        snapshot_name: str | None = None,
        create_snapshot: bool = True,
        delete_snapshot: bool = False,
        registry_id: str | None = None,
        ttl_seconds: int = _DEFAULT_DELETE_AFTER_STOP_SECONDS,
        delete_after_stop_seconds: int | None = None,
        idle_ttl_seconds: int = _DEFAULT_IDLE_TTL_SECONDS,
        workdir: str = _DEFAULT_WORKDIR,
        request_timeout_seconds: int = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        startup_timeout_seconds: int = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangSmith sandbox environment.

        Args:
            environment_dir: Harbor task environment directory.
            environment_name: Harbor environment name.
            session_id: Harbor trial session identifier.
            trial_paths: Harbor trial paths.
            task_env_config: Harbor task environment configuration.
            api_key: Optional LangSmith API key. Prefer `LANGSMITH_API_KEY`.
            langsmith_api_key: Alias for `api_key`.
            langsmith_endpoint: LangSmith API endpoint.
            sandbox_api_url: Fully qualified sandbox API URL.
            snapshot_name: Existing sandbox snapshot name to use.
            create_snapshot: Whether to create a snapshot from `docker_image`.
            delete_snapshot: Whether to delete snapshots created by this adapter.
            registry_id: Optional LangSmith sandbox registry UUID.
            ttl_seconds: Backwards-compatible alias for `delete_after_stop_seconds`.
                Must be zero or minute-aligned.
            delete_after_stop_seconds: Time after stop before sandbox cleanup.
                Defaults to `ttl_seconds`.
            idle_ttl_seconds: Idle TTL. `0` disables idle cleanup during a trial.
            workdir: Default command working directory when Harbor does not pass one.
            request_timeout_seconds: HTTP request timeout.
            poll_interval_seconds: Snapshot polling interval.
            startup_timeout_seconds: Snapshot startup timeout.
            **kwargs: Forward-compatible Harbor environment kwargs.
        """
        api_key_ = api_key or langsmith_api_key
        api_url = langsmith_endpoint or os.environ.get(_LANGSMITH_ENDPOINT_ENV)
        timeout_ms = request_timeout_seconds * _HTTP_TIMEOUT_MS_PER_SECOND
        self._client = Client(
            api_url=api_url,
            api_key=api_key_,
            timeout_ms=(timeout_ms, timeout_ms),
        )
        self._langsmith_endpoint = _langsmith_endpoint_from_api_url(
            api_url or self._client.api_url or _DEFAULT_LANGSMITH_ENDPOINT
        )
        configured_sandbox_api_url = sandbox_api_url or os.environ.get(
            _LANGSMITH_SANDBOX_API_URL_ENV
        )
        self._sandbox_api_url = configured_sandbox_api_url or _join_url(
            self._langsmith_endpoint, _SANDBOX_API_PATH
        )

        self._snapshot_name = snapshot_name
        self._create_snapshot = create_snapshot
        self._delete_snapshot = delete_snapshot
        self._registry_id = registry_id
        resolved_delete_after_stop_seconds = (
            ttl_seconds if delete_after_stop_seconds is None else delete_after_stop_seconds
        )
        self._delete_after_stop_seconds = _validate_ttl_seconds(
            "delete_after_stop_seconds", resolved_delete_after_stop_seconds
        )
        self._idle_ttl_seconds = _validate_ttl_seconds("idle_ttl_seconds", idle_ttl_seconds)
        self._workdir = workdir
        self._request_timeout_seconds = request_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._startup_timeout_seconds = startup_timeout_seconds

        self._claim_name = _k8s_name("harbor", session_id)
        self._claim_id: str | None = None
        self._created_snapshot_id: str | None = None
        self._active_snapshot_id: str | None = None
        self._dataplane_url: str | None = None

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

    @staticmethod
    def type() -> EnvironmentType:
        """Return a placeholder built-in type for Harbor's older enum contract."""
        return EnvironmentType.DOCKER

    @property
    def is_mounted(self) -> bool:
        """Whether Harbor log directories are host-mounted."""
        return False

    @property
    def supports_gpus(self) -> bool:
        """Whether this adapter supports GPU allocation."""
        return False

    @property
    def can_disable_internet(self) -> bool:
        """Whether this adapter can block outbound internet."""
        return True

    def _validate_definition(self) -> None:
        if self._snapshot_name:
            return
        if self.task_env_config.docker_image:
            return

        dockerfile = self.environment_dir / "Dockerfile"
        if dockerfile.exists():
            msg = (
                "LangSmithSandboxEnvironment requires [environment].docker_image "
                "or environment.kwargs.snapshot_name. Dockerfile build/push support "
                "is not implemented in this adapter yet."
            )
            raise ValueError(msg)

        msg = (
            "LangSmithSandboxEnvironment requires [environment].docker_image "
            "or environment.kwargs.snapshot_name."
        )
        raise ValueError(msg)

    async def start(self, force_build: bool) -> None:
        """Create a snapshot-backed LangSmith sandbox."""
        snapshot_name = await self._resolve_snapshot_name(force_build)
        claim = await self._request_json(
            "POST",
            self._api_url("boxes"),
            body=self._create_claim_payload(snapshot_name),
            expected_statuses={201},
        )

        self._claim_id = _expect_str(claim, "id")
        self._dataplane_url = _expect_str(claim, "dataplane_url")
        await self.exec(
            "mkdir -p /app /logs/agent /logs/verifier && chmod 777 /app /logs/agent /logs/verifier",
            cwd=_REMOTE_TMP_DIR,
        )

    async def stop(self, delete: bool) -> None:
        """Delete the LangSmith sandbox when Harbor requests cleanup."""
        try:
            if delete and self._claim_id:
                await self._request_bytes(
                    "DELETE",
                    self._api_url("boxes", self._claim_id),
                    expected_statuses={204},
                )
            elif not delete and self._claim_id:
                self.logger.info(
                    "Leaving LangSmith sandbox running because delete=False: %s",
                    self._claim_name,
                )

            if delete and self._delete_snapshot and self._created_snapshot_id:
                await self._request_bytes(
                    "DELETE",
                    self._api_url("snapshots", self._created_snapshot_id),
                    expected_statuses={204},
                )
        finally:
            self._claim_id = None
            self._dataplane_url = None

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a local file into the sandbox."""
        source = Path(source_path)
        await self._request_upload(
            self._dataplane_endpoint("upload", {"path": target_path}),
            source,
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a directory into the sandbox preserving nested paths."""
        source = Path(source_dir)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
            await asyncio.to_thread(_create_archive, source, Path(archive.name))
            remote_archive = (
                f"{_REMOTE_TMP_DIR}/{_k8s_name('harbor-upload', self.session_id)}.tar.gz"
            )
            await self.upload_file(archive.name, remote_archive)

        target = _sh_quote(target_dir)
        archive_path = _sh_quote(remote_archive)
        result = await self.exec(
            f"mkdir -p {target} && tar -xzf {archive_path} -C {target} && rm -f {archive_path}"
        )
        if result.return_code != 0:
            msg = f"upload_dir extraction failed: {result.stderr or result.stdout or ''}"
            raise RuntimeError(msg)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a sandbox file to the local machine."""
        data = await self._request_bytes(
            "GET",
            self._dataplane_endpoint("download", {"path": source_path}),
            expected_statuses={200},
        )
        target = Path(target_path)
        await asyncio.to_thread(_write_bytes, target, data)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a sandbox directory to the local machine."""
        remote_archive = f"{_REMOTE_TMP_DIR}/{_k8s_name('harbor-download', self.session_id)}.tar.gz"
        source = _sh_quote(source_dir)
        archive = _sh_quote(remote_archive)
        result = await self.exec(f"test -d {source} && tar -C {source} -czf {archive} .")
        if result.return_code != 0:
            msg = f"download_dir archive failed: {result.stderr or result.stdout or ''}"
            raise RuntimeError(msg)

        target = Path(target_dir)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive_file:
            await self.download_file(remote_archive, archive_file.name)
            await asyncio.to_thread(_extract_archive, Path(archive_file.name), target)
        await self.exec(f"rm -f {archive}")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute a shell command inside the sandbox."""
        if user is not None:
            msg = "LangSmith sandbox command execution does not support user overrides yet."
            raise ValueError(msg)

        payload: dict[str, Any] = {"command": command}
        payload["cwd"] = cwd or self._workdir
        if env:
            payload["env"] = env
        if timeout_sec is not None:
            payload["timeout_seconds"] = timeout_sec

        data = await self._request_json(
            "POST",
            self._dataplane_endpoint("execute"),
            body=payload,
            expected_statuses={200},
        )
        return ExecResult(
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            return_code=int(data["exit_code"]),
        )

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        """Return whether `path` is a directory in the sandbox."""
        if user is not None:
            return False
        result = await self.exec(f"test -d {_sh_quote(path)}", timeout_sec=10)
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        """Return whether `path` is a regular file in the sandbox."""
        if user is not None:
            return False
        result = await self.exec(f"test -f {_sh_quote(path)}", timeout_sec=10)
        return result.return_code == 0

    async def _resolve_snapshot_name(self, force_build: bool) -> str:
        if self._snapshot_name:
            snapshot = await self._find_snapshot(self._snapshot_name)
            if snapshot is None:
                msg = f'LangSmith sandbox snapshot "{self._snapshot_name}" was not found.'
                raise RuntimeError(msg)
            self._active_snapshot_id = _expect_str(snapshot, "id")
            await self._wait_for_snapshot_ready(self._active_snapshot_id)
            return self._snapshot_name

        image = self.task_env_config.docker_image
        if image is None:
            msg = "docker_image is required when snapshot_name is not provided."
            raise ValueError(msg)

        snapshot_name = _snapshot_name(self.environment_name, image, force_build, self.session_id)
        existing = None if force_build else await self._find_snapshot(snapshot_name)
        if existing is not None:
            snapshot_id = _expect_str(existing, "id")
            self._active_snapshot_id = snapshot_id
            await self._wait_for_snapshot_ready(snapshot_id)
            return snapshot_name

        if not self._create_snapshot:
            msg = f'LangSmith sandbox snapshot "{snapshot_name}" does not exist.'
            raise RuntimeError(msg)

        payload: dict[str, Any] = {
            "name": snapshot_name,
            "docker_image": image,
            "fs_capacity_bytes": self.task_env_config.storage_mb * _ONE_MIB,
        }
        if self._registry_id:
            payload["registry_id"] = self._registry_id

        snapshot = await self._request_json(
            "POST",
            self._api_url("snapshots"),
            body=payload,
            expected_statuses={201},
        )
        snapshot_id = _expect_str(snapshot, "id")
        self._created_snapshot_id = snapshot_id
        self._active_snapshot_id = snapshot_id
        await self._wait_for_snapshot_ready(snapshot_id)
        return snapshot_name

    async def _find_snapshot(self, name: str) -> dict[str, Any] | None:
        data = await self._request_json(
            "GET",
            self._api_url("snapshots", query={"name_contains": name, "limit": "50"}),
            expected_statuses={200},
        )
        for snapshot in data.get("snapshots", []):
            if snapshot.get("name") == name:
                return snapshot
        return None

    async def _wait_for_snapshot_ready(self, snapshot_id: str) -> None:
        deadline = time.monotonic() + self._startup_timeout_seconds
        while True:
            snapshot = await self._request_json(
                "GET",
                self._api_url("snapshots", snapshot_id),
                expected_statuses={200},
            )
            status = snapshot.get("status")
            if status == "ready":
                return
            if status == "failed":
                msg = snapshot.get("status_message") or "snapshot build failed"
                raise RuntimeError(str(msg))
            if time.monotonic() >= deadline:
                msg = f"Timed out waiting for snapshot {snapshot_id} to become ready."
                raise TimeoutError(msg)
            await asyncio.sleep(self._poll_interval_seconds)

    def _create_claim_payload(self, snapshot_name: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self._claim_name,
            "snapshot_name": snapshot_name,
            "vcpus": self.task_env_config.cpus,
            "mem_bytes": self.task_env_config.memory_mb * _ONE_MIB,
            "fs_capacity_bytes": self.task_env_config.storage_mb * _ONE_MIB,
            "idle_ttl_seconds": self._idle_ttl_seconds,
            "delete_after_stop_seconds": self._delete_after_stop_seconds,
        }
        if not self.task_env_config.allow_internet:
            payload["proxy_config"] = {
                "rules": [],
                "no_proxy": [],
                "access_control": {"deny_list": ["*"]},
            }
        return payload

    def _api_url(
        self,
        *parts: str,
        query: dict[str, str] | None = None,
    ) -> str:
        url = _join_url(self._sandbox_api_url, "/".join(parts))
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return url

    def _dataplane_endpoint(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> str:
        if self._dataplane_url is None:
            msg = "Sandbox dataplane URL is not available. Did start() complete?"
            raise RuntimeError(msg)
        url = _join_url(self._dataplane_url, path)
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return url

    async def _request_json(
        self,
        method: HttpMethod,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        expected_statuses: set[int],
    ) -> dict[str, Any]:
        response = await self._request(
            method,
            url,
            body=body,
            expected_statuses=expected_statuses,
        )
        if not response.content:
            return {}
        decoded = response.json()
        if not isinstance(decoded, dict):
            msg = f"Expected JSON object from {method} request."
            raise TypeError(msg)
        return decoded

    async def _request_upload(self, url: str, path: Path) -> None:
        data = await asyncio.to_thread(path.read_bytes)
        await self._request(
            "POST",
            url,
            files={"file": (path.name, data, "application/octet-stream")},
            expected_statuses={200},
        )

    async def _request_bytes(
        self,
        method: HttpMethod,
        url: str,
        *,
        expected_statuses: set[int],
    ) -> bytes:
        response = await self._request(method, url, expected_statuses=expected_statuses)
        return response.content

    async def _request(
        self,
        method: HttpMethod,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        expected_statuses: set[int],
    ) -> Response:
        return await asyncio.to_thread(
            self._request_sync,
            method,
            url,
            body,
            files,
            expected_statuses,
        )

    def _request_sync(
        self,
        method: HttpMethod,
        url: str,
        body: dict[str, Any] | None,
        files: dict[str, tuple[str, bytes, str]] | None,
        expected_statuses: set[int],
    ) -> Response:
        _validate_http_url(url)
        kwargs: dict[str, Any] = {
            "timeout": (self._request_timeout_seconds, self._request_timeout_seconds)
        }
        if body is not None:
            kwargs["json"] = body
        if files is not None:
            kwargs["files"] = files
        response = self._client.request_with_retries(
            method,
            url,
            **kwargs,
        )

        if response.status_code not in expected_statuses:
            detail = response.text[:1000]
            msg = f"{method} {_redact_query(url)} returned HTTP {response.status_code}: {detail}"
            raise RuntimeError(msg)
        return response


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.strip('/')}"


def _langsmith_endpoint_from_api_url(api_url: str) -> str:
    endpoint = api_url.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return endpoint or _DEFAULT_LANGSMITH_ENDPOINT


def _redact_query(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _expect_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        msg = f"Expected non-empty string field {key!r} in response."
        raise RuntimeError(msg)
    return value


def _validate_http_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        msg = f"Only absolute HTTP(S) URLs are supported: {_redact_query(url)}"
        raise ValueError(msg)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _create_archive(source: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in source.rglob("*"):
            tar.add(path, arcname=path.relative_to(source))


def _extract_archive(archive_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target, filter="data")


def _validate_ttl_seconds(name: str, value: int) -> int:
    if value < 0:
        msg = f"{name} must be >= 0."
        raise ValueError(msg)
    if value > 0 and value % 60 != 0:
        msg = f"{name} must be 0 or a multiple of 60."
        raise ValueError(msg)
    return value


def _k8s_name(prefix: str, value: str, *, max_length: int = 63) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    if not normalized:
        normalized = "sandbox"
    suffix = f"-{digest}"
    available = max_length - len(prefix) - len(suffix) - 1
    trimmed = normalized[:available].strip("-") or "sandbox"
    return f"{prefix}-{trimmed}{suffix}"


def _snapshot_name(
    environment_name: str,
    docker_image: str,
    force_build: bool,
    session_id: str,
) -> str:
    seed = f"{environment_name}:{docker_image}"
    if force_build:
        seed = f"{seed}:{session_id}"
    return _k8s_name("harbor-snap", seed)


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


__all__ = ["LangSmithSandboxEnvironment"]
