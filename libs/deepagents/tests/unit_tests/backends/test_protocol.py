"""Tests for BackendProtocol and SandboxBackendProtocol base class behavior.

Verifies that unimplemented protocol methods raise NotImplementedError
instead of silently returning None.
"""

import asyncio
import errno
import warnings
from unittest.mock import patch

import pytest

from deepagents.backends.filesystem import _map_exception_to_standard_error
from deepagents.backends.protocol import (
    ASYNC_GREP_TIMEOUT,
    DEFAULT_GREP_TIMEOUT,
    BackendProtocol,
    DeleteResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    _method_accepts_max_count,
    _supports_delete,
)


class BareBackend(BackendProtocol):
    """Minimal subclass that implements nothing."""


class BareSandboxBackend(SandboxBackendProtocol):
    """Minimal subclass that implements nothing."""


@pytest.fixture
def backend() -> BareBackend:
    return BareBackend()


@pytest.fixture
def sandbox_backend() -> BareSandboxBackend:
    return BareSandboxBackend()


class TestBackendProtocolRaisesNotImplemented:
    """All sync methods on BackendProtocol must raise NotImplementedError."""

    def test_ls_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.ls("/")

    def test_read(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.read("/file.txt")

    def test_grep(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.grep("pattern")

    def test_glob(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.glob("*.py")

    def test_write(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.write("/file.txt", "content")

    def test_edit(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.edit("/file.txt", "old", "new")

    def test_delete(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.delete("/file.txt")

    def test_upload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.upload_files([("/file.txt", b"data")])

    def test_download_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.download_files(["/file.txt"])


class TestSandboxBackendProtocolRaisesNotImplemented:
    """SandboxBackendProtocol.execute must raise NotImplementedError."""

    def test_execute(self, sandbox_backend: BareSandboxBackend) -> None:
        with pytest.raises(NotImplementedError):
            sandbox_backend.execute("ls")


class TestAsyncMethodsPropagateNotImplemented:
    """Async wrappers delegate to sync methods, so NotImplementedError propagates."""

    async def test_als_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.als("/")

    async def test_aread(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aread("/file.txt")

    async def test_agrep(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.agrep("pattern")

    async def test_aglob(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aglob("*.py")

    async def test_awrite(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.awrite("/file.txt", "content")

    async def test_aedit(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aedit("/file.txt", "old", "new")

    async def test_adelete(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.adelete("/file.txt")


class TestSupportsDelete:
    """`_supports_delete` detects whether a backend overrides `delete`."""

    def test_false_when_not_overridden(self, backend: BareBackend) -> None:
        assert _supports_delete(backend) is False

    def test_true_when_overridden(self) -> None:
        class MyBackend(BackendProtocol):
            def delete(self, file_path: str) -> DeleteResult:
                return DeleteResult(path=file_path)

        assert _supports_delete(MyBackend()) is True


class TestDeprecatedMethodsRouteToNewNames:
    """Old method names warn and delegate to the new implementations."""

    def test_ls_info_delegates_to_ls(self) -> None:
        class MyBackend(BackendProtocol):
            def ls(self, path: str) -> LsResult:
                return LsResult(entries=[{"path": path}])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert MyBackend().ls_info("/") == [{"path": "/"}]
        assert any("ls_info" in str(x.message) for x in w)

    def test_ls_info_raises_for_new_only_ls_behavior(self) -> None:
        class MyBackend(BackendProtocol):
            def ls(self, path: str) -> LsResult:
                return LsResult(error="error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(NotImplementedError, match="new `ls` API"):
                MyBackend().ls_info("/")
        assert any("ls_info" in str(x.message) for x in w)

    def test_grep_raw_delegates_to_grep(self) -> None:
        class MyBackend(BackendProtocol):
            def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
                return GrepResult(error="error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert MyBackend().grep_raw("x") == "error"
        assert any("grep_raw" in str(x.message) for x in w)

    def test_glob_info_delegates_to_glob(self) -> None:
        class MyBackend(BackendProtocol):
            def glob(self, pattern: str, path: str | None = None) -> GlobResult:
                return GlobResult(matches=[{"path": f"{path}/{pattern}"}])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert MyBackend().glob_info("*.py") == [{"path": "//*.py"}]
        assert any("glob_info" in str(x.message) for x in w)


class TestLegacySubclassOverrideRouting:
    """New method names detect legacy overrides and delegate back."""

    def test_ls_routes_to_ls_info_override(self) -> None:
        class LegacyBackend(BackendProtocol):
            def ls_info(self, path: str) -> list[dict[str, str]]:
                return [{"path": path}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert LegacyBackend().ls("/") == LsResult(entries=[{"path": "/"}])
        assert any("ls_info" in str(x.message) for x in w)

    def test_grep_routes_to_grep_raw_override(self) -> None:
        class LegacyBackend(BackendProtocol):
            def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[dict[str, str | int]] | str:
                return [{"path": "/f", "line": 1, "text": pattern}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert LegacyBackend().grep("x") == GrepResult(matches=[{"path": "/f", "line": 1, "text": "x"}])
        assert any("grep_raw" in str(x.message) for x in w)

    def test_grep_raw_override_respects_max_count(self) -> None:
        """`grep` caps a legacy `grep_raw` override post-hoc, honoring the boundary."""

        class LegacyBackend(BackendProtocol):
            def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[dict[str, str | int]] | str:
                return [
                    {"path": "/one", "line": 1, "text": pattern},
                    {"path": "/two", "line": 2, "text": pattern},
                    {"path": "/three", "line": 3, "text": pattern},
                ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            capped = LegacyBackend().grep("x", max_count=2)
            exact = LegacyBackend().grep("x", max_count=3)

        # More matches than the cap: trimmed to `max_count` and flagged truncated.
        assert capped.matches == [{"path": "/one", "line": 1, "text": "x"}, {"path": "/two", "line": 2, "text": "x"}]
        assert capped.truncated is True
        # Exactly `max_count` matches with none dropped is reported complete.
        assert exact.matches is not None
        assert len(exact.matches) == 3
        assert exact.truncated is False

    def test_glob_routes_to_glob_info_override(self) -> None:
        class LegacyBackend(BackendProtocol):
            def glob_info(self, pattern: str, path: str = "/") -> list[dict[str, str]]:
                return [{"path": f"{path}/{pattern}"}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert LegacyBackend().glob("*.py") == GlobResult(matches=[{"path": "//*.py"}])
        assert any("glob_info" in str(x.message) for x in w)

    async def test_aupload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aupload_files([("/file.txt", b"data")])

    async def test_adownload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.adownload_files(["/file.txt"])

    async def test_aexecute(self, sandbox_backend: BareSandboxBackend) -> None:
        with pytest.raises(NotImplementedError):
            await sandbox_backend.aexecute("ls")


class TestAgrepTimeout:
    """Tests for `agrep` async timeout safety net."""

    def test_agrep_timeout_exceeds_two_sync_grep_phases(self) -> None:
        """`agrep` gives `FilesystemBackend` headroom for `rg` timeout plus fallback timeout."""
        assert ASYNC_GREP_TIMEOUT > (2 * DEFAULT_GREP_TIMEOUT)

    async def test_agrep_returns_error_on_timeout(self, backend: BareBackend) -> None:
        """`agrep` catches `TimeoutError` and returns `GrepResult` with error."""
        seen_timeout = None

        async def mock_wait_for(coro, *, timeout):  # noqa: ASYNC109
            nonlocal seen_timeout
            seen_timeout = timeout
            coro.close()
            raise TimeoutError

        with patch.object(asyncio, "wait_for", mock_wait_for):
            result = await backend.agrep("pattern", "/path", "*.py")

        assert seen_timeout == ASYNC_GREP_TIMEOUT
        assert result.error is not None
        assert "timed out" in result.error
        assert result.matches is None

    async def test_agrep_propagates_not_implemented(self, backend: BareBackend) -> None:
        """`NotImplementedError` from `grep` still propagates through the timeout wrapper."""
        with pytest.raises(NotImplementedError):
            await backend.agrep("pattern")

    async def test_agrep_caps_legacy_grep_result(self) -> None:
        """The inherited async wrapper caps results from an old `grep` signature."""

        class LegacyBackend(BackendProtocol):
            def grep(  # ty: ignore[invalid-method-override]  # Intentionally models the old public signature.
                self,
                pattern: str,
                path: str | None = None,
                glob: str | None = None,
            ) -> GrepResult:
                return GrepResult(
                    matches=[
                        {"path": "/one.txt", "line": 1, "text": pattern},
                        {"path": "/two.txt", "line": 1, "text": pattern},
                        {"path": "/three.txt", "line": 1, "text": pattern},
                    ]
                )

        result = await LegacyBackend().agrep("needle", max_count=2)

        assert result.matches is not None
        assert len(result.matches) == 2
        assert result.truncated is True


def _runtime_error_from_eloop_context() -> RuntimeError:
    """Create the Python <=3.12 `Path.resolve()` symlink-loop shape via `__context__`."""
    exc = RuntimeError("resolver failed")
    exc.__context__ = OSError(errno.ELOOP, "Too many levels of symbolic links")
    return exc


def _runtime_error_from_eloop_cause() -> RuntimeError:
    """Same shape but using `__cause__` (explicit `raise ... from`)."""
    exc = RuntimeError("resolver failed")
    exc.__cause__ = OSError(errno.ELOOP, "Too many levels of symbolic links")
    return exc


class TestMapFileOperationError:
    """map_file_operation_error classifies exceptions into FileOperationError codes."""

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (FileNotFoundError("gone"), "file_not_found"),
            (PermissionError("denied"), "permission_denied"),
            (IsADirectoryError("dir"), "is_directory"),
            (ValueError("path traversal detected"), "invalid_path"),
            (ValueError("invalid path segment"), "invalid_path"),
            (NotADirectoryError("not a dir"), "invalid_path"),
            (FileExistsError("exists"), "invalid_path"),
            (OSError(errno.ELOOP, "Too many levels of symbolic links"), "invalid_path"),
            (_runtime_error_from_eloop_context(), "invalid_path"),
            (_runtime_error_from_eloop_cause(), "invalid_path"),
        ],
    )
    def test_known_exception_types(self, exc: Exception, expected: str) -> None:
        assert _map_exception_to_standard_error(exc) == expected

    def test_unrecognized_returns_none(self) -> None:
        """Non-stdlib exception types return None regardless of message."""
        assert _map_exception_to_standard_error(RuntimeError("something else")) is None
        assert _map_exception_to_standard_error(RuntimeError("permission denied")) is None
        assert _map_exception_to_standard_error(OSError("is a directory")) is None

    def test_value_error_maps_to_invalid_path(self) -> None:
        """All ValueError instances map to invalid_path regardless of message."""
        assert _map_exception_to_standard_error(ValueError("unexpected encoding")) == "invalid_path"
        assert _map_exception_to_standard_error(ValueError("invalid literal for int()")) == "invalid_path"
        assert _map_exception_to_standard_error(ValueError("Path traversal not allowed")) == "invalid_path"


class TestMethodAcceptsMaxCount:
    """`_method_accepts_max_count` decides whether the cap is forwarded or applied post-hoc."""

    def test_explicit_keyword_param_detected(self) -> None:
        class Backend(BackendProtocol):
            def grep(self, pattern: str, path: str | None = None, glob: str | None = None, *, max_count: int | None = None) -> GrepResult:
                return GrepResult(matches=[])

        assert _method_accepts_max_count(Backend, "grep") is True

    def test_var_keyword_param_detected(self) -> None:
        """A `**kwargs` grep is treated as accepting the cap (forwarded, not post-hoc)."""

        class Backend(BackendProtocol):
            def grep(self, pattern: str, path: str | None = None, glob: str | None = None, **kwargs: object) -> GrepResult:
                return GrepResult(matches=[])

        assert _method_accepts_max_count(Backend, "grep") is True

    def test_missing_param_not_detected(self) -> None:
        class Backend(BackendProtocol):
            def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:  # ty: ignore[invalid-method-override]
                return GrepResult(matches=[])

        assert _method_accepts_max_count(Backend, "grep") is False


class TestReadResultPaginationInvariants:
    """`ReadResult.__post_init__` rejects malformed pagination-field combinations."""

    def test_no_pagination_is_valid(self) -> None:
        """A bare result and an error result carry no window and must not raise."""
        assert ReadResult().start_line is None
        assert ReadResult(error="boom").total_lines is None

    def test_full_valid_window(self) -> None:
        """A well-formed window with matching metadata is accepted."""
        result = ReadResult(total_lines=5, start_line=2, end_line=3, next_offset=3)
        assert result.next_offset == result.end_line

    def test_terminal_window_has_no_next_offset(self) -> None:
        """The final page (`next_offset` unset) is valid even when it reaches EOF."""
        result = ReadResult(total_lines=3, start_line=2, end_line=3, next_offset=None)
        assert result.next_offset is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"start_line": 1}, id="start_without_end"),
            pytest.param({"end_line": 1}, id="end_without_start"),
            pytest.param({"next_offset": 5}, id="next_offset_without_window"),
            pytest.param({"total_lines": 10}, id="total_without_window"),
            pytest.param({"start_line": 3, "end_line": 2}, id="start_after_end"),
            pytest.param({"start_line": 0, "end_line": 2}, id="start_below_one"),
            pytest.param(
                {"start_line": 1, "end_line": 5, "total_lines": 3},
                id="total_below_end",
            ),
            pytest.param(
                {"start_line": 1, "end_line": 3, "next_offset": 99},
                id="next_offset_not_end_line",
            ),
        ],
    )
    def test_malformed_combinations_raise(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="ReadResult"):
            ReadResult(**kwargs)
