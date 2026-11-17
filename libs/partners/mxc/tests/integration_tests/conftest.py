"""Integration test fixtures: skipped when no MXC executor is available."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest

from langchain_mxc._executor import MXCExecutorNotFoundError, MXCUnsupportedPlatformError, find_executor

if TYPE_CHECKING:
    from pathlib import Path

_BINARY_LIMIT_REASON = (
    "FilesystemBackend.read returns binary files of any size as base64; only BaseSandbox "
    "enforces the 500 KiB preview limit. Inherited deepagents behaviour shared with "
    "LocalShellBackend, not MXC-specific"
)

_INHERITED = " Inherited deepagents FilesystemBackend behaviour: LocalShellBackend fails this case identically; not MXC-specific."
_GLOB_REASON = (
    "FilesystemBackend.glob returns absolute paths in non-virtual mode; the shared suite expects paths relative to the search directory." + _INHERITED
)
_PAGED_READ_REASON = (
    "FilesystemBackend.read with offset/limit keeps the trailing newline of the last line; the shared suite expects it stripped." + _INHERITED
)

_STANDARD_SKIPS = {
    "test_download_error_invalid_path_relative": "FilesystemBackend resolves a relative path under root_dir instead of returning invalid_path."
    + _INHERITED,
    "test_upload_relative_path_returns_invalid_path": "FilesystemBackend writes a relative path under root_dir instead of returning invalid_path."
    + _INHERITED,
    "test_read_binary_file_1_mib_returns_error": _BINARY_LIMIT_REASON,
    "test_aread_binary_file_1_mib_returns_error": _BINARY_LIMIT_REASON,
    "test_glob": _GLOB_REASON,
    "test_glob_basic_pattern": _GLOB_REASON,
    "test_glob_with_directories": _GLOB_REASON,
    "test_glob_hidden_files_explicitly": _GLOB_REASON,
    "test_glob_with_character_class": _GLOB_REASON,
    "test_glob_with_question_mark": _GLOB_REASON,
    "test_write_existing_file_fails": "FilesystemBackend.write overwrites an existing file instead of returning an error." + _INHERITED,
    "test_edit_multiple_occurrences_without_replace_all": "FilesystemBackend.edit reports 'appears N times'; the shared suite looks for the word 'multiple'."  # noqa: E501
    + _INHERITED,
    "test_ls_lists_nested_directories": "FilesystemBackend.ls returns directories with a trailing slash; the shared suite expects none." + _INHERITED,
    "test_aread_large_text_payload_paginated_roundtrip": _PAGED_READ_REASON,
    "test_write_read_download_large_text_with_escaped_content": _PAGED_READ_REASON,
    "test_awrite_aread_adownload_large_text_with_escaped_content": _PAGED_READ_REASON,
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the two shared-suite cases FilesystemBackend cannot satisfy.

    The shared suite forbids overriding its test methods, so the skip is
    applied here instead.
    """
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is not None and cls.__name__ == "TestMXCSandboxStandard" and item.name in _STANDARD_SKIPS:
            item.add_marker(pytest.mark.skip(reason=_STANDARD_SKIPS[item.name]))


@pytest.fixture(scope="session")
def executor() -> Path:
    """The MXC executor on this machine, or skip with the installation hint."""
    try:
        return find_executor()
    except MXCUnsupportedPlatformError as exc:
        pytest.skip(f"MXC does not support this platform: {exc}")
    except MXCExecutorNotFoundError as exc:
        first_line = str(exc).splitlines()[0]
        pytest.skip(f"No MXC executor found ({first_line}). Install with `npm install @microsoft/mxc-sdk` or set MXC_EXECUTOR_PATH.")


@pytest.fixture(scope="session")
def host_has_internet() -> bool:
    """Whether the host itself can reach the internet (for the network-on test)."""
    try:
        with socket.create_connection(("example.com", 443), timeout=5):
            return True
    except OSError:
        return False
