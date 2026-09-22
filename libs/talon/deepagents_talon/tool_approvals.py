"""Persist exact-name approval policies separately from invocation snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from deepagents_talon.mcp_config import locked_path

if TYPE_CHECKING:
    from langchain.agents.middleware import InterruptOnConfig
    from langchain_core.tools import BaseTool

_MAX_BYTES = 1_048_576
_MAX_TOOLS = 4096
_MAX_NAME = 256
_DEFAULTS = dict.fromkeys(
    ("update_tool_approvals", "delete_conversations", "update_mcp_server", "start_async_task"),
    True,
)


def _validate(document: object) -> dict[str, bool]:
    if not isinstance(document, Mapping) or len(document) > _MAX_TOOLS:
        msg = "Invalid tool approvals mapping."
        raise ValueError(msg)
    result = {}
    for name, value in document.items():
        if (
            not isinstance(name, str)
            or not 0 < len(name) <= _MAX_NAME
            or name != name.strip()
            or any(c in "*?[]" or unicodedata.category(c).startswith("C") for c in name)
            or type(value) is not bool
        ):
            msg = "Invalid tool approval entry."
            raise ValueError(msg)
        result[name] = value
    return result


def _unique_entries(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            msg = "Duplicate tool approval entry."
            raise ValueError(msg)
        result[name] = value
    return result


@dataclass(frozen=True)
class ApprovalSnapshot:
    """Immutable byte revision and exact-name policy; unspecified tools need no prompt."""

    revision: str
    approvals: Mapping[str, bool]

    def __post_init__(self) -> None:
        """Copy and freeze validated policy entries."""
        object.__setattr__(self, "approvals", MappingProxyType(_validate(self.approvals)))

    @property
    def interrupt_on(self) -> dict[str, bool | InterruptOnConfig]:
        """Return fresh approve/reject interrupts for enabled exact names."""
        return {
            name: {"allowed_decisions": ["approve", "reject"]}
            for name, enabled in self.approvals.items()
            if enabled
        }


ACTIVE_APPROVALS: ContextVar[ApprovalSnapshot | None] = ContextVar("ACTIVE_APPROVALS", default=None)
APPROVAL_OPERATOR: ContextVar[bool] = ContextVar("APPROVAL_OPERATOR", default=False)


class ToolApprovalStore:
    """Manage one operator-selected policy file with byte-revision compare-and-swap."""

    def __init__(self, path: Path) -> None:
        """Fix the parent directory without following the final path component."""
        self._path = path.parent.resolve() / path.name

    def ensure(self) -> ApprovalSnapshot:
        """Materialize defaults only when missing; fail on invalid existing policy."""
        with locked_path(self._path):
            try:
                return self.read()
            except FileNotFoundError:
                with suppress(FileExistsError):
                    self._write(_DEFAULTS, create=True)
                return self.read()

    def read(self) -> ApprovalSnapshot:
        """Read a bounded regular file without following symlinks; invalid policy raises."""
        descriptor = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                msg = "Tool approvals must be a regular file."
                raise ValueError(msg)
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            msg = "Tool approvals file is too large."
            raise ValueError(msg)
        document = json.loads(raw, object_pairs_hook=_unique_entries)
        return ApprovalSnapshot(hashlib.sha256(raw).hexdigest(), _validate(document))

    def _write(self, approvals: Mapping[str, bool], *, create: bool = False) -> None:
        raw = (json.dumps(dict(approvals), indent=2, sort_keys=True) + "\n").encode()
        if len(raw) > _MAX_BYTES:
            msg = "Tool approvals file is too large."
            raise ValueError(msg)
        descriptor, temporary = tempfile.mkstemp(prefix=".tools-", dir=self._path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if create:
                os.link(temporary, self._path)
            else:
                if self._path.is_symlink():
                    msg = "Tool approvals must not be a symlink."
                    raise ValueError(msg)
                Path(temporary).replace(self._path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def update(self, updates: dict[str, object], expected_revision: str) -> dict[str, object]:
        """Atomically merge a validated batch only when the persisted byte revision matches."""
        try:
            validated = _validate(updates)
            with locked_path(self._path):
                previous = self.read()
                if previous.revision != expected_revision:
                    return {"status": "conflict", "message": "Read tool approvals again."}
                merged = _validate(dict(previous.approvals) | validated)
                if merged != previous.approvals:
                    self._write(merged)
                saved = self.read()
        except TimeoutError:
            return {"status": "conflict", "message": "Tool approvals are busy; try again."}
        except (OSError, ValueError, TypeError, RecursionError):
            return {"status": "error", "message": "Cannot update tool approvals."}
        return {
            "status": "updated",
            "persisted_revision": saved.revision,
            "tools": dict(saved.approvals),
            "available": "next_invocation",
        }

    def view(self, active: ApprovalSnapshot) -> dict[str, object]:
        """Show persisted policy separately from the unchanged invocation snapshot."""
        result: dict[str, object] = {
            "active_revision": active.revision,
            "active_tools": dict(active.approvals),
        }
        try:
            saved = self.read()
        except (OSError, ValueError, TypeError, RecursionError):
            return result | {
                "status": "error",
                "message": "Cannot read persisted tool approvals.",
                "persisted_revision": None,
                "tools": None,
                "saved_changes_inactive": None,
            }
        return result | {
            "persisted_revision": saved.revision,
            "tools": dict(saved.approvals),
            "saved_changes_inactive": saved.revision != active.revision,
        }

    def tools(self, active: ApprovalSnapshot) -> tuple[BaseTool, BaseTool]:
        """Bind read/update tools to an invocation snapshot, not future saved policy."""

        @tool
        def get_tool_approvals() -> dict[str, object]:
            """Read saved and invocation policies; saved changes apply next invocation only."""
            return self.view(active)

        @tool
        def update_tool_approvals(
            updates: dict[str, object], expected_revision: str
        ) -> dict[str, object]:
            """Save an operator-authorized batch for next invocation; old snapshots stay unchanged.

            Args:
                updates: Exact tool names mapped to booleans; false disables prompting,
                    not operator authorization. Unspecified entries are preserved.
                expected_revision: persisted_revision from get_tool_approvals, for CAS.
            """
            if APPROVAL_OPERATOR.get() is not True or ACTIVE_APPROVALS.get() is None:
                return {"status": "error", "message": "Operator authorization is required."}
            result = self.update(updates, expected_revision)
            if result.get("status") == "updated":
                return result | self.view(active)
            return result

        return get_tool_approvals, update_tool_approvals
