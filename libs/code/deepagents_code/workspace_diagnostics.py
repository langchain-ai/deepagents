"""Safe, structured diagnostics for workspace binding and runtime rejections.

When the server refuses to bind or run a thread's workspace, the user sees
only a one-line refusal. This module defines the small, explicitly allowlisted
comparison snapshot persisted beside each durable binding (so a later refusal
can name *what* changed) and the structured diagnostics carried by
`WorkspaceConflictError`, the workspace HTTP route, and the TUI.

The snapshot allowlist is the security boundary: only policy fields whose
values are booleans, integers, tool-name/command allowlists, or short
enum-like identifiers may be recorded and reported. Model parameters, model
specs, profile overrides, system prompts, environment values, credentials,
and *paths* (which can embed user names or project structure) are never
persisted or logged — and never hashed as a reporting workaround.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from textual.content import Content

from deepagents_code.unicode_security import strip_dangerous_unicode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_SNAPSHOT_VERSION = 1
"""Persisted snapshot format generation.

Older bindings carry no snapshot row at all; they report
`snapshot_status="unavailable"` rather than being compared field by field.
"""

_MAX_SNAPSHOT_LENGTH = 16_000
"""Bound on the canonical snapshot JSON stored per thread."""

_MAX_VALUE_LENGTH = 256
"""Bound on any single persisted scalar, so values stay log-safe."""

# Snapshotted subset of `ServerConfig.to_workspace_payload()` keys. Deliberately
# excludes the path-valued payload keys (`sandbox_setup`, `mcp_config_path`,
# `extension_paths`, `sandbox_id`, `sandbox_snapshot_name`): paths can carry
# user-identifying directory names and are not needed to name what drifted.
# `test_snapshot_fields_stay_an_allowlisted_subset_of_the_payload` pins this set
# against the payload so a new payload field is never snapshotted by default.
SAFE_SNAPSHOT_FIELDS = frozenset(
    {
        "allow_fs_tools",
        "assistant_id",
        "auto_approve",
        "enable_ask_user",
        "enable_interpreter",
        "enable_memory",
        "enable_shell",
        "enable_skills",
        "interactive",
        "interpreter_ptc",
        "interpreter_ptc_acknowledge_unsafe",
        "interrupt_shell_only",
        "no_mcp",
        "recursion_limit",
        "sandbox_type",
        "shell_allow_list",
        "trust_project_extensions",
        "trust_project_mcp",
    }
)


def _safe_value(value: object) -> object:
    """Coerce a policy value to its bounded, JSON-safe reportable form.

    Lists are reported as sorted scalar lists (order is not significant in the
    allowlists the payload carries, and sorting keeps snapshots stable).
    Anything outside the expected scalar shapes — or too long — is omitted so
    diagnostics never change whether a workspace configuration is accepted.

    Returns:
        The reportable value, or `None` when the value must not be recorded.
    """
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_VALUE_LENGTH else None
    if isinstance(value, list | tuple) and all(
        isinstance(item, str) and len(item) <= _MAX_VALUE_LENGTH for item in value
    ):
        return sorted(cast("list[str]", value))
    return None


def build_snapshot(workspace_payload: Mapping[str, Any]) -> dict[str, object]:
    """Extract the allowlisted, reportable policy snapshot from a payload.

    Args:
        workspace_payload: The full `ServerConfig.to_workspace_payload()` dict
            persisted with the binding.

    Returns:
        Canonical snapshot mapping restricted to `SAFE_SNAPSHOT_FIELDS`.

    """
    snapshot: dict[str, object] = {}
    for key in sorted(SAFE_SNAPSHOT_FIELDS & workspace_payload.keys()):
        value = _safe_value(workspace_payload[key])
        if value is None and workspace_payload[key] is not None:
            continue
        candidate = {**snapshot, key: value}
        serialized = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        if len(serialized) <= _MAX_SNAPSHOT_LENGTH:
            snapshot = candidate
    return snapshot


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Versioned, allowlisted policy snapshot persisted beside a binding."""

    snapshot_version: int
    fields: Mapping[str, object]

    def to_json(self) -> str:
        """Serialize for durable storage.

        Returns:
            Canonical JSON with the version under a reserved key.
        """
        return json.dumps(
            {"version": self.snapshot_version, "fields": dict(self.fields)},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> WorkspaceSnapshot:
        """Parse a persisted snapshot.

        Args:
            raw: The stored snapshot JSON.

        Returns:
            The parsed snapshot.

        Raises:
            ValueError: If the payload is malformed or an unsupported version.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = "persisted workspace snapshot is not valid JSON"
            raise ValueError(msg) from exc
        if (
            not isinstance(data, dict)
            or data.get("version") != _SNAPSHOT_VERSION
            or not isinstance(data.get("fields"), dict)
        ):
            msg = "persisted workspace snapshot has an unsupported format"
            raise ValueError(msg)
        return cls(
            snapshot_version=_SNAPSHOT_VERSION,
            fields=cast("dict[str, object]", data["fields"]),
        )


def snapshot_for_payload(workspace_payload: Mapping[str, Any]) -> WorkspaceSnapshot:
    """Build the snapshot persisted with a workspace binding.

    Args:
        workspace_payload: The full `ServerConfig.to_workspace_payload()` dict.

    Returns:
        The versioned, allowlisted snapshot.
    """
    return WorkspaceSnapshot(
        snapshot_version=_SNAPSHOT_VERSION,
        fields=build_snapshot(workspace_payload),
    )


_DiagnosticCategory = Literal[
    "unbound_thread",
    "missing_context",
    "context_mismatch",
    "bound_elsewhere",
    "policy_drift",
    "config_drift",
    "fingerprint_mismatch",
    "identity_changed",
    "unsupported_schema",
    "unknown",
]


@dataclass(frozen=True)
class FieldChange:
    """One drifted policy field, reported within the snapshot allowlist."""

    name: str
    bound: object = field(default=None)
    current: object = field(default=None)
    state: Literal["changed", "values_unavailable"] = "changed"

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe wire form."""
        if self.state == "values_unavailable":
            return {"name": self.name, "state": self.state}
        return {
            "name": self.name,
            "state": self.state,
            "bound": self.bound,
            "current": self.current,
        }


@dataclass(frozen=True)
class WorkspaceDiagnostics:
    """Structured, secret-free detail attached to a workspace refusal."""

    category: _DiagnosticCategory
    reason: str
    changes: tuple[FieldChange, ...] = ()
    snapshot_status: Literal["current", "unavailable"] = "current"
    binding_schema_version: int | None = None
    server_schema_version: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe wire form carried in API 409 responses."""
        result: dict[str, object] = {
            "category": self.category,
            "reason": self.reason,
            "snapshot_status": self.snapshot_status,
        }
        if self.changes:
            result["changes"] = [change.to_dict() for change in self.changes]
        if self.binding_schema_version is not None:
            result["binding_schema_version"] = self.binding_schema_version
        if self.server_schema_version is not None:
            result["server_schema_version"] = self.server_schema_version
        return result

    @classmethod
    def from_dict(cls, data: object) -> WorkspaceDiagnostics | None:
        """Parse a diagnostics payload from a server response.

        Args:
            data: The raw `diagnostics` value from a response body.

        Returns:
            The parsed diagnostics, or `None` when absent or malformed; older
            servers and proxies omit the field, and callers must tolerate both.
        """
        if not isinstance(data, dict):
            return None
        category = data.get("category")
        reason = data.get("reason")
        snapshot_status = data.get("snapshot_status")
        if (
            not isinstance(category, str)
            or not isinstance(reason, str)
            or not isinstance(snapshot_status, str)
            or snapshot_status not in {"current", "unavailable"}
        ):
            return None
        changes: list[FieldChange] = []
        raw_changes = data.get("changes")
        if isinstance(raw_changes, list):
            for item in raw_changes:
                parsed = _parse_field_change(item)
                if parsed is None:
                    return None
                changes.append(parsed)
        binding_schema = data.get("binding_schema_version")
        server_schema = data.get("server_schema_version")
        return cls(
            category=cast("_DiagnosticCategory", category),
            reason=reason,
            changes=tuple(changes),
            snapshot_status=snapshot_status,
            binding_schema_version=(
                binding_schema if isinstance(binding_schema, int) else None
            ),
            server_schema_version=(
                server_schema if isinstance(server_schema, int) else None
            ),
        )

    def log_summary(self) -> str:
        """Return the one-line, secret-free summary for warning logs.

        Returns:
            The reason plus drifted field names (never values).
        """
        if not self.changes:
            return self.reason
        names = ", ".join(change.name for change in self.changes)
        return f"{self.reason}; changed field(s): {names}"


def _parse_field_change(item: object) -> FieldChange | None:
    """Parse one wire change entry; anything malformed fails the whole payload.

    Returns:
        The parsed change, or `None` when the entry is not well-formed.
    """
    if not isinstance(item, dict):
        return None
    entry = cast("dict[str, object]", item)
    name = entry.get("name")
    if not isinstance(name, str):
        return None
    state = entry.get("state")
    if state == "values_unavailable":
        return FieldChange(name=name, state="values_unavailable")
    if state == "changed":
        return FieldChange(
            name=name,
            bound=entry.get("bound"),
            current=entry.get("current"),
        )
    return None


def diff_snapshots(
    bound: WorkspaceSnapshot | None,
    current: WorkspaceSnapshot,
    *,
    changed_names: Sequence[str] | None = None,
) -> tuple[FieldChange, ...]:
    """Compare a persisted snapshot against the current one.

    Fields present in the snapshots are compared with values; fields named in
    `changed_names` (e.g. fingerprint-only or path-valued fields detected by a
    fingerprint comparison) are reported as changed without values.

    Args:
        bound: The snapshot persisted with the binding, or `None` for a
            legacy binding that predates snapshots.
        current: The snapshot built from the current resolved policy.
        changed_names: Field names known to differ that snapshots do not
            carry values for.

    Returns:
        Sorted drift entries, values included only where both sides are known.
    """
    changes: list[FieldChange] = []
    if bound is None:
        return tuple(
            FieldChange(name=name, state="values_unavailable")
            for name in sorted(changed_names or ())
        )
    bound_fields = bound.fields
    current_fields = current.fields
    named = set(changed_names or ())
    changes = [
        FieldChange(
            name=key,
            bound=bound_fields.get(key),
            current=current_fields.get(key),
            state=(
                "changed"
                if key in bound_fields and key in current_fields
                else "values_unavailable"
            ),
        )
        for key in sorted(set(bound_fields) | set(current_fields))
        if bound_fields.get(key) != current_fields.get(key)
    ]
    reported = {change.name for change in changes}
    changes.extend(
        FieldChange(name=name, state="values_unavailable")
        for name in sorted(named - reported)
    )
    return tuple(sorted(changes, key=lambda change: change.name))


def join_content(lines: Sequence[Content]) -> Content:
    """Join `Content` lines with newlines (`Content.join` is instance-level).

    Returns:
        One `Content` with each line on its own row.
    """
    return Content("\n").join(lines)


def _display_value(value: object) -> str:
    """Render an allowlisted snapshot value for the terminal.

    Returns:
        A short scalar rendering; complex values degrade to their type name.
    """
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ", ".join(cast("list[str]", value))
    return f"<{type(value).__name__}>"


def format_diagnostics_content(diagnostics: WorkspaceDiagnostics) -> Content:
    """Render workspace refusal diagnostics as markup-safe TUI content.

    Every dynamic value (field names, allowlisted values) flows through
    `Content.from_markup` `$var` substitution, so square brackets in a value
    can never break Rich markup parsing.

    Args:
        diagnostics: Parsed diagnostics from a server refusal.

    Returns:
        A styled summary of what changed since the thread was bound.
    """
    lines = [Content.styled(f"Server refusal: {diagnostics.reason}", "bold")]
    if diagnostics.changes:
        lines.append(
            Content("To resume this thread, restore these settings and relaunch:")
        )
        for change in diagnostics.changes:
            if change.state == "values_unavailable":
                lines.append(
                    Content.from_markup(
                        "  • Restore $field to its original value (unavailable)",
                        field=strip_dangerous_unicode(change.name),
                    )
                )
            else:
                lines.append(
                    Content.from_markup(
                        (
                            "  • Unset $field (currently $current)"
                            if change.bound is None
                            else "  • Set $field to $bound (currently $current)"
                        ),
                        field=strip_dangerous_unicode(change.name),
                        bound=strip_dangerous_unicode(_display_value(change.bound)),
                        current=strip_dangerous_unicode(_display_value(change.current)),
                    )
                )
    if diagnostics.snapshot_status == "unavailable":
        lines.append(
            Content.styled(
                "This thread predates recorded configuration snapshots, so a "
                "detailed comparison is unavailable.",
                "dim",
            )
        )
    return join_content(lines)
