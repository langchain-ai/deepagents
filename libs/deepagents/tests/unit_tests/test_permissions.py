"""Unit tests for FilesystemPermission, ToolPermission, and PermissionMiddleware."""

from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import StoreBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.tool_permissions import (
    PermissionMiddleware,
    _check_fs_permission,
    _check_tool_permission,
    _filter_paths_by_permission,
)
from deepagents.permissions import FilesystemPermission, ToolPermission


def _runtime(tool_call_id: str = "") -> ToolRuntime:
    return ToolRuntime(state={}, context=None, tool_call_id=tool_call_id, store=None, stream_writer=lambda _: None, config={})


def _make_backend(files: dict | None = None) -> StoreBackend:
    mem_store = InMemoryStore()
    if files:
        for path, content in files.items():
            mem_store.put(
                ("filesystem",),
                path,
                {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""},
            )
    return StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))


class TestFilesystemPermission:
    def test_default_effect_is_allow(self):
        rule = FilesystemPermission(operations=["read"], paths=["/workspace/**"])
        assert rule.mode == "allow"

    def test_deny_effect(self):
        rule = FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")
        assert rule.mode == "deny"

    def test_multiple_operations(self):
        rule = FilesystemPermission(operations=["read", "write"], paths=["/secrets/**"], mode="deny")
        assert "read" in rule.operations
        assert "write" in rule.operations


class TestToolPermission:
    def test_default_effect_is_allow(self):
        rule = ToolPermission(name="execute", args={"command": "pytest *"})
        assert rule.mode == "allow"

    def test_deny_effect(self):
        rule = ToolPermission(name="execute", mode="deny")
        assert rule.mode == "deny"

    def test_no_arguments_matches_all(self):
        rule = ToolPermission(name="execute", mode="deny")
        assert rule.args is None

    def test_arguments_dict(self):
        rule = ToolPermission(name="execute", args={"command": "pytest *"})
        assert rule.args == {"command": "pytest *"}


class TestCheckToolPermission:
    def test_no_rules_allows_everything(self):
        assert _check_tool_permission([], "execute", {"command": "rm -rf /"}) == "allow"

    def test_deny_all_invocations(self):
        rules = [ToolPermission(name="execute", mode="deny")]
        assert _check_tool_permission(rules, "execute", {}) == "deny"

    def test_allow_matching_arg_pattern(self):
        rules = [
            ToolPermission(name="execute", args={"command": "pytest *"}),
            ToolPermission(name="execute", mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"command": "pytest tests/"}) == "allow"

    def test_deny_non_matching_command(self):
        rules = [
            ToolPermission(name="execute", args={"command": "pytest *"}),
            ToolPermission(name="execute", mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"command": "rm -rf /"}) == "deny"

    def test_unrelated_tool_is_allowed(self):
        rules = [ToolPermission(name="execute", mode="deny")]
        assert _check_tool_permission(rules, "read_file", {"file_path": "/foo"}) == "allow"

    def test_first_matching_rule_wins(self):
        rules = [
            ToolPermission(name="execute", mode="deny"),
            ToolPermission(name="execute", mode="allow"),
        ]
        assert _check_tool_permission(rules, "execute", {}) == "deny"

    def test_multi_arg_matching_all_must_match(self):
        rules = [
            ToolPermission(name="grep", args={"path": "/secrets/*", "glob": "*.py"}, mode="deny"),
        ]
        # Both match → deny
        assert _check_tool_permission(rules, "grep", {"path": "/secrets/key.txt", "glob": "*.py"}) == "deny"
        # Only path matches, glob doesn't → no match → allow
        assert _check_tool_permission(rules, "grep", {"path": "/secrets/key.txt", "glob": "*.txt"}) == "allow"

    def test_globstar_pattern(self):
        rules = [
            ToolPermission(name="execute", args={"command": "make *"}),
            ToolPermission(name="execute", mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"command": "make test"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "make build"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "npm install"}) == "deny"

    def test_missing_arg_treated_as_empty_string(self):
        rules = [ToolPermission(name="execute", args={"command": "pytest *"})]
        # arg "command" not present → coerced to "" → doesn't match "pytest *"
        assert _check_tool_permission(rules, "execute", {}) == "allow"  # falls through to default allow


class TestPermissionMiddleware:
    def _make_request(self, tool_name: str, args: dict, tool_call_id: str = "tc1"):
        return ToolCallRequest(
            runtime=_runtime(tool_call_id),
            tool_call={"id": tool_call_id, "name": tool_name, "args": args},
            state={},
            tool=None,
        )

    def test_allow_passes_through(self):
        middleware = PermissionMiddleware(
            rules=[
                ToolPermission(name="execute", args={"command": "pytest *"}),
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        request = self._make_request("execute", {"command": "pytest tests/"})
        expected = ToolMessage(content="ok", tool_call_id="tc1")

        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    def test_deny_returns_error_message(self):
        middleware = PermissionMiddleware(
            rules=[
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        request = self._make_request("execute", {"command": "rm -rf /"})

        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "denied" in result.content
        assert result.tool_call_id == "tc1"
        assert result.name == "execute"

    def test_unrelated_tool_allowed(self):
        middleware = PermissionMiddleware(
            rules=[
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        request = self._make_request("read_file", {"file_path": "/foo.txt"})
        expected = ToolMessage(content="content", tool_call_id="tc1")

        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    async def test_async_deny_returns_error_message(self):
        middleware = PermissionMiddleware(
            rules=[
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        request = self._make_request("execute", {"command": "rm -rf /"})

        async def async_handler(_):
            return ToolMessage(content="should not reach", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "denied" in result.content

    async def test_async_allow_passes_through(self):
        middleware = PermissionMiddleware(
            rules=[
                ToolPermission(name="execute", args={"command": "pytest *"}),
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        request = self._make_request("execute", {"command": "pytest tests/"})
        expected = ToolMessage(content="passed", tool_call_id="tc1")

        async def async_handler(_):
            return expected

        result = await middleware.awrap_tool_call(request, async_handler)
        assert result is expected


class TestPermissionMiddlewareFilesystem:
    """Tests for PermissionMiddleware enforcing FilesystemPermission rules via wrap_tool_call."""

    def _make_request(self, tool_name: str, args: dict, tool_call_id: str = "tc1"):
        return ToolCallRequest(
            runtime=_runtime(tool_call_id),
            tool_call={"id": tool_call_id, "name": tool_name, "args": args},
            state={},
            tool=None,
        )

    def test_read_denied_on_restricted_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request("read_file", {"file_path": "/secrets/key.txt"})
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "permission denied" in result.content
        assert "read" in result.content

    def test_read_allowed_on_permitted_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request("read_file", {"file_path": "/workspace/file.txt"})
        expected = ToolMessage(content="file content", tool_call_id="tc1")
        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    def test_write_denied_on_restricted_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        )
        request = self._make_request("write_file", {"file_path": "/foo.txt", "content": "data"})
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "permission denied" in result.content
        assert "write" in result.content

    def test_edit_denied_on_restricted_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["write"], paths=["/protected/**"], mode="deny")],
        )
        request = self._make_request("edit_file", {"file_path": "/protected/file.txt", "old_string": "a", "new_string": "b"})
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "permission denied" in result.content

    def test_ls_pre_check_denied(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets/", "/secrets"], mode="deny")],
        )
        request = self._make_request("ls", {"path": "/secrets"})
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "permission denied" in result.content

    def test_ls_post_filters_denied_children(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")],
        )
        request = self._make_request("ls", {"path": "/"})
        # Simulate ls returning an artifact with paths including denied ones
        ls_result = ToolMessage(
            content="['/public', '/secrets']",
            artifact=["/public", "/secrets"],
            tool_call_id="tc1",
            name="ls",
        )
        result = middleware.wrap_tool_call(request, lambda _: ls_result)
        assert isinstance(result, ToolMessage)
        assert "/secrets" not in result.content
        assert "/public" in result.content
        assert result.artifact == ["/public"]

    def test_ls_no_filter_when_write_only_deny(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        )
        request = self._make_request("ls", {"path": "/"})
        ls_result = ToolMessage(
            content="['/public']",
            artifact=["/public"],
            tool_call_id="tc1",
            name="ls",
        )
        result = middleware.wrap_tool_call(request, lambda _: ls_result)
        assert "/public" in result.content

    def test_deny_read_allows_write(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/vault/**"], mode="deny")],
        )
        request = self._make_request("write_file", {"file_path": "/vault/file.txt", "content": "data"})
        expected = ToolMessage(content="Updated file /vault/file.txt", tool_call_id="tc1")
        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    def test_glob_post_filters_denied_paths(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request("glob", {"pattern": "**/*.txt", "path": "/"})
        glob_result = ToolMessage(
            content="['/public/a.txt', '/secrets/b.txt']",
            artifact=["/public/a.txt", "/secrets/b.txt"],
            tool_call_id="tc1",
            name="glob",
        )
        result = middleware.wrap_tool_call(request, lambda _: glob_result)
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content
        assert result.artifact == ["/public/a.txt"]

    def test_grep_post_filters_denied_paths(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request("grep", {"pattern": "keyword"})
        matches = [
            {"path": "/public/a.txt", "line": 1, "text": "keyword here"},
            {"path": "/secrets/b.txt", "line": 1, "text": "keyword there"},
        ]
        grep_result = ToolMessage(
            content="/public/a.txt\n/secrets/b.txt",
            artifact=matches,
            tool_call_id="tc1",
            name="grep",
        )
        result = middleware.wrap_tool_call(request, lambda _: grep_result)
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content
        assert len(result.artifact) == 1

    def test_no_rules_allows_everything(self):
        middleware = PermissionMiddleware(rules=[])
        request = self._make_request("read_file", {"file_path": "/secrets/key.txt"})
        expected = ToolMessage(content="top secret", tool_call_id="tc1")
        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    def test_non_canonical_backend_path_bypasses_deny_rule(self):
        """_check_fs_permission alone does not canonicalize paths.

        A non-canonical path like '/secrets/./key.txt' won't match '/secrets/**'.
        In practice this is not exploitable because `validate_path` (called
        before every permission check) rejects `..` traversals and normalizes
        redundant separators. This test documents the raw matcher behavior.
        """
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        assert _check_fs_permission(rules, "read", "/secrets/key.txt") == "deny"
        assert _check_fs_permission(rules, "read", "/secrets/./key.txt") == "allow"

    def test_grep_path_none_skips_pre_check(self):
        """When grep path is None, pre-check is skipped but post-filter still applies."""
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request("grep", {"pattern": "keyword", "path": None})
        matches = [
            {"path": "/public/a.txt", "line": 1, "text": "keyword here"},
            {"path": "/secrets/b.txt", "line": 1, "text": "keyword there"},
        ]
        grep_result = ToolMessage(
            content="/public/a.txt\n/secrets/b.txt",
            artifact=matches,
            tool_call_id="tc1",
            name="grep",
        )
        result = middleware.wrap_tool_call(request, lambda _: grep_result)
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content


class TestCheckToolPermissionGlobbing:
    """Tests targeting specific glob pattern features in _check_tool_permission."""

    def test_star_crosses_slash_in_tool_args(self):
        # Tool argument values are matched with fnmatch semantics (no path separator rules),
        # so * crosses / — matching the design doc example of "pytest *" matching "pytest tests/".
        rules = [
            ToolPermission(name="execute", args={"command": "pytest *"}, mode="allow"),
            ToolPermission(name="execute", mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"command": "pytest tests/"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "pytest tests/unit/foo.py"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "rm -rf /"}) == "deny"

    def test_question_mark_matches_any_single_char_including_slash(self):
        # With fnmatch semantics, ? matches any single character including /.
        rules = [
            ToolPermission(name="execute", args={"command": "ls ?"}, mode="allow"),
            ToolPermission(name="execute", mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"command": "ls a"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "ls /"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "ls ab"}) == "deny"

    def test_brace_expansion(self):
        rules = [
            ToolPermission(name="execute", args={"command": "{pytest,make} *"}, mode="allow"),
            ToolPermission(name="execute", mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"command": "pytest tests/"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "make test"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "npm install"}) == "deny"

    def test_non_string_arg_coerced_to_str(self):
        rules = [
            ToolPermission(name="execute", args={"timeout": "3*"}, mode="deny"),
        ]
        # Integer 30 → "30" → matches "3*"
        assert _check_tool_permission(rules, "execute", {"timeout": 30}) == "deny"
        # Integer 60 → "60" → does not match "3*"
        assert _check_tool_permission(rules, "execute", {"timeout": 60}) == "allow"

    def test_bool_arg_coerced_to_str(self):
        rules = [
            ToolPermission(name="execute", args={"flag": "True"}, mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"flag": True}) == "deny"
        assert _check_tool_permission(rules, "execute", {"flag": False}) == "allow"

    def test_empty_string_arg_with_empty_pattern(self):
        rules = [ToolPermission(name="execute", args={"command": ""}, mode="deny")]
        # wcmatch.fnmatch("", "") returns False — empty pattern does not match
        # empty string, so the rule doesn't fire and falls through to default allow
        assert _check_tool_permission(rules, "execute", {"command": ""}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "anything"}) == "allow"

    def test_multi_rule_none_match_default_allow(self):
        rules = [
            ToolPermission(name="read_file", mode="deny"),
            ToolPermission(name="write_file", mode="deny"),
        ]
        # Neither rule names match "execute" → fall through to permissive default
        assert _check_tool_permission(rules, "execute", {"command": "rm -rf /"}) == "allow"


class TestCheckFsPermissionGlobbing:
    """Tests targeting specific glob pattern features in _check_fs_permission."""

    def test_question_mark_matches_single_char(self):
        rules = [FilesystemPermission(operations=["read"], paths=["/data/?"], mode="deny")]
        assert _check_fs_permission(rules, "read", "/data/a") == "deny"
        assert _check_fs_permission(rules, "read", "/data/ab") == "allow"

    def test_brace_expansion(self):
        rules = [FilesystemPermission(operations=["read"], paths=["/data/{a,b}.txt"], mode="deny")]
        assert _check_fs_permission(rules, "read", "/data/a.txt") == "deny"
        assert _check_fs_permission(rules, "read", "/data/b.txt") == "deny"
        assert _check_fs_permission(rules, "read", "/data/c.txt") == "allow"

    def test_multiple_paths_in_one_rule(self):
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/private/**"], mode="deny")]
        assert _check_fs_permission(rules, "read", "/secrets/key.txt") == "deny"
        assert _check_fs_permission(rules, "read", "/private/data.bin") == "deny"
        assert _check_fs_permission(rules, "read", "/public/readme.txt") == "allow"

    def test_operation_mismatch_skips_rule(self):
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        # Rule is write-only; read should not be affected
        assert _check_fs_permission(rules, "read", "/secrets/key.txt") == "allow"

    def test_first_matching_rule_wins(self):
        rules = [
            FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny"),
            FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="allow"),
        ]
        assert _check_fs_permission(rules, "read", "/secrets/key.txt") == "deny"

    def test_no_rules_returns_allow(self):
        assert _check_fs_permission([], "read", "/anything/goes.txt") == "allow"
        assert _check_fs_permission([], "write", "/anything/goes.txt") == "allow"

    def test_globstar_matches_deeply_nested_path(self):
        rules = [FilesystemPermission(operations=["read"], paths=["/vault/**"], mode="deny")]
        assert _check_fs_permission(rules, "read", "/vault/a/b/c/deep.txt") == "deny"
        assert _check_fs_permission(rules, "read", "/other/file.txt") == "allow"


class TestFilterPathsByPermission:
    """Tests for _filter_paths_by_permission post-filtering logic."""

    def test_empty_paths_returns_empty(self):
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        assert _filter_paths_by_permission(rules, "read", []) == []

    def test_no_rules_returns_all_paths(self):
        paths = ["/a/file.txt", "/b/file.txt", "/c/file.txt"]
        assert _filter_paths_by_permission([], "read", paths) == paths

    def test_denied_paths_removed_allowed_kept(self):
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        paths = ["/workspace/a.txt", "/secrets/key.txt", "/workspace/b.txt"]
        result = _filter_paths_by_permission(rules, "read", paths)
        assert "/secrets/key.txt" not in result
        assert "/workspace/a.txt" in result
        assert "/workspace/b.txt" in result

    def test_all_paths_allowed_when_rule_targets_different_op(self):
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        paths = ["/a.txt", "/b.txt"]
        # Rule is write-only; read filter passes all
        assert _filter_paths_by_permission(rules, "read", paths) == paths

    def test_all_paths_denied(self):
        rules = [FilesystemPermission(operations=["read"], paths=["/**"], mode="deny")]
        paths = ["/a.txt", "/b.txt", "/c.txt"]
        assert _filter_paths_by_permission(rules, "read", paths) == []

    def test_multiple_deny_patterns_filter_each(self):
        rules = [
            FilesystemPermission(operations=["read"], paths=["/secrets/**", "/private/**"], mode="deny"),
        ]
        paths = ["/secrets/a.txt", "/private/b.txt", "/public/c.txt"]
        assert _filter_paths_by_permission(rules, "read", paths) == ["/public/c.txt"]


class TestCanonicalizationBypass:
    """Tests verifying that path traversal bypasses are blocked by canonicalization."""

    def test_dotdot_traversal_blocked_by_validate_path(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = read_tool.invoke({"runtime": _runtime(), "file_path": "/workspace/../secrets/key.txt"})
        assert "Path traversal not allowed" in result

    def test_dotdot_traversal_blocked_even_without_permission_rules(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = read_tool.invoke({"runtime": _runtime(), "file_path": "/workspace/../secrets/key.txt"})
        assert "Path traversal not allowed" in result

    def test_redundant_separators_normalized_then_caught_by_permission(self):
        """validate_path normalizes then PermissionMiddleware catches the deny rule.

        /secrets//key.txt is normalized to /secrets/key.txt by validate_path,
        then PermissionMiddleware denies it.
        """
        perm = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = ToolCallRequest(
            runtime=_runtime("tc1"),
            tool_call={"id": "tc1", "name": "read_file", "args": {"file_path": "/secrets//key.txt"}},
            state={},
            tool=None,
        )
        result = perm.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert "permission denied" in result.content

    def test_dotdot_write_traversal_blocked_by_validate_path(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        result = write_tool.invoke({"runtime": _runtime(), "file_path": "/workspace/../restricted/file.txt", "content": "data"})
        assert "Path traversal not allowed" in result

    def test_non_traversal_path_still_allowed(self):
        backend = _make_backend({"/workspace/safe.txt": "safe content"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = read_tool.invoke({"runtime": _runtime(), "file_path": "/workspace/safe.txt"})
        assert "Path traversal" not in result

    def test_grep_dotdot_traversal_blocked_by_validate_path(self):
        """Grep rejects ../ traversal via validate_path in the tool itself."""
        backend = _make_backend({"/secrets/key.txt": "top secret data"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        result = grep_tool.invoke({"runtime": _runtime(), "pattern": "secret", "path": "/workspace/../secrets"})
        assert "Path traversal not allowed" in result


class TestGlobToolPermissions:
    """Tests for glob tool permission checks via PermissionMiddleware."""

    def _make_request(self, args: dict, tool_call_id: str = "tc1"):
        return ToolCallRequest(
            runtime=_runtime(tool_call_id),
            tool_call={"id": tool_call_id, "name": "glob", "args": args},
            state={},
            tool=None,
        )

    def test_glob_denied_on_restricted_base_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")],
        )
        request = self._make_request({"pattern": "*.txt", "path": "/secrets"})
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="x", tool_call_id="tc1"))
        assert "permission denied" in result.content

    def test_glob_allowed_on_unrestricted_base_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request({"pattern": "*.txt", "path": "/workspace"})
        expected = ToolMessage(content="['/workspace/file.txt']", artifact=["/workspace/file.txt"], tool_call_id="tc1", name="glob")
        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert "permission denied" not in result.content

    def test_glob_post_filters_denied_results(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request({"pattern": "**/*.txt", "path": "/"})
        glob_result = ToolMessage(
            content="['/public/a.txt', '/secrets/b.txt']",
            artifact=["/public/a.txt", "/secrets/b.txt"],
            tool_call_id="tc1",
            name="glob",
        )
        result = middleware.wrap_tool_call(request, lambda _: glob_result)
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content

    async def test_glob_denied_on_restricted_base_path_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")],
        )
        request = self._make_request({"pattern": "*.txt", "path": "/secrets"})

        async def handler(_):
            return ToolMessage(content="x", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, handler)
        assert "permission denied" in result.content

    async def test_glob_post_filters_denied_results_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request({"pattern": "**/*.txt", "path": "/"})
        glob_result = ToolMessage(
            content="['/public/a.txt', '/secrets/b.txt']",
            artifact=["/public/a.txt", "/secrets/b.txt"],
            tool_call_id="tc1",
            name="glob",
        )

        async def handler(_):
            return glob_result

        result = await middleware.awrap_tool_call(request, handler)
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content


class TestGrepToolPermissions:
    """Tests for grep tool permission checks via PermissionMiddleware."""

    def _make_request(self, args: dict, tool_call_id: str = "tc1"):
        return ToolCallRequest(
            runtime=_runtime(tool_call_id),
            tool_call={"id": tool_call_id, "name": "grep", "args": args},
            state={},
            tool=None,
        )

    def _grep_result(self, matches, tool_call_id="tc1"):
        return ToolMessage(
            content="\n".join(m["path"] for m in matches),
            artifact=matches,
            tool_call_id=tool_call_id,
            name="grep",
        )

    def test_grep_denied_on_restricted_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")],
        )
        request = self._make_request({"pattern": "secret", "path": "/secrets"})
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="x", tool_call_id="tc1"))
        assert "permission denied" in result.content

    def test_grep_allowed_on_unrestricted_path(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request({"pattern": "hello", "path": "/workspace"})
        matches = [{"path": "/workspace/file.txt", "line": 1, "text": "hello world"}]
        result = middleware.wrap_tool_call(request, lambda _: self._grep_result(matches))
        assert "permission denied" not in result.content

    def test_grep_post_filters_denied_results(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request({"pattern": "keyword"})
        matches = [
            {"path": "/public/a.txt", "line": 1, "text": "keyword here"},
            {"path": "/secrets/b.txt", "line": 1, "text": "keyword there"},
        ]
        result = middleware.wrap_tool_call(request, lambda _: self._grep_result(matches))
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content

    def test_grep_path_none_skips_pre_check_but_filters_results(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request({"pattern": "keyword", "path": None})
        matches = [
            {"path": "/public/a.txt", "line": 1, "text": "keyword here"},
            {"path": "/secrets/b.txt", "line": 1, "text": "keyword there"},
        ]
        result = middleware.wrap_tool_call(request, lambda _: self._grep_result(matches))
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content

    async def test_grep_denied_on_restricted_path_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")],
        )
        request = self._make_request({"pattern": "secret", "path": "/secrets"})

        async def handler(_):
            return ToolMessage(content="x", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, handler)
        assert "permission denied" in result.content

    async def test_grep_post_filters_denied_results_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request({"pattern": "keyword"})
        matches = [
            {"path": "/public/a.txt", "line": 1, "text": "keyword here"},
            {"path": "/secrets/b.txt", "line": 1, "text": "keyword there"},
        ]

        async def handler(_):
            return self._grep_result(matches)

        result = await middleware.awrap_tool_call(request, handler)
        assert "/secrets/b.txt" not in result.content
        assert "/public/a.txt" in result.content


class TestExecuteToolPermissions:
    """Tests for ToolPermission rules targeting the execute tool via PermissionMiddleware."""

    def _make_sandbox_backend(self) -> SandboxBackendProtocol:
        class MockSandbox(SandboxBackendProtocol, StoreBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output=f"ran: {command}", exit_code=0, truncated=False)

            async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
                return ExecuteResponse(output=f"ran: {command}", exit_code=0, truncated=False)

            @property
            def id(self) -> str:
                return "mock-sandbox"

        mem_store = InMemoryStore()
        return MockSandbox(store=mem_store, namespace=lambda _ctx: ("filesystem",))

    def test_execute_denied_by_tool_permission(self):
        rules = [ToolPermission(name="execute", mode="deny")]
        assert _check_tool_permission(rules, "execute", {"command": "rm -rf /"}) == "deny"

    def test_execute_allowed_by_arg_pattern(self):
        rules = [
            ToolPermission(name="execute", args={"command": "pytest *"}),
            ToolPermission(name="execute", mode="deny"),
        ]
        assert _check_tool_permission(rules, "execute", {"command": "pytest tests/"}) == "allow"
        assert _check_tool_permission(rules, "execute", {"command": "rm -rf /"}) == "deny"

    def test_execute_middleware_denies(self):
        middleware = PermissionMiddleware(rules=[ToolPermission(name="execute", mode="deny")])
        request = ToolCallRequest(
            runtime=_runtime("tc1"),
            tool_call={"id": "tc1", "name": "execute", "args": {"command": "rm -rf /"}},
            state={},
            tool=None,
        )
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "denied" in result.content

    def test_execute_middleware_allows_matching_pattern(self):
        middleware = PermissionMiddleware(
            rules=[
                ToolPermission(name="execute", args={"command": "pytest *"}),
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        request = ToolCallRequest(
            runtime=_runtime("tc1"),
            tool_call={"id": "tc1", "name": "execute", "args": {"command": "pytest tests/unit"}},
            state={},
            tool=None,
        )
        expected = ToolMessage(content="ok", tool_call_id="tc1")
        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    def test_execute_tool_invocation_denied_via_middleware(self):
        """Full integration: execute tool on a sandbox backend, denied by PermissionMiddleware."""
        backend = self._make_sandbox_backend()
        middleware = PermissionMiddleware(rules=[ToolPermission(name="execute", mode="deny")])
        fs_middleware = FilesystemMiddleware(backend=backend)
        execute_tool = next(t for t in fs_middleware.tools if t.name == "execute")

        request = ToolCallRequest(
            runtime=_runtime("tc1"),
            tool_call={"id": "tc1", "name": "execute", "args": {"command": "echo hello"}},
            state={},
            tool=execute_tool,
        )
        result = middleware.wrap_tool_call(
            request,
            lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"),
        )
        assert isinstance(result, ToolMessage)
        assert "denied" in result.content

    async def test_execute_middleware_denies_async(self):
        middleware = PermissionMiddleware(rules=[ToolPermission(name="execute", mode="deny")])
        request = ToolCallRequest(
            runtime=_runtime("tc1"),
            tool_call={"id": "tc1", "name": "execute", "args": {"command": "rm -rf /"}},
            state={},
            tool=None,
        )

        async def async_handler(_):
            return ToolMessage(content="should not reach", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "denied" in result.content


class TestAsyncPermissionMiddlewareFilesystem:
    """Async variants of PermissionMiddleware filesystem permission checks."""

    def _make_request(self, tool_name: str, args: dict, tool_call_id: str = "tc1"):
        return ToolCallRequest(
            runtime=_runtime(tool_call_id),
            tool_call={"id": tool_call_id, "name": tool_name, "args": args},
            state={},
            tool=None,
        )

    async def test_read_denied_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request("read_file", {"file_path": "/secrets/key.txt"})

        async def handler(_):
            return ToolMessage(content="should not reach", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, handler)
        assert "permission denied" in result.content
        assert "read" in result.content

    async def test_read_allowed_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        request = self._make_request("read_file", {"file_path": "/workspace/file.txt"})
        expected = ToolMessage(content="file content", tool_call_id="tc1")

        async def handler(_):
            return expected

        result = await middleware.awrap_tool_call(request, handler)
        assert result is expected

    async def test_write_denied_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        )
        request = self._make_request("write_file", {"file_path": "/foo.txt", "content": "data"})

        async def handler(_):
            return ToolMessage(content="should not reach", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, handler)
        assert "permission denied" in result.content

    async def test_ls_denied_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")],
        )
        request = self._make_request("ls", {"path": "/secrets"})

        async def handler(_):
            return ToolMessage(content="should not reach", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, handler)
        assert "permission denied" in result.content

    async def test_ls_post_filters_denied_results_async(self):
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")],
        )
        request = self._make_request("ls", {"path": "/"})
        ls_result = ToolMessage(
            content="['/public', '/secrets']",
            artifact=["/public", "/secrets"],
            tool_call_id="tc1",
            name="ls",
        )

        async def handler(_):
            return ls_result

        result = await middleware.awrap_tool_call(request, handler)
        assert "/secrets" not in result.content
        assert "/public" in result.content


class TestToolPermissionOnFilesystemTools:
    """Tests for ToolPermission rules targeting filesystem tool names.

    The design doc states: '`ToolPermission` also works on filesystem tools
    (e.g., `ToolPermission(name="read_file", mode="deny")`)'.
    """

    def test_tool_permission_denies_read_file_by_name(self):
        rules = [ToolPermission(name="read_file", mode="deny")]
        assert _check_tool_permission(rules, "read_file", {"file_path": "/foo.txt"}) == "deny"

    def test_tool_permission_denies_write_file_by_name(self):
        rules = [ToolPermission(name="write_file", mode="deny")]
        assert _check_tool_permission(rules, "write_file", {"file_path": "/foo.txt", "content": "x"}) == "deny"

    def test_tool_permission_on_fs_tool_with_arg_pattern(self):
        rules = [
            ToolPermission(name="read_file", args={"file_path": "/secrets/*"}, mode="deny"),
        ]
        assert _check_tool_permission(rules, "read_file", {"file_path": "/secrets/key.txt"}) == "deny"
        assert _check_tool_permission(rules, "read_file", {"file_path": "/workspace/file.txt"}) == "allow"

    def test_tool_permission_does_not_affect_other_fs_tools(self):
        rules = [ToolPermission(name="read_file", mode="deny")]
        assert _check_tool_permission(rules, "write_file", {"file_path": "/foo.txt"}) == "allow"
        assert _check_tool_permission(rules, "ls", {"path": "/"}) == "allow"
        assert _check_tool_permission(rules, "glob", {"pattern": "**"}) == "allow"

    def test_tool_permission_middleware_denies_read_file(self):
        middleware = PermissionMiddleware(rules=[ToolPermission(name="read_file", mode="deny")])
        request = ToolCallRequest(
            runtime=_runtime("tc1"),
            tool_call={"id": "tc1", "name": "read_file", "args": {"file_path": "/foo.txt"}},
            state={},
            tool=None,
        )
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "denied" in result.content
