"""Unit tests for FilesystemPermission, ToolPermission, and PermissionMiddleware."""

from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import StoreBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.permissions import PermissionMiddleware, _check_fs_permission, _check_tool_permission, _filter_paths_by_permission
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


def _invoke_with_permissions(tool, args, rules, tool_call_id="test"):
    """Invoke a tool through PermissionMiddleware, return the content string."""
    perm = PermissionMiddleware(rules=rules)
    runtime = _runtime(tool_call_id)

    def handler(_req):
        raw = tool.invoke({**args, "runtime": runtime})
        if isinstance(raw, ToolMessage):
            return raw
        return ToolMessage(content=str(raw), tool_call_id=tool_call_id, name=tool.name)

    request = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": tool_call_id, "name": tool.name, "args": args},
        state={},
        tool=tool,
    )
    result = perm.wrap_tool_call(request, handler)
    if isinstance(result, ToolMessage):
        return result.content
    return str(result)


async def _ainvoke_with_permissions(tool, args, rules, tool_call_id="test"):
    """Async version of _invoke_with_permissions."""
    perm = PermissionMiddleware(rules=rules)
    runtime = _runtime(tool_call_id)

    async def handler(_req):
        raw = await tool.ainvoke({**args, "runtime": runtime})
        if isinstance(raw, ToolMessage):
            return raw
        return ToolMessage(content=str(raw), tool_call_id=tool_call_id, name=tool.name)

    request = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": tool_call_id, "name": tool.name, "args": args},
        state={},
        tool=tool,
    )
    result = await perm.awrap_tool_call(request, handler)
    if isinstance(result, ToolMessage):
        return result.content
    return str(result)


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


class TestFilesystemMiddlewarePermissions:
    def test_read_denied_on_restricted_path(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(read_tool, {"file_path": "/secrets/key.txt"}, rules)
        assert "permission denied" in result
        assert "read" in result

    def test_read_allowed_on_permitted_path(self):
        backend = _make_backend({"/workspace/file.txt": "hello"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(read_tool, {"file_path": "/workspace/file.txt"}, rules)
        assert "permission denied" not in result

    def test_write_denied_on_restricted_path(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(write_tool, {"file_path": "/foo.txt", "content": "data"}, rules)
        assert "permission denied" in result
        assert "write" in result

    def test_edit_denied_on_restricted_path(self):
        backend = _make_backend({"/protected/file.txt": "original"})
        middleware = FilesystemMiddleware(backend=backend)
        edit_tool = next(t for t in middleware.tools if t.name == "edit_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/protected/**"], mode="deny")]
        result = _invoke_with_permissions(
            edit_tool,
            {
                "file_path": "/protected/file.txt",
                "old_string": "original",
                "new_string": "changed",
            },
            rules,
        )
        assert "permission denied" in result

    def test_ls_filters_denied_results(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        # Deny the /secrets/ directory entry itself so it's filtered from ls output
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets/", "/secrets"], mode="deny")]
        # ls /secrets directly should be denied (pre-check on the queried path)
        result_secrets = _invoke_with_permissions(ls_tool, {"path": "/secrets"}, rules)
        assert "permission denied" in result_secrets

    def test_ls_no_filter_when_all_allowed(self):
        backend = _make_backend({"/public/a.txt": "pub", "/public/b.txt": "pub2"})
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(ls_tool, {"path": "/"}, rules)
        assert "/public" in result

    def test_no_rules_allows_everything(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = read_tool.invoke({"runtime": _runtime(), "file_path": "/secrets/key.txt"})
        assert "permission denied" not in result

    def test_ls_denied_on_restricted_root(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(ls_tool, {"path": "/secrets"}, rules)
        assert "permission denied" in result

    def test_ls_post_filters_denied_children(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(ls_tool, {"path": "/"}, rules)
        assert "/secrets" not in result
        assert "/public" in result

    def test_deny_read_allows_write(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/vault/**"], mode="deny")]
        result = _invoke_with_permissions(write_tool, {"file_path": "/vault/file.txt", "content": "data"}, rules)
        assert "permission denied" not in result

    def test_non_canonical_backend_path_bypasses_deny_rule(self):
        """_check_fs_permission alone does not canonicalize paths.

        A non-canonical path like '/secrets/./key.txt' won't match '/secrets/**'.
        In practice this is not exploitable because `validate_path` (called
        before every permission check) rejects `..` traversals and normalizes
        redundant separators. This test documents the raw matcher behavior.
        """
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        # A canonical path is correctly denied
        assert _check_fs_permission(rules, "read", "/secrets/key.txt") == "deny"
        # A non-canonical path that resolves to the same file is NOT denied — this is the gap
        assert _check_fs_permission(rules, "read", "/secrets/./key.txt") == "allow"


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
        # validate_path rejects .. before permission checking even runs,
        # so traversal is blocked regardless of permission rules.
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = _invoke_with_permissions(read_tool, {"file_path": "/workspace/../secrets/key.txt"}, rules)
        assert "Path traversal not allowed" in result

    def test_dotdot_traversal_blocked_even_without_permission_rules(self):
        # Traversal is rejected by validate_path even when no permission rules are set.
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = read_tool.invoke({"runtime": _runtime(), "file_path": "/workspace/../secrets/key.txt"})
        assert "Path traversal not allowed" in result

    def test_redundant_separators_normalized(self):
        # /secrets//key.txt is normalized by validate_path to /secrets/key.txt
        # and then caught by the permission rule.
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = _invoke_with_permissions(read_tool, {"file_path": "/secrets//key.txt"}, rules)
        assert "permission denied" in result

    def test_dotdot_write_traversal_blocked_by_validate_path(self):
        # validate_path rejects .. on write paths too.
        rules = [FilesystemPermission(operations=["write"], paths=["/restricted/**"], mode="deny")]
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        result = _invoke_with_permissions(write_tool, {"file_path": "/workspace/../restricted/file.txt", "content": "data"}, rules)
        assert "Path traversal not allowed" in result

    def test_non_traversal_path_still_allowed(self):
        # Verify that normal paths are not affected by the canonicalization logic.
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        backend = _make_backend({"/workspace/safe.txt": "safe content"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = _invoke_with_permissions(read_tool, {"file_path": "/workspace/safe.txt"}, rules)
        assert "permission denied" not in result
        assert "Path traversal" not in result


class TestGlobToolPermissions:
    """Tests for the glob tool permission checks in FilesystemMiddleware."""

    def test_glob_denied_on_restricted_base_path(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "*.txt", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    def test_glob_allowed_on_unrestricted_base_path(self):
        backend = _make_backend({"/workspace/file.txt": "hello"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "*.txt", "path": "/workspace"}, rules)
        assert "permission denied" not in result

    def test_glob_filters_denied_results(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "**/*.txt", "path": "/"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    def test_glob_no_filter_annotation_when_all_allowed(self):
        backend = _make_backend({"/public/a.txt": "pub", "/public/b.txt": "pub2"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "**/*.txt", "path": "/"}, rules)
        assert "permission denied" not in result

    async def test_glob_denied_on_restricted_base_path_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = await _ainvoke_with_permissions(glob_tool, {"pattern": "*.txt", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    async def test_glob_filters_denied_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(glob_tool, {"pattern": "**/*.txt", "path": "/"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result


class TestGrepToolPermissions:
    """Tests for the grep tool permission checks in FilesystemMiddleware."""

    def test_grep_denied_on_restricted_path(self):
        backend = _make_backend({"/secrets/key.txt": "top secret data"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "secret", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    def test_grep_dotdot_traversal_blocked_by_validate_path(self):
        """Grep rejects ../ traversal via validate_path before the permission check runs."""
        backend = _make_backend({"/secrets/key.txt": "top secret data"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "secret", "path": "/workspace/../secrets"}, rules)
        assert "Path traversal not allowed" in result

    def test_grep_allowed_on_unrestricted_path(self):
        backend = _make_backend({"/workspace/file.txt": "hello world"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "hello", "path": "/workspace"}, rules)
        assert "permission denied" not in result

    def test_grep_filters_denied_results_from_matches(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "keyword"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    def test_grep_no_filter_annotation_when_all_allowed(self):
        backend = _make_backend({"/public/a.txt": "keyword", "/public/b.txt": "keyword"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "keyword"}, rules)
        assert "permission denied" not in result

    def test_grep_path_none_bypasses_pre_check_but_filters_results(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "keyword", "path": None}, rules)
        assert "permission denied" not in result
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    async def test_grep_denied_on_restricted_path_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret data"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = await _ainvoke_with_permissions(grep_tool, {"pattern": "secret", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    async def test_grep_filters_denied_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(grep_tool, {"pattern": "keyword"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    async def test_grep_path_none_bypasses_pre_check_but_filters_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(grep_tool, {"pattern": "keyword", "path": None}, rules)
        assert "permission denied" not in result
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result


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


class TestAsyncFilesystemMiddlewarePermissions:
    """Async variants of the core filesystem tool permission checks (read, write, edit, ls)."""

    async def test_read_denied_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(read_tool, {"file_path": "/secrets/key.txt"}, rules)
        assert "permission denied" in result
        assert "read" in result

    async def test_read_allowed_async(self):
        backend = _make_backend({"/workspace/file.txt": "hello"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(read_tool, {"file_path": "/workspace/file.txt"}, rules)
        assert "permission denied" not in result

    async def test_write_denied_async(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = await _ainvoke_with_permissions(write_tool, {"file_path": "/foo.txt", "content": "data"}, rules)
        assert "permission denied" in result
        assert "write" in result

    async def test_write_allowed_async(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(write_tool, {"file_path": "/workspace/file.txt", "content": "data"}, rules)
        assert "permission denied" not in result

    async def test_edit_denied_async(self):
        backend = _make_backend({"/protected/file.txt": "original"})
        middleware = FilesystemMiddleware(backend=backend)
        edit_tool = next(t for t in middleware.tools if t.name == "edit_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/protected/**"], mode="deny")]
        result = await _ainvoke_with_permissions(
            edit_tool,
            {
                "file_path": "/protected/file.txt",
                "old_string": "original",
                "new_string": "changed",
            },
            rules,
        )
        assert "permission denied" in result

    async def test_ls_denied_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = await _ainvoke_with_permissions(ls_tool, {"path": "/secrets"}, rules)
        assert "permission denied" in result

    async def test_ls_filters_denied_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(ls_tool, {"path": "/"}, rules)
        assert "/secrets/b.txt" not in result


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
