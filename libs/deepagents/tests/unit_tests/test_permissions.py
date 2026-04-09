"""Unit tests for FilesystemPermission and PermissionMiddleware."""

from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.permissions import PermissionMiddleware, _check_fs_permission, _filter_paths_by_permission
from deepagents.permissions import FilesystemPermission


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


def _invoke_with_permissions(tool, args, rules, tool_call_id="test", backend=None):
    """Invoke a tool through PermissionMiddleware, return the content string."""
    perm = PermissionMiddleware(rules=rules, backend=backend or _make_backend())
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


async def _ainvoke_with_permissions(tool, args, rules, tool_call_id="test", backend=None):
    """Async version of _invoke_with_permissions."""
    perm = PermissionMiddleware(rules=rules, backend=backend or _make_backend())
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


class TestPermissionMiddleware:
    def _backend(self):
        return _make_backend()

    def _make_request(self, tool_name: str, args: dict, tool_call_id: str = "tc1"):
        return ToolCallRequest(
            runtime=_runtime(tool_call_id),
            tool_call={"id": tool_call_id, "name": tool_name, "args": args},
            state={},
            tool=None,
        )

    def test_allow_passes_through(self):
        middleware = PermissionMiddleware(rules=[FilesystemPermission(operations=["read"], paths=["/workspace/**"])], backend=self._backend())
        request = self._make_request("read_file", {"file_path": "/workspace/file.txt"})
        expected = ToolMessage(content="ok", tool_call_id="tc1")

        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    def test_deny_returns_error_message(self):
        middleware = PermissionMiddleware(rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")], backend=self._backend())
        request = self._make_request("read_file", {"file_path": "/secrets/key.txt"})

        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="should not reach", tool_call_id="tc1"))
        assert isinstance(result, ToolMessage)
        assert "permission denied" in result.content
        assert result.tool_call_id == "tc1"
        assert result.name == "read_file"

    def test_unrelated_tool_allowed(self):
        middleware = PermissionMiddleware(rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")], backend=self._backend())
        request = self._make_request("some_other_tool", {"input": "hello"})
        expected = ToolMessage(content="content", tool_call_id="tc1")

        result = middleware.wrap_tool_call(request, lambda _: expected)
        assert result is expected

    async def test_async_deny_returns_error_message(self):
        middleware = PermissionMiddleware(rules=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")], backend=self._backend())
        request = self._make_request("write_file", {"file_path": "/foo.txt", "content": "data"})

        async def async_handler(_):
            return ToolMessage(content="should not reach", tool_call_id="tc1")

        result = await middleware.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "permission denied" in result.content

    async def test_async_allow_passes_through(self):
        middleware = PermissionMiddleware(rules=[FilesystemPermission(operations=["read"], paths=["/workspace/**"])], backend=self._backend())
        request = self._make_request("read_file", {"file_path": "/workspace/file.txt"})
        expected = ToolMessage(content="passed", tool_call_id="tc1")

        async def async_handler(_):
            return expected

        result = await middleware.awrap_tool_call(request, async_handler)
        assert result is expected

    def test_raises_not_implemented_for_sandbox_backend(self):
        """PermissionMiddleware rejects backends that support execution."""
        from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

        class MockSandbox(SandboxBackendProtocol, StoreBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="", exit_code=0, truncated=False)

            async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
                return ExecuteResponse(output="", exit_code=0, truncated=False)

            @property
            def id(self) -> str:
                return "mock"

        mem_store = InMemoryStore()
        sandbox = MockSandbox(store=mem_store, namespace=lambda _ctx: ("filesystem",))
        import pytest

        with pytest.raises(NotImplementedError, match="execute"):
            PermissionMiddleware(
                rules=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
                backend=sandbox,
            )

    def test_raises_not_implemented_for_composite_with_sandbox_default(self):
        """PermissionMiddleware rejects CompositeBackend whose default supports execution."""
        from deepagents.backends.composite import CompositeBackend
        from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

        class MockSandbox(SandboxBackendProtocol, StoreBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="", exit_code=0, truncated=False)

            async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
                return ExecuteResponse(output="", exit_code=0, truncated=False)

            @property
            def id(self) -> str:
                return "mock"

        mem_store = InMemoryStore()
        sandbox = MockSandbox(store=mem_store, namespace=lambda _ctx: ("filesystem",))
        composite = CompositeBackend(default=sandbox, routes={})
        import pytest

        with pytest.raises(NotImplementedError, match="execute"):
            PermissionMiddleware(
                rules=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
                backend=composite,
            )

    def test_allows_composite_without_sandbox_default(self):
        """PermissionMiddleware accepts CompositeBackend whose default does not support execution."""
        from deepagents.backends.composite import CompositeBackend

        composite = CompositeBackend(default=self._backend(), routes={})
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
            backend=composite,
        )
        assert middleware._fs_rules

    def test_allows_composite_with_sandbox_route_but_non_sandbox_default(self):
        """CompositeBackend with sandbox in a route but non-sandbox default is allowed.

        Execution is only delegated to the default backend in CompositeBackend,
        so a sandbox in a route doesn't expose execution capability.
        """
        from deepagents.backends.composite import CompositeBackend
        from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

        class MockSandbox(SandboxBackendProtocol, StoreBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="", exit_code=0, truncated=False)

            async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
                return ExecuteResponse(output="", exit_code=0, truncated=False)

            @property
            def id(self) -> str:
                return "mock"

        mem_store = InMemoryStore()
        sandbox = MockSandbox(store=mem_store, namespace=lambda _ctx: ("filesystem",))
        composite = CompositeBackend(default=self._backend(), routes={"/sandbox/": sandbox})
        middleware = PermissionMiddleware(
            rules=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
            backend=composite,
        )
        assert middleware._fs_rules


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
