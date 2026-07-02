"""Unit tests for filesystem permission enforcement in `FilesystemMiddleware`."""

import threading

import pytest
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import StateBackend, StoreBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import EditResult, ExecuteResponse, GlobResult, ReadResult, SandboxBackendProtocol, WriteResult
from deepagents.backends.utils import _glob_anchor, _paths_overlap
from deepagents.graph import create_deep_agent
from deepagents.middleware import filesystem as filesystem_module
from deepagents.middleware._fs_interrupt import _build_interrupt_on_from_permissions, _make_fs_when_predicate
from deepagents.middleware.filesystem import (
    FilesystemMiddleware,
    FilesystemPermission,
    _all_paths_scoped_to_routes,
    _check_fs_permission,
    _filter_paths_by_permission,
    _find_delete_deny_patterns,
)
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT


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
    """Invoke a FilesystemMiddleware tool configured with permissions."""
    resolved_backend = backend
    if resolved_backend is None:
        parent = getattr(tool, "func", None)
        if parent is not None:
            closure = getattr(parent, "__closure__", None) or ()
            for cell in closure:
                candidate = getattr(cell, "cell_contents", None)
                if isinstance(candidate, FilesystemMiddleware):
                    resolved_backend = candidate.backend
                    break
    if resolved_backend is None:
        resolved_backend = _make_backend()
    configured_middleware = FilesystemMiddleware(backend=resolved_backend, _permissions=rules)
    configured_tool = next(t for t in configured_middleware.tools if t.name == tool.name)
    runtime = _runtime(tool_call_id)

    def handler(_req):
        raw = configured_tool.invoke({**args, "runtime": runtime})
        if isinstance(raw, ToolMessage):
            return raw
        return ToolMessage(content=str(raw), tool_call_id=tool_call_id, name=configured_tool.name)

    request = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": tool_call_id, "name": configured_tool.name, "args": args},
        state={},
        tool=configured_tool,
    )
    result = configured_middleware.wrap_tool_call(request, handler)
    if isinstance(result, ToolMessage):
        return result.content
    return str(result)


async def _ainvoke_with_permissions(tool, args, rules, tool_call_id="test", backend=None):
    """Async version of _invoke_with_permissions."""
    resolved_backend = backend
    if resolved_backend is None:
        parent = getattr(tool, "func", None)
        if parent is not None:
            closure = getattr(parent, "__closure__", None) or ()
            for cell in closure:
                candidate = getattr(cell, "cell_contents", None)
                if isinstance(candidate, FilesystemMiddleware):
                    resolved_backend = candidate.backend
                    break
    if resolved_backend is None:
        resolved_backend = _make_backend()
    configured_middleware = FilesystemMiddleware(backend=resolved_backend, _permissions=rules)
    configured_tool = next(t for t in configured_middleware.tools if t.name == tool.name)
    runtime = _runtime(tool_call_id)

    async def handler(_req):
        raw = await configured_tool.ainvoke({**args, "runtime": runtime})
        if isinstance(raw, ToolMessage):
            return raw
        return ToolMessage(content=str(raw), tool_call_id=tool_call_id, name=configured_tool.name)

    request = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": tool_call_id, "name": configured_tool.name, "args": args},
        state={},
        tool=configured_tool,
    )
    result = await configured_middleware.awrap_tool_call(request, handler)
    if isinstance(result, ToolMessage):
        return result.content
    return str(result)


class TestRecursiveDeletePermissions:
    """All-or-nothing write-permission enforcement for recursive delete.

    A recursive delete is refused if any ``deny`` write rule's pattern overlaps
    the target subtree, in either direction: a rule scoped inside the target
    (``/work/secrets/**``) and a rule the target falls under (``/work``) both
    block it. The check is rule-based (no filesystem enumeration).
    """

    def _fs_backend(self, tmp_path):
        # /work/a.txt, /work/logs/run.log, /work/secrets/{key,token}.txt
        (tmp_path / "work" / "logs").mkdir(parents=True)
        (tmp_path / "work" / "secrets").mkdir(parents=True)
        (tmp_path / "work" / "a.txt").write_text("a")
        (tmp_path / "work" / "logs" / "run.log").write_text("log")
        (tmp_path / "work" / "secrets" / "key.txt").write_text("k")
        (tmp_path / "work" / "secrets" / "token.txt").write_text("t")
        return FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

    def test_recursive_delete_refused_when_descendant_denied(self, tmp_path):
        backend = self._fs_backend(tmp_path)
        rules = [FilesystemPermission(operations=["write"], paths=["/work/secrets/**"], mode="deny")]
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = _invoke_with_permissions(tool, {"file_path": "/work"}, rules, backend=backend)

        assert "permission denied for write" in result
        assert "/work/secrets" in result
        # All-or-nothing: nothing was removed.
        assert (tmp_path / "work" / "a.txt").exists()
        assert (tmp_path / "work" / "secrets" / "key.txt").exists()

    def test_recursive_delete_allowed_when_whole_subtree_allowed(self, tmp_path):
        backend = self._fs_backend(tmp_path)
        rules = [FilesystemPermission(operations=["write"], paths=["/other/**"], mode="deny")]
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = _invoke_with_permissions(tool, {"file_path": "/work"}, rules, backend=backend)

        assert "Deleted" in result
        assert not (tmp_path / "work").exists()

    def test_delete_denied_directory_path_itself(self, tmp_path):
        # A rule on the bare directory path (no /**) still blocks deleting it.
        backend = self._fs_backend(tmp_path)
        rules = [FilesystemPermission(operations=["write"], paths=["/work/secrets"], mode="deny")]
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = _invoke_with_permissions(tool, {"file_path": "/work"}, rules, backend=backend)

        assert "permission denied for write" in result
        assert (tmp_path / "work" / "secrets" / "key.txt").exists()

    def test_delete_refused_when_ancestor_denied(self, tmp_path):
        # Any overlap denies: a rule on an ancestor of the target also blocks,
        # since the target subtree falls under it.
        backend = self._fs_backend(tmp_path)
        rules = [FilesystemPermission(operations=["write"], paths=["/work/**"], mode="deny")]
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = _invoke_with_permissions(tool, {"file_path": "/work/logs"}, rules, backend=backend)

        assert "permission denied for write" in result
        assert (tmp_path / "work" / "logs" / "run.log").exists()

    def test_delete_unaffected_sibling_subtree(self, tmp_path):
        backend = self._fs_backend(tmp_path)
        rules = [FilesystemPermission(operations=["write"], paths=["/work/secrets", "/work/secrets/**"], mode="deny")]
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = _invoke_with_permissions(tool, {"file_path": "/work/logs"}, rules, backend=backend)

        assert "Deleted" in result
        assert not (tmp_path / "work" / "logs").exists()
        assert (tmp_path / "work" / "secrets" / "key.txt").exists()

    def test_delete_reports_multiple_overlapping_patterns(self, tmp_path):
        # The agent gets context: several overlapping deny patterns are reported.
        backend = self._fs_backend(tmp_path)
        rules = [
            FilesystemPermission(operations=["write"], paths=["/work/secrets/**"], mode="deny"),
            FilesystemPermission(operations=["write"], paths=["/work/logs/**"], mode="deny"),
        ]
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = _invoke_with_permissions(tool, {"file_path": "/work"}, rules, backend=backend)

        assert "/work/secrets/**" in result
        assert "/work/logs/**" in result
        assert (tmp_path / "work" / "a.txt").exists()

    def test_single_file_delete_still_works(self, tmp_path):
        backend = self._fs_backend(tmp_path)
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = _invoke_with_permissions(tool, {"file_path": "/work/a.txt"}, [], backend=backend)

        assert "Deleted" in result
        assert not (tmp_path / "work" / "a.txt").exists()

    async def test_recursive_delete_refused_async(self, tmp_path):
        backend = self._fs_backend(tmp_path)
        rules = [FilesystemPermission(operations=["write"], paths=["/work/secrets/**"], mode="deny")]
        tool = next(t for t in FilesystemMiddleware(backend=backend).tools if t.name == "delete")

        result = await _ainvoke_with_permissions(tool, {"file_path": "/work"}, rules, backend=backend)

        assert "permission denied for write" in result
        assert (tmp_path / "work" / "secrets" / "key.txt").exists()

    def test_delete_registered_as_bulk_interrupt(self):
        # An interrupt rule on a descendant must fire for a recursive delete,
        # so delete is registered (bulk scope) in the interrupt builder.
        rules = [FilesystemPermission(operations=["write"], paths=["/work/secrets/**"], mode="interrupt")]
        out = _build_interrupt_on_from_permissions(rules)
        assert "delete" in out


def _deny(*paths: str) -> FilesystemPermission:
    return FilesystemPermission(operations=["write"], paths=list(paths), mode="deny")


class TestFindDeleteDenyPatterns:
    """Exhaustive unit tests for the pure `_find_delete_deny_patterns` helper.

    Returns every `deny` write pattern whose glob overlaps the delete subtree
    (target + descendants). An empty result means the delete is permitted.
    """

    @pytest.mark.parametrize(
        ("pattern", "target", "expected"),
        [
            # --- no overlap -> permitted (empty result) ----------------------
            pytest.param("/other/**", "/work", [], id="unrelated-subtree"),
            pytest.param("/workshop/**", "/work", [], id="sibling-prefix-glob"),
            pytest.param("/work2", "/work", [], id="sibling-prefix-literal"),
            pytest.param("/work/secrets", "/work/logs", [], id="sibling-leaf"),
            # Single-component file glob: non-matching siblings are allowed
            pytest.param("/work/*.log", "/work/notes.txt", [], id="file-glob-does-not-block-non-matching-sibling"),
            # Single-component wildcard: ancestor that matches the glob blocks
            # delete of its descendants (wildcard-denied dirs get the same
            # protection as literal-denied dirs)
            pytest.param("/work/*", "/work/app/child", ["/work/*"], id="wildcard-ancestor-blocks-descendant-delete"),
            pytest.param("/work/*.log", "/work/app.log/child", ["/work/*.log"], id="file-glob-ancestor-blocks-descendant-delete"),
            pytest.param("/work/*", "/work/app/deep/nested", ["/work/*"], id="wildcard-ancestor-blocks-deep-descendant-delete"),
            # Directory wildcard after anchor: target below anchor is blocked (fail closed)
            pytest.param("/work/*/secrets", "/work/app", ["/work/*/secrets"], id="dir-wildcard-blocks-ancestor-target"),
            pytest.param("/work/**/secrets", "/work/app", ["/work/**/secrets"], id="globstar-wildcard-blocks-ancestor-target"),
            # Recursive glob with suffix: deleting /work/sub would remove /work/sub/a.log
            pytest.param("/work/**/*.log", "/work/sub", ["/work/**/*.log"], id="recursive-glob-blocks-descendant-that-contains-match"),
            # --- overlap in either direction -> blocked ----------------------
            pytest.param("/work/a.txt", "/work/a.txt", ["/work/a.txt"], id="exact-file"),
            pytest.param("/work", "/work", ["/work"], id="exact-dir-literal"),
            pytest.param("/work/secrets/**", "/work", ["/work/secrets/**"], id="descendant-pattern-blocks-ancestor-target"),
            pytest.param("/work/**", "/work/logs", ["/work/**"], id="ancestor-glob-blocks-descendant-target"),
            pytest.param("/work", "/work/sub/deep", ["/work"], id="ancestor-literal-blocks-descendant-target"),
            pytest.param("/work/secrets", "/work", ["/work/secrets"], id="bare-directory-pattern"),
            # --- wildcard / root anchors -------------------------------------
            pytest.param("/anything/**", "/", ["/anything/**"], id="root-target-blocked-by-any-rule"),
            pytest.param("/**/secrets", "/work", ["/**/secrets"], id="leading-wildcard-anchor-is-root"),
            # --- trailing-slash normalization (both sides) -------------------
            pytest.param("/work/**", "/work/", ["/work/**"], id="target-trailing-slash"),
            pytest.param("/work/", "/work/sub", ["/work/"], id="pattern-trailing-slash"),
            pytest.param("/work/", "/work/", ["/work/"], id="both-trailing-slash"),
        ],
    )
    def test_overlap_geometry(self, pattern: str, target: str, expected: list[str]):
        assert _find_delete_deny_patterns([_deny(pattern)], target) == expected

    @pytest.mark.parametrize(
        ("operations", "mode", "expected"),
        [
            pytest.param(["write"], "deny", ["/work/**"], id="write-deny-blocks"),
            pytest.param(["read", "write"], "deny", ["/work/**"], id="readwrite-deny-blocks"),
            pytest.param(["write"], "allow", [], id="allow-ignored"),
            pytest.param(["write"], "interrupt", [], id="interrupt-ignored"),
            pytest.param(["read"], "deny", [], id="read-only-deny-ignored"),
        ],
    )
    def test_mode_and_operation_filtering(self, operations, mode, expected):
        # Only deny + write rules count; the target always overlaps the pattern.
        rule = FilesystemPermission(operations=operations, paths=["/work/**"], mode=mode)
        assert _find_delete_deny_patterns([rule], "/work") == expected

    @pytest.mark.parametrize(
        ("rule_paths", "target", "expected"),
        [
            pytest.param([], "/work", [], id="no-permissions"),
            pytest.param([["/work/a/**"], ["/work/b/**"]], "/work", ["/work/a/**", "/work/b/**"], id="multiple-rules"),
            pytest.param(
                [["/work/a/**", "/work/b/**", "/other/**"]],
                "/work",
                ["/work/a/**", "/work/b/**"],
                id="multiple-paths-one-rule-filters-non-overlapping",
            ),
            pytest.param(
                [[f"/work/d{i}/**"] for i in range(8)],
                "/work",
                [f"/work/d{i}/**" for i in range(8)],
                id="all-overlapping-no-cap",
            ),
            pytest.param(
                [["/work/x/**"], ["/work/x/**", "/work/y/**"]],
                "/work",
                ["/work/x/**", "/work/y/**"],
                id="deduplicated-first-seen-order",
            ),
            pytest.param(
                [["/other/**"], ["/work/x/**"], ["/elsewhere/**"]],
                "/work",
                ["/work/x/**"],
                id="only-overlapping-returned",
            ),
        ],
    )
    def test_multiple_rule_aggregation(self, rule_paths, target, expected):
        rules = [_deny(*paths) for paths in rule_paths]
        assert _find_delete_deny_patterns(rules, target) == expected


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

    def test_path_without_leading_slash_raises(self):
        with pytest.raises(ValueError, match="Permission path must start with '/'"):
            FilesystemPermission(operations=["read"], paths=["workspace/**"])

    def test_mixed_paths_with_missing_slash_raises(self):
        with pytest.raises(ValueError, match="Permission path must start with '/'"):
            FilesystemPermission(operations=["read"], paths=["/valid/**", "invalid/**"])

    def test_path_with_dotdot_raises(self):
        with pytest.raises(ValueError, match=r"must not contain '\.\.'"):
            FilesystemPermission(operations=["read"], paths=["/workspace/../secrets/**"])

    def test_path_with_tilde_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="must not contain '~'"):
            FilesystemPermission(operations=["read"], paths=["/~/data/**"])

    def test_backslash_path_with_dotdot_raises(self):
        r"""`FilesystemPermission` normalizes backslashes before traversal checks.

        A Windows-style path with `\..\` escaping a leading-slash prefix must
        still be rejected: without normalization, `PurePosixPath(r"/a\..\b").parts`
        yields the single component `r"a\..\b"` and the `'..' in parts` guard
        would never fire, letting a traversal pattern slip past.
        """
        with pytest.raises(ValueError, match=r"must not contain '\.\.'"):
            FilesystemPermission(operations=["read"], paths=["/workspace\\..\\secrets\\**"])

    def test_mixed_separator_path_with_dotdot_raises(self):
        """Mixed separators must also be rejected when they contain traversal."""
        with pytest.raises(ValueError, match=r"must not contain '\.\.'"):
            FilesystemPermission(operations=["read"], paths=["/workspace/..\\secrets/**"])

    def test_backslash_path_without_traversal_accepted(self):
        r"""Backslashes alone must not be rejected -- only `..` components are.

        After `to_posix_path`, a path like `/workspace\sub` becomes `/workspace/sub`,
        which has no traversal components and should pass validation.
        """
        rule = FilesystemPermission(operations=["read"], paths=["/workspace\\sub\\**"])
        assert rule.paths == ["/workspace\\sub\\**"]

    def test_interrupt_mode_accepted(self):
        rule = FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")
        assert rule.mode == "interrupt"


class _FakeReq:
    """Stand-in for ToolCallRequest; we only read `tool_call['args']` in the predicate."""

    def __init__(self, args: dict) -> None:
        self.tool_call = {"args": args}


class TestCheckFsPermissionInterrupt:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/secrets/x.txt", "interrupt"),
            ("/workspace/x.txt", "allow"),
        ],
    )
    def test_interrupt_rule_resolution(self, path, expected):
        rules = [FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")]
        assert _check_fs_permission(rules, "write", path) == expected

    def test_deny_rule_takes_precedence_when_listed_first(self):
        """First-match wins; if deny is listed first, it beats a later interrupt rule."""
        rules = [
            FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt"),
        ]
        assert _check_fs_permission(rules, "write", "/secrets/x.txt") == "deny"

    def test_filter_paths_does_not_drop_interrupt_paths(self):
        """Interrupt-mode paths must remain in result-filtered lists.

        The pre-execution `when` predicate is scope-aware: it fires before the
        tool runs for both exact-path tools and bulk-path tools (including the
        pathless and parent-path cases), so by the time result-filtering runs
        the user has either approved or the call was rejected. Stripping
        interrupt-mode results here would silently empty out a listing the
        user just approved.
        """
        rules = [FilesystemPermission(operations=["read"], paths=["/secret/**"], mode="interrupt")]
        kept = _filter_paths_by_permission(rules, "read", ["/secret/a.txt", "/public/b.txt"])
        assert kept == ["/secret/a.txt", "/public/b.txt"]


class TestBuildInterruptOnFromPermissions:
    def test_empty_when_no_rules(self):
        assert _build_interrupt_on_from_permissions([]) == {}

    def test_empty_when_no_interrupt_rules(self):
        rules = [FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")]
        assert _build_interrupt_on_from_permissions(rules) == {}

    def test_registers_only_tools_whose_op_could_interrupt(self):
        """A write-only interrupt rule registers only the write-op tools."""
        rule = FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")
        out = _build_interrupt_on_from_permissions([rule])
        assert set(out) == {"write_file", "edit_file", "delete"}

    def test_registers_read_tools_for_read_interrupt(self):
        rule = FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="interrupt")
        out = _build_interrupt_on_from_permissions([rule])
        assert set(out) == {"ls", "read_file", "glob", "grep"}

    @pytest.mark.parametrize(
        ("file_path", "expected"),
        [
            # literal match inside the rule's subtree
            ("/secrets/key.pem", True),
            # outside the rule
            ("/workspace/x.txt", False),
            # the parent dir itself doesn't match `/secrets/**` (** requires content after the slash)
            ("/secrets", False),
        ],
    )
    def test_exact_predicate(self, file_path, expected):
        """Exact-scope tools fire iff the literal path arg matches the rule."""
        rule = FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")
        when = _make_fs_when_predicate([rule], "write", "file_path", "exact")
        assert when(_FakeReq({"file_path": file_path})) is expected

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            # pathless: can't localize → fire
            ({"path": None}, True),
            ({}, True),
            # current-dir aliases collapse to root → fire
            ({"path": "."}, True),
            ({"path": ""}, True),
            ({"path": "./"}, True),
            ({"path": "/."}, True),
            ({"path": "/"}, True),
            # ancestor of the rule's anchor → fire (listing surfaces protected children)
            ({"path": "/secrets"}, True),
            # inside the rule's subtree → fire
            ({"path": "/secrets/sub"}, True),
            # unrelated subtree → no fire
            ({"path": "/workspace"}, False),
            # prefix lookalike — component-aware match, not string prefix
            ({"path": "/secret"}, False),
            # path-validation failure short-circuits to no interrupt
            ({"path": "/secrets/../etc/passwd"}, False),
        ],
    )
    def test_bulk_predicate(self, args, expected):
        """Bulk-scope tools fire when the call subtree could intersect a rule.

        Covers the HITL-bypass regressions for pathless calls and current-dir
        aliases like `"."`/`""`/`"./"`: `validate_path` collapses those to
        `/.`, which doesn't string-prefix any anchor, so the predicate
        previously returned False and an agent could call e.g. `grep(pattern,
        path=".")` to read interrupt-protected paths with no HITL prompt.
        """
        rule = FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="interrupt")
        when = _make_fs_when_predicate([rule], "read", "path", "bulk")
        assert when(_FakeReq(args)) is expected

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            # Absolute pattern: Python `glob` ignores `path` (the backend's
            # `os.chdir(path)` has no effect on an absolute pattern), so gate on
            # the pattern's own anchor. A benign `path` must not suppress the
            # interrupt when the pattern reaches into a protected subtree.
            ({"pattern": "/secrets/**", "path": "/workspace"}, True),
            ({"pattern": "/secrets/sub/*.txt", "path": "/workspace"}, True),
            # Absolute pattern anchored at root → overlaps everything → fire.
            ({"pattern": "/**/key.pem", "path": "/workspace"}, True),
            # Absolute pattern outside any interrupt rule → no fire.
            ({"pattern": "/workspace/**", "path": "/workspace"}, False),
            # Relative pattern climbing out of `path` via `..` can't be
            # localized statically → fire conservatively.
            ({"pattern": "../secrets/*", "path": "/workspace"}, True),
            ({"pattern": "../../etc/*", "path": "/workspace/sub"}, True),
            # Benign relative pattern under a non-overlapping path → no fire.
            ({"pattern": "*.txt", "path": "/workspace"}, False),
            # Relative pattern under the protected subtree still fires via the
            # existing `path` check.
            ({"pattern": "*.txt", "path": "/secrets"}, True),
        ],
    )
    def test_bulk_glob_pattern_arg(self, args, expected):
        """The glob bulk predicate gates on its `pattern` arg, not just `path`.

        Regression for the HITL bypass where `glob(pattern="/secrets/**",
        path="/workspace")` slipped past an interrupt rule on `/secrets/**`:
        the predicate saw only the benign `/workspace` path while the sandbox
        backend (`os.chdir(path)` then `glob.glob(pattern)`) enumerated
        `/secrets` anyway, because Python's `glob` ignores the working
        directory for absolute patterns.
        """
        rule = FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="interrupt")
        when = _build_interrupt_on_from_permissions([rule])["glob"]["when"]
        assert when(_FakeReq(args)) is expected


class TestGlobAnchorAndOverlap:
    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            ("/secrets/**", "/secrets"),
            ("/a/*/b", "/a"),
            ("/secrets/key.pem", "/secrets/key.pem"),
            ("/*/foo", "/"),
            ("/**/secrets", "/"),
        ],
    )
    def test_glob_anchor(self, pattern, expected):
        assert _glob_anchor(pattern) == expected

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            # equal
            ("/a/b", "/a/b", True),
            # call inside rule
            ("/a/b/c", "/a/b", True),
            # rule inside call
            ("/a", "/a/b", True),
            # root overlaps everything, either direction
            ("/", "/anywhere", True),
            ("/anywhere", "/", True),
            # component-aware: prefix lookalikes don't overlap
            ("/secret", "/secrets", False),
            ("/secrets", "/secret", False),
            # disjoint
            ("/workspace", "/secrets", False),
        ],
    )
    def test_paths_overlap(self, a, b, expected):
        assert _paths_overlap(a, b) is expected


class TestFilesystemMiddlewarePermissionInit:
    def _backend(self):
        return _make_backend()

    def test_raises_not_implemented_for_sandbox_backend(self):
        """FilesystemMiddleware rejects permissions for backends that support execution."""

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

        with pytest.raises(NotImplementedError, match="execute"):
            FilesystemMiddleware(
                backend=sandbox,
                _permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
            )

    def test_raises_not_implemented_for_composite_with_sandbox_default(self):
        """FilesystemMiddleware rejects CompositeBackend whose default supports execution."""

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

        with pytest.raises(NotImplementedError, match="execute"):
            FilesystemMiddleware(
                backend=composite,
                _permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
            )

    def test_allows_composite_without_sandbox_default(self):
        """FilesystemMiddleware accepts CompositeBackend whose default does not support execution."""
        composite = CompositeBackend(default=self._backend(), routes={})
        middleware = FilesystemMiddleware(
            backend=composite,
            _permissions=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        assert middleware._permissions

    def test_allows_composite_with_sandbox_route_but_non_sandbox_default(self):
        """CompositeBackend with sandbox in a route but non-sandbox default is allowed.

        Execution is only delegated to the default backend in CompositeBackend,
        so a sandbox in a route doesn't expose execution capability.
        """

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
        middleware = FilesystemMiddleware(
            backend=composite,
            _permissions=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
        )
        assert middleware._permissions

    def test_all_paths_scoped_to_routes_helper(self):
        composite = CompositeBackend(default=self._backend(), routes={"/memories/": self._backend()})
        rules = [FilesystemPermission(operations=["read"], paths=["/memories/**"], mode="deny")]
        assert _all_paths_scoped_to_routes(rules, composite) is True


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

    def test_read_binary_allowed_on_permitted_path(self):
        class ImageBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return ReadResult(file_data={"content": "<base64_data>", "encoding": "base64"})

        middleware = FilesystemMiddleware(backend=ImageBackend())
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(read_tool, {"file_path": "/app/screenshot.png"}, rules)
        assert isinstance(result, list)
        assert result[0]["type"] == "image"
        assert result[0]["base64"] == "<base64_data>"

    def test_read_binary_denied_on_restricted_path(self):
        class ImageBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return ReadResult(file_data={"content": "<base64_data>", "encoding": "base64"})

        middleware = FilesystemMiddleware(backend=ImageBackend())
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(read_tool, {"file_path": "/secrets/screenshot.png"}, rules)
        assert "permission denied" in result
        assert "read" in result

    def test_read_backend_error_passthrough_when_allowed(self):
        class ErrorBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return ReadResult(error="file_not_found")

        middleware = FilesystemMiddleware(backend=ErrorBackend())
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(read_tool, {"file_path": "/workspace/missing.txt"}, rules)
        assert result == "Error: file_not_found"

    def test_read_first_matching_rule_wins_at_tool_level(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [
            FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny"),
            FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="allow"),
        ]
        result = _invoke_with_permissions(read_tool, {"file_path": "/secrets/key.txt"}, rules)
        assert "permission denied" in result

    def test_write_denied_on_restricted_path(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(write_tool, {"file_path": "/foo.txt", "content": "data"}, rules)
        assert "permission denied" in result
        assert "write" in result

    def test_write_backend_error_passthrough_when_allowed(self):
        class ErrorBackend(StateBackend):
            def write(self, path, content):

                return WriteResult(error="disk full", path=path)

        middleware = FilesystemMiddleware(backend=ErrorBackend())
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(write_tool, {"file_path": "/workspace/out.txt", "content": "data"}, rules)
        assert result == "disk full"

    def test_write_first_matching_rule_wins_at_tool_level(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [
            FilesystemPermission(operations=["write"], paths=["/workspace/**"], mode="deny"),
            FilesystemPermission(operations=["write"], paths=["/workspace/**"], mode="allow"),
        ]
        result = _invoke_with_permissions(write_tool, {"file_path": "/workspace/out.txt", "content": "data"}, rules)
        assert "permission denied" in result

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

    def test_edit_backend_error_passthrough_when_allowed(self):
        class ErrorBackend(StateBackend):
            def edit(self, path, old_string, new_string, *, replace_all=False):
                return EditResult(error="no unique match", path=path, occurrences=0)

        middleware = FilesystemMiddleware(backend=ErrorBackend())
        edit_tool = next(t for t in middleware.tools if t.name == "edit_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/protected/**"], mode="deny")]
        result = _invoke_with_permissions(
            edit_tool,
            {
                "file_path": "/workspace/file.txt",
                "old_string": "original",
                "new_string": "changed",
            },
            rules,
        )
        assert result == "no unique match"

    def test_edit_first_matching_rule_wins_at_tool_level(self):
        backend = _make_backend({"/workspace/file.txt": "original"})
        middleware = FilesystemMiddleware(backend=backend)
        edit_tool = next(t for t in middleware.tools if t.name == "edit_file")
        rules = [
            FilesystemPermission(operations=["write"], paths=["/workspace/**"], mode="deny"),
            FilesystemPermission(operations=["write"], paths=["/workspace/**"], mode="allow"),
        ]
        result = _invoke_with_permissions(
            edit_tool,
            {
                "file_path": "/workspace/file.txt",
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
        assert "permission denied" not in result.content

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
        assert "Path traversal not allowed" in result.content

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

    def test_sync_glob_rejects_when_timed_out_workers_are_saturated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class BlockingGlobBackend(StoreBackend):
            def __init__(self, *, release: threading.Event, started: threading.Semaphore) -> None:
                super().__init__(store=InMemoryStore(), namespace=lambda _ctx: ("filesystem",))
                self.release = release
                self.started = started
                self._calls = 0
                self._lock = threading.Lock()

            @property
            def calls(self) -> int:
                with self._lock:
                    return self._calls

            def glob(self, pattern: str, path: str | None = None) -> GlobResult:
                with self._lock:
                    self._calls += 1
                self.started.release()
                self.release.wait(timeout=5)
                return GlobResult(matches=[])

        monkeypatch.setattr(filesystem_module, "GLOB_TIMEOUT", 0.01)
        release = threading.Event()
        started = threading.Semaphore(0)
        backend = BlockingGlobBackend(release=release, started=started)
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")

        try:
            for idx in range(filesystem_module._SYNC_GLOB_WORKERS):
                result = glob_tool.invoke({"runtime": _runtime(f"glob-{idx}"), "pattern": "**/*", "path": "/"})
                assert "glob timed out" in result.content
                assert started.acquire(timeout=1)

            result = glob_tool.invoke({"runtime": _runtime("glob-saturated"), "pattern": "**/*", "path": "/"})

            assert "too many glob calls are already running" in result.content
            assert backend.calls == filesystem_module._SYNC_GLOB_WORKERS
        finally:
            release.set()
            middleware._glob_executor.shutdown(wait=True)

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

    async def test_read_binary_allowed_async(self):
        class ImageBackend(StateBackend):
            async def aread(self, path, *, offset=0, limit=100):
                return ReadResult(file_data={"content": "<base64_data>", "encoding": "base64"})

        middleware = FilesystemMiddleware(backend=ImageBackend())
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(read_tool, {"file_path": "/app/screenshot.png"}, rules)
        assert isinstance(result, list)
        assert result[0]["type"] == "image"
        assert result[0]["base64"] == "<base64_data>"

    async def test_read_backend_error_passthrough_async(self):
        class ErrorBackend(StateBackend):
            async def aread(self, path, *, offset=0, limit=100):
                return ReadResult(error="file_not_found")

        middleware = FilesystemMiddleware(backend=ErrorBackend())
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(read_tool, {"file_path": "/workspace/missing.txt"}, rules)
        assert result == "Error: file_not_found"

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


def _filesystem_permissions_for(agent, subagent_name: str | None = None):
    """Walk a compiled deep agent to fetch a `FilesystemMiddleware._permissions`.

    When `subagent_name` is None, returns the main agent's permissions.
    Otherwise descends into the named subagent registered on the `task` tool.
    """
    if subagent_name is not None:
        task_tool = agent.nodes["tools"].bound._tools_by_name["task"]
        agents_dict = next(
            cell.cell_contents for cell in task_tool.func.__closure__ if isinstance(cell.cell_contents, dict) and subagent_name in cell.cell_contents
        )
        agent = agents_dict[subagent_name]

    read_tool = agent.nodes["tools"].bound._tools_by_name["read_file"]
    fs = next(cell.cell_contents for cell in read_tool.func.__closure__ if isinstance(cell.cell_contents, FilesystemMiddleware))
    return fs._permissions


class TestGeneralPurposeSubagentPermissionInheritance:
    """Regression tests: auto-added GP subagent must inherit parent permissions."""

    def test_auto_added_gp_subagent_inherits_parent_permissions(self):
        parent_perms = [
            FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
        ]
        agent = create_deep_agent(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="done")])),
            permissions=parent_perms,
        )

        assert _filesystem_permissions_for(agent) == parent_perms
        assert _filesystem_permissions_for(agent, "general-purpose") == parent_perms

    def test_explicit_gp_subagent_permissions_override_parent(self):
        parent_perms = [
            FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
        ]
        override_perms = [
            FilesystemPermission(operations=["read"], paths=["/foo/**"], mode="deny"),
        ]
        agent = create_deep_agent(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="done")])),
            permissions=parent_perms,
            subagents=[{**GENERAL_PURPOSE_SUBAGENT, "permissions": override_perms}],
        )

        assert _filesystem_permissions_for(agent) == parent_perms
        assert _filesystem_permissions_for(agent, "general-purpose") == override_perms
