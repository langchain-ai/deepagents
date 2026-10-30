"""Interrupt-mode filesystem rules for tool calls made from code.

A code interpreter calls tools directly, skipping the human-in-the-loop step
that enforces `mode="interrupt"` rules. Such calls carry a marker naming their
own `tool_call_id`; the filesystem tools then fail closed on approval-gated
paths instead of silently allowing them.
"""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import StoreBackend
from deepagents.middleware.filesystem import (
    PROGRAMMATIC_TOOL_CALL_CONFIG_KEY,
    FilesystemMiddleware,
    FilesystemPermission,
)

_APPROVAL = "requires human approval"
_FILES = {
    "/hr/pay.csv": "name,salary\nada,250000\n",
    "/public/readme.md": "hello salary world\n",
    "/secrets/key.txt": "sk-not-real\n",
}
_RULES = [
    FilesystemPermission(operations=["read", "write"], paths=["/hr/**"], mode="interrupt"),
    FilesystemPermission(operations=["read", "write"], paths=["/secrets/**"], mode="deny"),
]


def _backend() -> StoreBackend:
    store = InMemoryStore()
    for path, content in _FILES.items():
        store.put(("filesystem",), path, {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""})
    return StoreBackend(store=store, namespace=lambda _ctx: ("filesystem",))


def _runtime(*, marker: str | None, tool_call_id: str = "call_1") -> ToolRuntime:
    config = {"configurable": {PROGRAMMATIC_TOOL_CALL_CONFIG_KEY: marker}} if marker else {}
    return ToolRuntime(state={}, context=None, tool_call_id=tool_call_id, store=None, stream_writer=lambda _: None, config=config)


def _call(name: str, args: dict, *, marker: str | None, tool_call_id: str = "call_1") -> ToolMessage:
    tools = FilesystemMiddleware(backend=_backend(), permissions=_RULES).tools
    tool = next(t for t in tools if t.name == name)
    return tool.invoke({**args, "runtime": _runtime(marker=marker, tool_call_id=tool_call_id)})


async def _acall(name: str, args: dict, *, marker: str | None) -> ToolMessage:
    tools = FilesystemMiddleware(backend=_backend(), permissions=_RULES).tools
    tool = next(t for t in tools if t.name == name)
    return await tool.ainvoke({**args, "runtime": _runtime(marker=marker)})


class TestCodeCallerFailsClosedOnApprovalRules:
    def test_read_of_gated_path_requires_approval(self):
        result = _call("read_file", {"file_path": "/hr/pay.csv"}, marker="call_1")
        assert result.status == "error"
        assert _APPROVAL in result.content
        assert "250000" not in result.content

    async def test_async_read_of_gated_path_requires_approval(self):
        result = await _acall("read_file", {"file_path": "/hr/pay.csv"}, marker="call_1")
        assert result.status == "error"
        assert _APPROVAL in result.content

    def test_write_to_gated_path_requires_approval(self):
        result = _call("write_file", {"file_path": "/hr/new.csv", "content": "x"}, marker="call_1")
        assert result.status == "error"
        assert _APPROVAL in result.content

    def test_glob_rooted_in_gated_dir_finds_nothing(self):
        result = _call("glob", {"pattern": "*.csv", "path": "/hr"}, marker="call_1")
        assert "/hr/pay.csv" not in result.content

    def test_listings_and_search_hide_gated_paths(self):
        ls = _call("ls", {"path": "/"}, marker="call_1")
        glob = _call("glob", {"pattern": "**/*"}, marker="call_1")
        grep = _call("grep", {"pattern": "salary", "output_mode": "files_with_matches"}, marker="call_1")
        for result in (ls, glob, grep):
            assert "/hr" not in result.content
        assert "/public/readme.md" in grep.content

    def test_deny_rules_keep_their_message(self):
        result = _call("read_file", {"file_path": "/secrets/key.txt"}, marker="call_1")
        assert "permission denied" in result.content
        assert _APPROVAL not in result.content

    def test_ungated_paths_still_work(self):
        result = _call("read_file", {"file_path": "/public/readme.md"}, marker="call_1")
        assert result.status == "success"
        assert "hello salary world" in result.content


class TestModelCallsUnchanged:
    def test_gated_read_without_marker_returns_content(self):
        result = _call("read_file", {"file_path": "/hr/pay.csv"}, marker=None)
        assert "250000" in result.content

    def test_marker_for_another_call_is_ignored(self):
        result = _call("read_file", {"file_path": "/hr/pay.csv"}, marker="outer_call", tool_call_id="call_1")
        assert "250000" in result.content

    def test_listing_without_marker_includes_gated_paths(self):
        result = _call("glob", {"pattern": "**/*"}, marker=None)
        assert "/hr/pay.csv" in result.content
