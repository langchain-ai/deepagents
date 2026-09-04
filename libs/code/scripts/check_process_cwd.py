"""Reject unreviewed reads of the process working directory.

One server process serves many bound workspaces, so a request must use the
directory recorded for its workspace. A read of the process directory returns
the directory the server was launched in, which is the wrong answer for every
workspace but one.

This check reports two things. A call that no allowlist entry covers is
unreviewed. An allowlist entry that no call matches is stale. To clear an
unreviewed call, either pass the workspace directory to it, or add a `CallSite`
entry whose reason names the process the code runs in and why the process
directory is correct there.

The check reads `Path.cwd()`, `os.getcwd()`, and every call to
`find_project_root`, which falls back to the process directory when its
argument is empty. It does not see equivalent spellings such as
`Path(".").resolve()`, `os.path.abspath(".")`, or a call through an attribute
or a saved reference. A clean run means every read it can see is reviewed, not
that no other read exists.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_Kind = Literal["Path.cwd", "os.getcwd", "find_project_root"]


@dataclass(frozen=True)
class CallSite:
    """Identity for one process working-directory read.

    `path` is package-relative and POSIX-separated. `scope` is the dot-joined
    chain of enclosing class and function names, or `<module>`. `token` digests
    the call and its enclosing statement, so an entry stays bound to one call
    when its neighbors change. Editing the statement changes the token, which
    reports the old entry as stale and the edited call as unreviewed: that pair
    is the signal to review the change. `occurrence` disambiguates calls that
    remain identical under that digest, and is the one component that shifts
    when such a call is added or removed.
    """

    path: str
    scope: str
    kind: _Kind
    token: str
    occurrence: int = 1


_ALLOWLIST: dict[CallSite, str] = {
    CallSite(
        "agent.py", "get_system_prompt", "Path.cwd", "aafaff15"
    ): "Direct agent builders may omit `cwd`; workspace server builders do not.",
    CallSite(
        "agent.py", "_format_execute_description", "Path.cwd", "664fe885"
    ): "The shell approval display is owned by a separate change.",
    CallSite(
        "agent.py", "create_cli_agent._subagent_cli_middleware", "Path.cwd", "9f5c0d79"
    ): "The workspace server always supplies `effective_cwd` at graph build time.",
    CallSite(
        "agent.py", "create_cli_agent", "Path.cwd", "c572ebd9"
    ): "The local backend fallback serves direct builders without project context.",
    CallSite(
        "agent.py", "create_cli_agent", "Path.cwd", "cbdd2016"
    ): "Auto mode uses this only without project context or `effective_cwd`.",
    CallSite(
        "agent.py", "create_cli_agent", "Path.cwd", "9f5c0d79"
    ): "Main hooks use this only outside the workspace server build path.",
    CallSite(
        "app.py", "DeepAgentsApp.__init__", "Path.cwd", "d2c4b637"
    ): "The TUI runs in the client process.",
    CallSite(
        "client/commands/config.py", "_config_paths", "Path.cwd", "e6106216"
    ): "The config command runs in the client process.",
    CallSite(
        "client/commands/config.py", "_config_paths", "Path.cwd", "53b35c21"
    ): "The config command builds project context in the client process.",
    CallSite(
        "client/launch/server_manager.py",
        "_capture_project_context",
        "Path.cwd",
        "2956b508",
    ): "The client captures the directory used to launch or bind the server.",
    CallSite(
        "client/non_interactive.py", "_run_agent_loop", "Path.cwd", "8826e907"
    ): "The headless client records its local working directory.",
    CallSite(
        "client/non_interactive.py", "_run_agent_loop", "Path.cwd", "8826e907", 2
    ): "The headless client evaluates local hook trust.",
    CallSite(
        "client/non_interactive.py", "run_non_interactive", "Path.cwd", "164dbd9d"
    ): "The headless entrypoint builds client-side project context.",
    CallSite(
        "client/non_interactive.py", "run_non_interactive", "Path.cwd", "164dbd9d", 2
    ): "The headless entrypoint evaluates client-side hook trust.",
    CallSite(
        "client/non_interactive.py", "run_non_interactive", "Path.cwd", "0f2377b6"
    ): "The headless client sends its directory in run context.",
    CallSite(
        "config.py", "_dotenv_environment", "Path.cwd", "55cd4e5f"
    ): "Workspace server callers pass `start_path`; client processes fall back "
    "to their own launch directory, which is the correct project directory there.",
    CallSite(
        "config.py", "_get_git_branch", "Path.cwd", "7a7a7e4f"
    ): "Client stream metadata describes the local checkout.",
    CallSite(
        "config.py", "_get_git_commit_sha", "Path.cwd", "7a7a7e4f"
    ): "Client stream metadata describes the local checkout.",
    CallSite(
        "config.py", "_get_repository_metadata", "Path.cwd", "7a7a7e4f"
    ): "Client stream metadata describes the local checkout.",
    CallSite(
        "config.py", "build_stream_config", "Path.cwd", "7a7a7e4f"
    ): "The client builds tracing metadata before sending a run.",
    CallSite(
        "config.py",
        "Credentials.snapshot_from_environment",
        "find_project_root",
        "8eea0706",
    ): "The skills CLI omits `start_path`. Bootstrap supplies the launch "
    "directory. Server callers pass the workspace path.",
    CallSite(
        "config.py", "Credentials._reload_values", "find_project_root", "a0389172"
    ): "Client reloads may omit `start_path`; server callers supply workspace context.",
    CallSite(
        "extensions/runtime.py", "_prepare", "Path.cwd", "454696d6"
    ): "Workspace server callers supply `cwd`; the fallback preserves direct loading.",
    CallSite(
        "file_ops.py", "resolve_physical_path", "Path.cwd", "41c20ddf"
    ): "Approval previews and file tracking run in the client process.",
    CallSite(
        "input.py", "parse_file_mentions", "Path.cwd", "848de78e"
    ): "Prompt parsing runs in the client process.",
    CallSite(
        "input.py", "_resolve_with_unicode_space_variants", "Path.cwd", "f71ec895"
    ): "Prompt path correction runs in the client process.",
    CallSite(
        "main.py", "_normalize_cwd_filter", "Path.cwd", "af8d4df1"
    ): "The CLI normalizes a client-side session filter.",
    CallSite(
        "main.py", "_preload_session_mcp_server_info", "Path.cwd", "53b35c21"
    ): "The CLI preloads MCP data before server requests.",
    CallSite(
        "main.py", "run_textual_cli_async", "Path.cwd", "47da3baa"
    ): "The interactive CLI captures its launch directory.",
    CallSite(
        "main.py", "_run_acp_cli_async", "Path.cwd", "53b35c21"
    ): "The ACP client builds local project context.",
    CallSite(
        "main.py", "_check_mcp_project_trust", "Path.cwd", "53b35c21"
    ): "The CLI checks MCP trust for its local project.",
    CallSite(
        "main.py", "_check_project_hooks_trust", "Path.cwd", "3124f423"
    ): "The CLI checks hook trust for its local project.",
    CallSite(
        "main.py", "_check_project_extensions_trust", "Path.cwd", "3124f423"
    ): "The CLI checks extension trust for its local project.",
    CallSite(
        "mcp_tools.py", "_resolve_project_config_base", "find_project_root", "55e278e6"
    ): "Client callers may omit project context; workspace server callers supply it.",
    CallSite(
        "mcp_tools.py", "_resolve_project_config_base", "Path.cwd", "9050f1d8"
    ): "Client callers without a git root fall back to their process directory.",
    CallSite(
        "offload_middleware.py", "_runtime_cwd", "Path.cwd", "c5238c5c"
    ): "Runs without workspace context retain the process-directory behavior.",
    CallSite(
        "project_utils.py", "find_project_root", "Path.cwd", "60fa8f5e"
    ): "The helper default supports client callers; server callers pass a start path.",
    CallSite(
        "skills/invocation.py", "discover_skills_and_roots", "Path.cwd", "7649626b"
    ): "Skill invocation discovery runs in the TUI or headless client.",
    CallSite(
        "tool_catalog.py", "_load_mcp_server_info", "Path.cwd", "53b35c21"
    ): "The standalone tool catalog command runs in the client process.",
    CallSite(
        "tool_display.py", "format_tool_display.abbreviate_path", "Path.cwd", "3aa6f3a0"
    ): "Tool result rendering runs in the client process.",
    CallSite(
        "tui/widgets/autocomplete.py",
        "FuzzyFileController.__init__",
        "Path.cwd",
        "1c6451f3",
    ): "File autocomplete runs in the client TUI.",
    CallSite(
        "tui/widgets/autocomplete.py",
        "FuzzyFileController.__init__",
        "find_project_root",
        "fb0caed2",
    ): "File autocomplete resolves its process or caller-provided directory first.",
    CallSite(
        "tui/widgets/chat_input.py", "ChatInput.__init__", "Path.cwd", "ded5e2c9"
    ): "The chat input runs in the client TUI.",
    CallSite(
        "tui/widgets/messages.py",
        "ToolCallMessage._format_search_output",
        "Path.cwd",
        "bb3631d2",
    ): "Tool output formatting runs in the client TUI.",
    CallSite(
        "tui/widgets/status.py", "StatusBar.__init__", "Path.cwd", "5d61c81b"
    ): "The status bar runs in the client TUI.",
    CallSite(
        "tui/widgets/thread_selector.py", "_safe_cwd_string", "Path.cwd", "af8d4df1"
    ): "The thread selector filters against the client directory.",
    CallSite(
        "tui/widgets/welcome.py", "WelcomeBanner.__init__", "Path.cwd", "ecc0d716"
    ): "The welcome banner runs in the client TUI.",
    CallSite(
        "workspace.py", "resolve_workspace", "find_project_root", "6091a322"
    ): "Workspace resolution validates and canonicalizes the request directory first.",
    CallSite(
        "project_utils.py",
        "ProjectContext.from_user_cwd",
        "find_project_root",
        "67630849",
    ): "The argument is the resolved directory the caller supplied.",
    CallSite(
        "project_utils.py",
        "get_server_project_context",
        "find_project_root",
        "32fc96f5",
    ): "Server bootstrap supplies this directory from the environment.",
}


def _statement_text(node: ast.stmt) -> str:
    """Return a summary of a statement that its body cannot change."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return f"{type(node).__name__}:{node.name}"
    if isinstance(node, ast.If):
        return f"if {ast.unparse(node.test)}"
    if isinstance(node, ast.While):
        return f"while {ast.unparse(node.test)}"
    if isinstance(node, ast.For | ast.AsyncFor):
        return f"for {ast.unparse(node.target)} in {ast.unparse(node.iter)}"
    if isinstance(node, ast.With | ast.AsyncWith):
        return "with " + ", ".join(ast.unparse(item) for item in node.items)
    if isinstance(node, ast.Try):
        return "try"
    return ast.unparse(node)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, tree: ast.AST) -> None:
        self.path = path
        self.scopes: list[str] = []
        self.statement: ast.stmt | None = None
        self.counts: Counter[tuple[str, str, str]] = Counter()
        self.sites: list[tuple[CallSite, int]] = []
        self.path_names: set[str] = set()
        self.pathlib_names: set[str] = set()
        self.os_names: set[str] = set()
        self.getcwd_names: set[str] = set()
        self.project_root_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pathlib":
                        self.pathlib_names.add(alias.asname or alias.name)
                    elif alias.name == "os":
                        self.os_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if node.module == "pathlib" and alias.name == "Path":
                        self.path_names.add(name)
                    elif node.module == "os" and alias.name == "getcwd":
                        self.getcwd_names.add(name)
                    elif alias.name == "find_project_root":
                        self.project_root_names.add(name)
            elif (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == "find_project_root"
            ):
                # The defining module calls the helper without importing it.
                self.project_root_names.add(node.name)

    def _call_kind(self, node: ast.Call) -> _Kind | None:
        function = node.func
        if isinstance(function, ast.Name) and function.id in self.project_root_names:
            return "find_project_root"
        if node.args or node.keywords:
            return None
        if isinstance(function, ast.Name):
            if function.id in self.getcwd_names:
                return "os.getcwd"
            return None
        if not isinstance(function, ast.Attribute):
            return None
        if (
            function.attr == "cwd"
            and isinstance(function.value, ast.Name)
            and function.value.id in self.path_names
        ):
            return "Path.cwd"
        if (
            function.attr == "cwd"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "Path"
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id in self.pathlib_names
        ):
            return "Path.cwd"
        if (
            function.attr == "getcwd"
            and isinstance(function.value, ast.Name)
            and function.value.id in self.os_names
        ):
            return "os.getcwd"
        return None

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt):
            previous = self.statement
            self.statement = node
            super().visit(node)
            self.statement = previous
            return
        super().visit(node)

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scopes.append(name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        kind = self._call_kind(node)
        if kind is not None:
            scope = ".".join(self.scopes) or "<module>"
            statement = (
                "<module>"
                if self.statement is None
                else _statement_text(self.statement)
            )
            digest = hashlib.sha256(
                f"{statement}|{ast.unparse(node)}".encode()
            ).hexdigest()[:8]
            key = (scope, kind, digest)
            self.counts[key] += 1
            site = CallSite(self.path, scope, kind, digest, self.counts[key])
            self.sites.append((site, node.lineno))
        self.generic_visit(node)


def _token_text(site: CallSite) -> str:
    """Render the identity an allowlist entry must repeat for this call.

    Returns:
        The token, suffixed with the occurrence when more than one call shares
        it, which is the pair a new `CallSite` entry has to name.
    """
    if site.occurrence > 1:
        return f"{site.token}#{site.occurrence}"
    return site.token


def find_violations(package_dir: Path) -> list[str]:
    """Return one message per unreviewed call and per stale allowlist entry.

    Args:
        package_dir: Root of the package to scan. Every `*.py` file below it is
            parsed, and allowlist paths are relative to it.

    Returns:
        The messages, sorted. An empty list means every call the check can see
        is allowlisted and every entry matches a call.

    Raises:
        SystemExit: If a file cannot be read or parsed, because its reads would
            otherwise go unchecked.
    """
    found: dict[CallSite, int] = {}
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            msg = f"{relative}: cannot be read, so its reads are unchecked: {exc}"
            raise SystemExit(msg) from exc
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            msg = f"{relative}: cannot be parsed, so its reads are unchecked: {exc}"
            raise SystemExit(msg) from exc
        visitor = _Visitor(relative, tree)
        visitor.visit(tree)
        found.update(visitor.sites)

    violations = [
        f"{site.path}:{line}: unreviewed {site.kind} call "
        f"[{_token_text(site)}] in {site.scope}"
        for site, line in found.items()
        if site not in _ALLOWLIST
    ]
    violations.extend(
        f"stale allowlist entry: {site.path}: {site.kind} "
        f"[{site.token}] in {site.scope}: {reason}"
        for site, reason in _ALLOWLIST.items()
        if site not in found
    )
    return sorted(violations)


def main() -> int:
    """Check the package named on the command line, or `deepagents_code`.

    Returns:
        Zero when every call the check can see is allowlisted and every entry
        matches a call, otherwise one.
    """
    package_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("deepagents_code")
    violations = find_violations(package_dir)
    if violations:
        print("Reads of the process working directory must be allowlisted:")
        print("\n".join(f"- {violation}" for violation in violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
