"""Reject unreviewed process working-directory reads."""

from __future__ import annotations

import ast
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class CallSite:
    """Stable identity for one process working-directory read."""

    path: str
    scope: str
    kind: str
    occurrence: int = 1


_ALLOWLIST = {
    CallSite(
        "agent.py", "get_system_prompt", "Path.cwd"
    ): "Direct agent builders may omit `cwd`; workspace server builders do not.",
    CallSite(
        "agent.py", "_format_execute_description", "Path.cwd"
    ): "The shell approval display is owned by a separate change.",
    CallSite(
        "agent.py", "create_cli_agent._subagent_cli_middleware", "Path.cwd"
    ): "The workspace server always supplies `effective_cwd` at graph build time.",
    CallSite(
        "agent.py", "create_cli_agent", "Path.cwd", 1
    ): "The local backend fallback serves direct builders without project context.",
    CallSite(
        "agent.py", "create_cli_agent", "Path.cwd", 2
    ): "Auto mode uses this only without project context or `effective_cwd`.",
    CallSite(
        "agent.py", "create_cli_agent", "Path.cwd", 3
    ): "Main hooks use this only outside the workspace server build path.",
    CallSite(
        "app.py", "DeepAgentsApp.__init__", "Path.cwd"
    ): "The TUI runs in the client process.",
    CallSite(
        "client/commands/config.py", "_config_paths", "Path.cwd", 1
    ): "The config command runs in the client process.",
    CallSite(
        "client/commands/config.py", "_config_paths", "Path.cwd", 2
    ): "The config command builds project context in the client process.",
    CallSite(
        "client/launch/server_manager.py", "_capture_project_context", "Path.cwd"
    ): "The client captures the directory used to launch or bind the server.",
    CallSite(
        "client/non_interactive.py", "_run_agent_loop", "Path.cwd", 1
    ): "The headless client records its local working directory.",
    CallSite(
        "client/non_interactive.py", "_run_agent_loop", "Path.cwd", 2
    ): "The headless client evaluates local hook trust.",
    CallSite(
        "client/non_interactive.py", "run_non_interactive", "Path.cwd", 1
    ): "The headless entrypoint builds client-side project context.",
    CallSite(
        "client/non_interactive.py", "run_non_interactive", "Path.cwd", 2
    ): "The headless entrypoint evaluates client-side hook trust.",
    CallSite(
        "client/non_interactive.py", "run_non_interactive", "Path.cwd", 3
    ): "The headless client sends its directory in run context.",
    CallSite(
        "config.py", "_preview_dotenv_environ", "Path.cwd"
    ): "The client-side tracing diagnostic previews the local dotenv file.",
    CallSite(
        "config.py", "_get_git_branch", "Path.cwd"
    ): "Client stream metadata describes the local checkout.",
    CallSite(
        "config.py", "_get_git_commit_sha", "Path.cwd"
    ): "Client stream metadata describes the local checkout.",
    CallSite(
        "config.py", "_get_repository_metadata", "Path.cwd"
    ): "Client stream metadata describes the local checkout.",
    CallSite(
        "config.py", "build_stream_config", "Path.cwd"
    ): "The client builds tracing metadata before sending a run.",
    CallSite(
        "extensions/runtime.py", "_prepare", "Path.cwd"
    ): "Workspace server callers supply `cwd`; the fallback preserves direct loading.",
    CallSite(
        "file_ops.py", "resolve_physical_path", "Path.cwd"
    ): "Approval previews and file tracking run in the client process.",
    CallSite(
        "input.py", "parse_file_mentions", "Path.cwd"
    ): "Prompt parsing runs in the client process.",
    CallSite(
        "input.py", "_resolve_with_unicode_space_variants", "Path.cwd"
    ): "Prompt path correction runs in the client process.",
    CallSite(
        "main.py", "_normalize_cwd_filter", "Path.cwd"
    ): "The CLI normalizes a client-side session filter.",
    CallSite(
        "main.py", "_preload_session_mcp_server_info", "Path.cwd"
    ): "The CLI preloads MCP data before server requests.",
    CallSite(
        "main.py", "run_textual_cli_async", "Path.cwd"
    ): "The interactive CLI captures its launch directory.",
    CallSite(
        "main.py", "_run_acp_cli_async", "Path.cwd"
    ): "The ACP client builds local project context.",
    CallSite(
        "main.py", "_check_mcp_project_trust", "Path.cwd"
    ): "The CLI checks MCP trust for its local project.",
    CallSite(
        "main.py", "_check_project_hooks_trust", "Path.cwd"
    ): "The CLI checks hook trust for its local project.",
    CallSite(
        "main.py", "_check_project_extensions_trust", "Path.cwd"
    ): "The CLI checks extension trust for its local project.",
    CallSite(
        "mcp_tools.py", "_resolve_project_config_base", "find_project_root"
    ): "Client callers may omit project context; workspace server callers supply it.",
    CallSite(
        "mcp_tools.py", "_resolve_project_config_base", "Path.cwd"
    ): "Client callers without a git root fall back to their process directory.",
    CallSite(
        "offload_middleware.py", "_runtime_cwd", "Path.cwd"
    ): "Runs without workspace context retain the process-directory behavior.",
    CallSite(
        "project_utils.py", "find_project_root", "Path.cwd"
    ): "The helper default supports client callers; server callers pass a start path.",
    CallSite(
        "skills/invocation.py", "discover_skills_and_roots", "Path.cwd"
    ): "Skill invocation discovery runs in the TUI or headless client.",
    CallSite(
        "tool_catalog.py", "_load_mcp_server_info", "Path.cwd"
    ): "The standalone tool catalog command runs in the client process.",
    CallSite(
        "tool_display.py", "format_tool_display.abbreviate_path", "Path.cwd"
    ): "Tool result rendering runs in the client process.",
    CallSite(
        "tui/widgets/autocomplete.py", "FuzzyFileController.__init__", "Path.cwd"
    ): "File autocomplete runs in the client TUI.",
    CallSite(
        "tui/widgets/chat_input.py", "ChatInput.__init__", "Path.cwd"
    ): "The chat input runs in the client TUI.",
    CallSite(
        "tui/widgets/messages.py", "ToolCallMessage._format_search_output", "Path.cwd"
    ): "Tool output formatting runs in the client TUI.",
    CallSite(
        "tui/widgets/status.py", "StatusBar.__init__", "Path.cwd"
    ): "The status bar runs in the client TUI.",
    CallSite(
        "tui/widgets/thread_selector.py", "_safe_cwd_string", "Path.cwd"
    ): "The thread selector filters against the client directory.",
    CallSite(
        "tui/widgets/welcome.py", "WelcomeBanner.__init__", "Path.cwd"
    ): "The welcome banner runs in the client TUI.",
}


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, tree: ast.AST) -> None:
        self.path = path
        self.scopes: list[str] = []
        self.counts: Counter[tuple[str, str]] = Counter()
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

    def _call_kind(self, node: ast.Call) -> str | None:
        if node.args or node.keywords:
            return None
        function = node.func
        if isinstance(function, ast.Name):
            if function.id in self.getcwd_names:
                return "os.getcwd"
            if function.id in self.project_root_names:
                return "find_project_root"
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
            key = (scope, kind)
            self.counts[key] += 1
            site = CallSite(self.path, scope, kind, self.counts[key])
            self.sites.append((site, node.lineno))
        self.generic_visit(node)


def find_violations(package_dir: Path) -> list[str]:
    """Return unreviewed and stale process-cwd allowlist entries."""
    found: dict[CallSite, int] = {}
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _Visitor(relative, tree)
        visitor.visit(tree)
        found.update(visitor.sites)

    violations = [
        f"{site.path}:{line}: unreviewed {site.kind} call in {site.scope}"
        for site, line in found.items()
        if site not in _ALLOWLIST
    ]
    violations.extend(
        f"stale allowlist entry: {site.path}: {site.kind} in {site.scope}"
        for site in _ALLOWLIST
        if site not in found
    )
    return sorted(violations)


def main() -> int:
    """Check the package selected on the command line.

    Returns:
        Zero when every process-cwd read is reviewed, otherwise one.
    """
    package_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("deepagents_code")
    violations = find_violations(package_dir)
    if violations:
        print("Process working-directory reads must be reviewed and allowlisted:")
        print("\n".join(f"- {violation}" for violation in violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
