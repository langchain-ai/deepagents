# Windows Compatibility Changes for DeepAgents CLI

This document summarizes the changes implemented to make `deepagents-cli` work smoothly on Windows, including file paths and terminal execution. It records before/after code snippets and file paths so maintainers can track and upgrade accordingly.

## Summary
- Accept Windows absolute paths (e.g., `C:\...`) in filesystem middleware.
- Support absolute Windows glob patterns (e.g., `C:\...\**\*.py`).
- Avoid Unix-only imports at module load; use lazy imports inside functions.
- Default editor on Windows set to `notepad`.
- UI copy uses cross-platform “SHELL MODE” and help examples.
- Prefer local `libs/deepagents` over installed package to ensure patched middleware is used during development.

## Changes

### deepagents-cli

- `libs/deepagents-cli/deepagents_cli/execution.py`
  - Before
    ```python
    import termios
    import tty
    ```
  - After
    ```python
    # Removed top-level imports
    
    def prompt_for_tool_approval(...):
        try:
            import termios  # type: ignore
            import tty  # type: ignore
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ...
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            # Fallback for non-Unix systems
            console.print("  ☐ (A)pprove  (default)")
            console.print("  ☐ (R)eject")
            console.print("  ☐ (Auto)-accept all going forward")
            choice = input("\nChoice (A/R/Auto, default=Approve): ").strip().lower()
    ```

- `libs/deepagents-cli/deepagents_cli/input.py`
  - Before
    ```python
    if current_text.startswith("!"):
        parts.append(("bg:#ff1493 fg:#ffffff bold", " BASH MODE "))
    
    if "EDITOR" not in os.environ:
        os.environ["EDITOR"] = "nano"
    ```
  - After
    ```python
    if current_text.startswith("!"):
        parts.append(("bg:#ff1493 fg:#ffffff bold", " SHELL MODE "))
    
    if "EDITOR" not in os.environ:
        os.environ["EDITOR"] = "notepad" if sys.platform.startswith("win") else "nano"
    ```

- `libs/deepagents-cli/deepagents_cli/ui.py`
  - Before
    ```python
    "  !command        Type ! to run bash commands (e.g., !ls, !git status)"
    ```
  - After
    ```python
    "  !command        Type ! to run shell commands (e.g., !dir on Windows, !ls on macOS/Linux)"
    ```

- `libs/deepagents-cli/deepagents_cli/token_utils.py`
  - Before
    ```python
    project_deepagents_dir = f"{project_root}/.deepagents"
    ```
  - After
    ```python
    project_deepagents_dir = str(project_root / ".deepagents")
    ```

- `libs/deepagents-cli/deepagents_cli/main.py`
  - Before
    ```python
    from deepagents.backends.protocol import SandboxBackendProtocol
    ```
  - After
    ```python
    # Prefer local monorepo deepagents library when available
    try:
        _libs_dir = Path(__file__).resolve().parents[2]
        _local_deepagents = _libs_dir / "deepagents"
        if _local_deepagents.exists():
            sys.path.insert(0, str(_local_deepagents))
    except Exception:
        pass
    
    from deepagents.backends.protocol import SandboxBackendProtocol
    ```

### deepagents (middleware and backend)

- `libs/deepagents/deepagents/middleware/filesystem.py`
  - Before
    ```python
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)
    
    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    ```
  - After
    ```python
    if re.match(r"^[a-zA-Z]:", path):
        normalized = os.path.normpath(path)
        normalized = normalized.replace("\\", "/")
        return normalized
    
    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")
    if not re.match(r"^[a-zA-Z]:", normalized) and not normalized.startswith("/"):
        normalized = f"/{normalized}"
    ```

- `libs/deepagents/deepagents/backends/filesystem.py`
  - Before (excerpt from `glob_info` path handling)
    ```python
    for matched_path in search_path.rglob(pattern):
        ...
        abs_path = str(matched_path)
        # no special handling for drive-letter absolute patterns
    ```
  - After (added absolute Windows pattern support)
    ```python
    import glob
    
    if re.match(r"^[a-zA-Z]:", pattern):
        results = []
        matches = glob.glob(pattern, recursive=True)
        for abs_path in matches:
            p = Path(abs_path)
            if not p.is_file():
                continue
            if not self.virtual_mode:
                st = p.stat()
                results.append({
                    "path": abs_path,
                    "is_dir": False,
                    "size": int(st.st_size),
                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                })
            else:
                # convert to virtual path under cwd
                ...
        return sorted(results, key=lambda x: x.get("path", ""))
    
    # fallback: search_path.rglob(pattern)
    ```

## Upgrade Guidance
- The CLI depends on the `deepagents` library’s filesystem middleware and backend for file tools. We vendored fixes in the monorepo under `libs/deepagents`.
- If you upgrade to a newer official `deepagents` release, ensure the following Windows compatibility features exist:
  - Middleware `_validate_path` accepts Windows drive-letter absolute paths and normalizes separators.
  - Backend `glob_info` supports absolute Windows patterns via `glob.glob(..., recursive=True)`.
- If the upstream does not include these changes yet, re-apply the diffs above or keep using the monorepo `libs/deepagents` to preserve Windows behavior.

## Notes
- Web search in the CLI uses the Tavily Search API. Set `TAVILY_API_KEY` to enable real-time web access.
- Secrets in `.env` (e.g., `OPENAI_API_KEY`) should never be checked into public repos or printed in logs.

