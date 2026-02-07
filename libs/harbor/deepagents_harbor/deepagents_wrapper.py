"""A wrapper for DeepAgents to run in Harbor environments."""

import json
import os
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from deepagents import create_deep_agent
from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import get_default_coding_instructions, settings
from deepagents_cli.tools import fetch_url, http_request, web_search
from dotenv import load_dotenv
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from langchain.chat_models import init_chat_model
from langchain.messages import UsageMetadata
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langsmith import trace
from langsmith.client import Client

from deepagents_harbor.backend import HarborSandbox
from deepagents_harbor.middleware import (
    AdaptiveReasoningMiddleware,
    APIErrorRecoveryMiddleware,
    ContextBudgetMiddleware,
    LocalContextMiddleware,
    LoopDetectionMiddleware,
    PreCompletionCheckMiddleware,
)

# Load .env file if present
load_dotenv()

# Directories that can be massive — show they exist but don't recurse into them.
NO_RECURSE_DIRS = frozenset({
    "node_modules", ".git",
})


@dataclass(frozen=True)
class ModelProfileSettings:
    """Runtime tuning profile for Harbor runs."""

    name: str
    recursion_limit: int
    loop_soft_warning_threshold: int
    loop_hard_reflection_threshold: int
    context_max_output_lines: int
    context_warn_threshold_percent: int
    max_context_tokens: int
    max_input_tokens: int


MODEL_PROFILES: dict[str, ModelProfileSettings] = {
    # Preserves current behavior.
    "default": ModelProfileSettings(
        name="default",
        recursion_limit=12000,
        loop_soft_warning_threshold=7,
        loop_hard_reflection_threshold=12,
        context_max_output_lines=200,
        context_warn_threshold_percent=70,
        max_context_tokens=128000,
        max_input_tokens=140000,
    ),
    # OpenAI reasoning models tend to produce longer traces/tool chatter.
    "openai_reasoning": ModelProfileSettings(
        name="openai_reasoning",
        recursion_limit=14000,
        loop_soft_warning_threshold=6,
        loop_hard_reflection_threshold=10,
        context_max_output_lines=160,
        context_warn_threshold_percent=65,
        max_context_tokens=300000,
        max_input_tokens=400000,
    ),
    # Opus generally benefits from more context headroom and less aggressive loop forcing.
    "anthropic_opus": ModelProfileSettings(
        name="anthropic_opus",
        recursion_limit=12000,
        loop_soft_warning_threshold=8,
        loop_hard_reflection_threshold=14,
        context_max_output_lines=240,
        context_warn_threshold_percent=72,
        max_context_tokens=180000,
        max_input_tokens=200000,
    ),
}

_DEFAULT_ANTHROPIC_OPUS46_BETA = "context-1m-2025-08-07"


_DEFAULT_TASK_TIMEOUT_SEC = 900.0  # Most common TB2 task timeout


def _get_task_timeout_from_cache(task_name: str) -> float:
    """Read the task-specific timeout from the harbor task cache.

    Falls back to 900s (the most common TB2 default) if the task.toml
    cannot be found or parsed.
    """
    if not task_name:
        return _DEFAULT_TASK_TIMEOUT_SEC
    cache_base = Path.home() / ".cache" / "harbor" / "tasks"
    try:
        for task_toml in cache_base.glob(f"*/{task_name}/task.toml"):
            with open(task_toml, "rb") as f:
                config = tomllib.load(f)
            timeout = config.get("agent", {}).get("timeout_sec")
            if timeout is not None:
                return float(timeout)
            break  # Only check the first match
    except Exception:
        pass
    return _DEFAULT_TASK_TIMEOUT_SEC


def _parse_csv_list(value: str | None) -> list[str]:
    """Parse comma-separated values into a normalized string list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_anthropic_betas(model_name: str, anthropic_betas: str | None) -> list[str]:
    """Resolve Anthropic beta flags with sensible defaults for Opus 4.6.

    Behavior:
    - If `anthropic_betas` is explicitly provided (including empty string),
      parse and use it as-is.
    - Otherwise, default Opus 4.6 to the long-context beta.
    """
    if anthropic_betas is not None:
        return _parse_csv_list(anthropic_betas)

    model_lower = model_name.lower()
    if "claude-opus-4-6" in model_lower:
        return [_DEFAULT_ANTHROPIC_OPUS46_BETA]

    return []


def _infer_model_profile_name(model_name: str) -> str:
    """Infer a default profile from model name."""
    model_lower = model_name.lower()
    if "codex" in model_lower or "gpt-5" in model_lower:
        return "openai_reasoning"
    if "claude-opus" in model_lower:
        return "anthropic_opus"
    return "default"


def _resolve_model_profile(
    model_name: str,
    profile_name: str | None,
) -> ModelProfileSettings:
    """Resolve profile from explicit name or model inference."""
    selected = profile_name or _infer_model_profile_name(model_name)
    if selected not in MODEL_PROFILES:
        allowed = ", ".join(sorted(MODEL_PROFILES.keys()))
        msg = f"Unknown model_profile '{selected}'. Allowed values: {allowed}"
        raise ValueError(msg)
    return MODEL_PROFILES[selected]


async def _get_sandbox_context(backend: "HarborSandbox") -> dict[str, str | None]:
    """Gather comprehensive context from the sandbox environment.

    This mirrors LocalContextMiddleware but queries the sandbox instead of local filesystem.
    Provides git info, package managers, directory tree, Makefile preview, and test commands.

    Args:
        backend: Harbor sandbox backend to query

    Returns:
        Dict with context fields: current_dir, git_branch, git_main_branches,
        package_manager, directory_tree, makefile_preview, test_command, language
    """
    context: dict[str, str | None] = {
        "current_dir": None,
        "git_branch": None,
        "git_main_branches": None,
        "package_manager": None,
        "directory_tree": None,
        "makefile_preview": None,
        "test_command": None,
        "language": None,
        "python_path": None,
        "test_files": None,
        "available_tools": None,
    }

    # Get current directory
    pwd_result = await backend.aexecute("pwd")
    context["current_dir"] = pwd_result.output.strip() if pwd_result.output else "/app"

    # Get git info
    git_branch_result = await backend.aexecute("git rev-parse --abbrev-ref HEAD 2>/dev/null || echo ''")
    if git_branch_result.output and git_branch_result.output.strip():
        context["git_branch"] = git_branch_result.output.strip()

        # Check for main/master branches
        git_branches_result = await backend.aexecute("git branch 2>/dev/null | tr -d ' *' || echo ''")
        if git_branches_result.output:
            branches = set(git_branches_result.output.strip().split("\n"))
            main_branches = []
            if "main" in branches:
                main_branches.append("main")
            if "master" in branches:
                main_branches.append("master")
            if main_branches:
                context["git_main_branches"] = ", ".join(main_branches)

    # Detect package manager and language
    pkg_check = await backend.aexecute(
        "ls -la 2>/dev/null | grep -E '(uv.lock|poetry.lock|Pipfile|requirements.txt|package.json|Cargo.toml|go.mod|pom.xml|build.gradle)' || echo ''"
    )
    if pkg_check.output:
        output = pkg_check.output
        if "uv.lock" in output:
            context["package_manager"] = "uv"
            context["language"] = "Python"
        elif "poetry.lock" in output:
            context["package_manager"] = "poetry"
            context["language"] = "Python"
        elif "Pipfile" in output:
            context["package_manager"] = "pipenv"
            context["language"] = "Python"
        elif "requirements.txt" in output:
            context["package_manager"] = "pip"
            context["language"] = "Python"
        elif "package.json" in output:
            # Check for specific Node package managers
            node_lock_check = await backend.aexecute(
                "ls 2>/dev/null | grep -E '(bun.lock|pnpm-lock.yaml|yarn.lock|package-lock.json)' || echo ''"
            )
            if node_lock_check.output:
                if "bun.lock" in node_lock_check.output:
                    context["package_manager"] = "bun"
                elif "pnpm-lock.yaml" in node_lock_check.output:
                    context["package_manager"] = "pnpm"
                elif "yarn.lock" in node_lock_check.output:
                    context["package_manager"] = "yarn"
                else:
                    context["package_manager"] = "npm"
            else:
                context["package_manager"] = "npm"
            context["language"] = "JavaScript/TypeScript"
        elif "Cargo.toml" in output:
            context["package_manager"] = "cargo"
            context["language"] = "Rust"
        elif "go.mod" in output:
            context["package_manager"] = "go mod"
            context["language"] = "Go"
        elif "pom.xml" in output or "build.gradle" in output:
            context["package_manager"] = "maven/gradle"
            context["language"] = "Java"

    # Get directory tree (5 levels, max 60 entries)
    # Show everything. Only NO_RECURSE_DIRS (node_modules, .git) are shown
    # but not recursed into since they can contain thousands of files.
    no_recurse_args = " ".join(
        f"\\( -name '{d}' -type d -print -prune \\) -o" for d in NO_RECURSE_DIRS
    )
    tree_result = await backend.aexecute(
        f"find . -maxdepth 5 {no_recurse_args} -print 2>/dev/null | head -60 | sort"
    )
    if tree_result.output and tree_result.output.strip():
        context["directory_tree"] = tree_result.output.strip()

    # Get Makefile preview (first 15 lines)
    makefile_result = await backend.aexecute("head -15 Makefile 2>/dev/null || echo ''")
    if makefile_result.output and makefile_result.output.strip() and "No such file" not in makefile_result.output:
        context["makefile_preview"] = makefile_result.output.strip()

    # Detect test command
    if context["makefile_preview"] and ("test:" in context["makefile_preview"] or "tests:" in context["makefile_preview"]):
        context["test_command"] = "make test"
    elif context["language"] == "Python":
        pytest_check = await backend.aexecute("ls pytest.ini pyproject.toml 2>/dev/null || echo ''")
        if pytest_check.output and ("pytest.ini" in pytest_check.output or "pyproject.toml" in pytest_check.output):
            context["test_command"] = "pytest"
    elif context["language"] == "JavaScript/TypeScript":
        context["test_command"] = "npm test"
    elif context["language"] == "Rust":
        context["test_command"] = "cargo test"
    elif context["language"] == "Go":
        context["test_command"] = "go test ./..."

    # Discover Python interpreter (may be in non-standard paths like uv cache)
    python_result = await backend.aexecute(
        "which python3 2>/dev/null || which python 2>/dev/null || "
        "find /root -maxdepth 5 -name 'python3' -type f 2>/dev/null | head -1 || "
        "find /usr -maxdepth 4 -name 'python3' -type f 2>/dev/null | head -1 || echo ''"
    )
    if python_result.output and python_result.output.strip():
        context["python_path"] = python_result.output.strip().split("\n")[0]

    # Detect test/verifier files so the agent knows to read them
    test_result = await backend.aexecute(
        "ls /tests/test_outputs.py tests/test_outputs.py test_outputs.py "
        "/tests/test_*.py tests/test_*.py 2>/dev/null | head -5 || echo ''"
    )
    if test_result.output and test_result.output.strip():
        context["test_files"] = test_result.output.strip()

        # Read test file contents so the agent sees the exact interface
        # without needing an extra tool call (which it consistently skips)
        test_contents_parts = []
        for tf in context["test_files"].split("\n"):
            tf = tf.strip()
            if not tf:
                continue
            content_result = await backend.aexecute(
                f"head -80 '{tf}' 2>/dev/null || echo ''"
            )
            if content_result.output and content_result.output.strip():
                test_contents_parts.append(
                    f"### {tf}\n```\n{content_result.output.strip()}\n```"
                )
        if test_contents_parts:
            context["test_file_contents"] = "\n\n".join(test_contents_parts)

    # Discover key available tools (compact: just names, one line)
    tools_result = await backend.aexecute(
        "echo $(for cmd in gcc g++ make cmake rustc cargo go node npm java javac "
        "docker ffmpeg curl wget git sqlite3 R Rscript ocaml coq ghc; do "
        "which $cmd >/dev/null 2>&1 && echo $cmd; done)"
    )
    if tools_result.output and tools_result.output.strip():
        context["available_tools"] = tools_result.output.strip()

    return context


def _format_environment_context(context: dict[str, str | None], file_listing: str) -> str:
    """Format dynamic sandbox environment info for middleware injection.

    This produces the per-task environment context that gets injected as a
    HumanMessage via LocalContextMiddleware.before_agent. Contains only
    dynamic, task-specific information discovered from the sandbox.

    Args:
        context: Context dict from _get_sandbox_context
        file_listing: Formatted file listing string

    Returns:
        Formatted environment context string for middleware injection
    """
    sections = ["## Sandbox Environment", ""]

    # Current directory
    sections.append(f"**Current Directory**: `{context['current_dir']}`")
    sections.append("")

    # Language and package manager
    if context["language"] or context["package_manager"]:
        project_lines = []
        if context["language"]:
            project_lines.append(f"Language: {context['language']}")
        if context["package_manager"]:
            project_lines.append(f"Package manager: {context['package_manager']}")
        sections.append("**Project**:")
        sections.extend(f"- {line}" for line in project_lines)
        sections.append("")

    # Git info
    if context["git_branch"]:
        git_text = f"**Git**: Current branch `{context['git_branch']}`"
        if context["git_main_branches"]:
            git_text += f", main branches available: {context['git_main_branches']}"
        sections.append(git_text)
        sections.append("")

    # Python interpreter
    if context.get("python_path"):
        sections.append(f"**Python**: `{context['python_path']}`")
        sections.append("")

    # Available tools
    if context.get("available_tools"):
        sections.append(f"**Available tools**: {context['available_tools']}")
        sections.append("")

    # Test command
    if context["test_command"]:
        sections.append(f"**Run Tests**: `{context['test_command']}`")
        sections.append("")

    # Test/verifier files — inject contents so the agent sees the exact interface
    if context.get("test_file_contents"):
        sections.append("**Verifier test files** (use these to understand the exact interface — field names, CLI flags, file paths, output formats):")
        sections.append(context["test_file_contents"])
        sections.append("")
    elif context.get("test_files"):
        sections.append(f"**Verifier tests found**: `{context['test_files']}`")
        sections.append("Read these files FIRST to understand the exact interface expected.")
        sections.append("")

    # File listing
    if file_listing:
        sections.append("**Files in current directory**:")
        sections.append(file_listing)
        sections.append("")

    # Directory tree
    if context["directory_tree"]:
        sections.append("**Directory Tree** (5 levels):")
        sections.append("```")
        sections.append(context["directory_tree"])
        sections.append("```")
        sections.append("")

    # Makefile preview
    if context["makefile_preview"]:
        sections.append("**Makefile** (first 15 lines):")
        sections.append("```makefile")
        sections.append(context["makefile_preview"])
        sections.append("```")
        sections.append("")

    # Important notes
    sections.append("This context is provided at task start. Avoid redundant `ls` or `pwd` commands.")
    sections.append("Work in the current directory unless explicitly instructed otherwise.")

    return "\n".join(sections)


def _format_static_rules() -> str:
    """Format task-independent benchmark rules for the system prompt.

    These are static rules that must persist across summarization:
    integrity rules, sandbox behavior reminders, and time management.
    They stay in the system prompt which is prepended fresh on every
    model call, ensuring they are never lost to context compression.

    Returns:
        Formatted static rules string for the system prompt
    """
    return """---

## INTEGRITY RULES (VIOLATIONS = AUTOMATIC FAILURE)

1. **NO DATA FABRICATION**: NEVER invent, guess, or fabricate data. If you cannot recover/extract actual data, report the failure. Do not make up values.

2. **NO TEST OVERFITTING**: Your solution must work for ANY valid input, not just test examples. Do NOT hardcode test values as answers. You SHOULD read test files to understand the expected interface (field names, CLI flags, file paths, output formats) — that is not overfitting.

3. **NO BENCHMARK GAMING**: Do NOT modify timing functions, measurement code, or evaluation logic. Optimize the actual algorithm, not the measurement.

4. **NO INTERFACE BYPASS**: When a task specifies using a particular API/function, use ONLY that interface. Do NOT read source code or access internal variables to bypass constraints.

5. **NO SERVICE KILLING**: When you start a server/service for a task, do NOT terminate it. Verification needs it running. Use background execution (`&`, `nohup`, daemon mode).

6. **NO SOLUTION HUNTING**: Do NOT use web search to find solutions to this task. Specifically:
   - Do NOT search for the task name or benchmark name
   - Do NOT fetch source code that solves this task from GitHub or other repos
   - Do NOT access skill marketplaces, solution databases, or benchmark answer sites
   - Do NOT search for "how to solve [this problem]" to find existing implementations

   You MAY use web search for:
   - Official documentation (language docs, library APIs, tool manuals)
   - Technical references (RFCs, specifications, algorithm descriptions)
   - Downloading source code/data that the task explicitly requires

   Finding and copying existing solutions is cheating. Solve the task yourself.

---

## SANDBOX REMINDERS

### BACKGROUND PROCESSES DON'T PERSIST
Verification runs in a SEPARATE process. Background servers won't be running.
- Verify servers work BEFORE you finish (curl them)
- Or create a startup script the verifier can run

### PREFER STDLIB AND CLI TOOLS
The sandbox may lack third-party packages. Prefer standard library and CLI tools already present.
```
# BAD: pip install some-library → may not be available
# GOOD: use built-in modules or CLI tools already on the system
```

### OPTIMIZATION TASKS
For any task asking you to "optimize" or "improve performance":
- ALWAYS measure BEFORE making changes
- ALWAYS measure AFTER to verify improvement
- Do NOT assume any particular optimization helps - MEASURE
- Profile first to understand what's actually slow

---

## TIME MANAGEMENT

You have a LIMITED time budget. Commands that hang waste your entire budget.

**Package installation:**
1. Check if already installed: `which <tool>` or `dpkg -l | grep <pkg>`
2. Use `apt-get install -y --no-install-recommends <pkg>` (skip suggested packages)
3. If apt-get hangs >3 min, KILL IT and try: pre-built binary, pip, or download to /usr/local
4. NEVER run `apt-get upgrade` or `apt-get dist-upgrade` — these take forever and break things

**Before compiling from source:**
1. Check if binary already exists: `which <binary>` or `find / -name <binary> 2>/dev/null`
2. Check if installable via apt/pip instead: `apt-cache search <name>`
3. Large C++ projects (Caffe, OpenCV, LLVM, Doom) can take 30+ min to compile. Use pre-built packages or binaries whenever possible.

**Hung commands (>2 min with no output):**
- It is stuck. Kill it immediately. Try a different approach.
- Common culprits: apt-get (lock contention), make (insufficient RAM), pip (building C extensions from source)
- Use timeouts: `timeout 120 make -j$(nproc)` to auto-kill stuck builds

---

## COMMON PITFALLS

- When creating CLI scripts, use argparse with descriptive named flags (`--output_path`, `--csv_path`) rather than positional arguments.
- Use the exact libraries and packages mentioned in the task description, not alternatives.
- When building from source, complete the full pipeline: configure, build, install, then test end-to-end.
- Before finishing, remove any temporary or test files you created during development. Only required output files should remain.
- Before writing parsing or matching logic, EXAMINE at least 5-10 lines of the actual input data. Do not assume formats — check delimiters, quoting, whitespace, and edge cases.
- For image inputs, use the `open_image` tool. Use `read_file` for text/code files only.
- When a task lists multiple categories to find or fix, systematically check EACH one. Do not stop after finding the first.
- If the task specifies an execution command (e.g., `uv run`, `make test`), test with THAT EXACT command, not a substitute.
- When extracting or copying files, verify filenames and case match what is expected (some systems produce uppercase, tests may expect lowercase, or vice versa).
- Do NOT stop or kill servers/services at the end of your work. External verification needs them running.

---

## EFFECTIVE PROBLEM-SOLVING PATTERNS

### Diagnose before retrying
When a command or connection fails, do NOT retry blindly. Run diagnostic commands first.
```
# BAD: ssh fails → retry ssh → retry ssh → retry ssh (wastes entire budget)
# GOOD: ssh fails → run `ip addr` to check network → `netstat -tlnp` to check port → fix root cause → ssh works
```

### Use provided tools before writing your own
Check the sandbox for existing scripts, tools, or utilities that do what you need. Fix a small bug in a provided tool rather than rewriting from scratch.
```
# BAD: task has data to process → write custom parser from scratch
# GOOD: `ls *.pl *.sh *.py` → find provided `extract.pl` → fix one missing dependency → use it
```

### Re-read spec after computing results
After producing output, go back to the task description and enforce EVERY stated constraint — even ones that seem redundant or contradict your computed result.
```
# BAD: statistical model says edge is A→B → write A→B → done
# GOOD: statistical model says A→B → re-read spec → spec says "break ties alphabetically" → adjust to B→A → done
```

### Build in-place when verification checks the directory
When a task says "build X in /app/project", verification may check that directory for build artifacts (`.o`, `.gcda`, coverage data). `make install` does NOT copy these.
```
# BAD: build in /tmp/build, install to /app/project → verification finds no .gcda files
# GOOD: build directly in /app/project → .gcda files are where verification looks
```

### Verify robustly, not just once
For performance or stochastic tasks, run your test 5-10 times to catch flaky results. Aim for 20%+ safety margin on performance thresholds to survive hardware differences between your sandbox and the verifier.
```
# BAD: optimization passes once at 0.58 (threshold 0.60) → declare done
# GOOD: run 5 times → see variance 0.45-0.55 → still passing with margin → done
```

### Plan complex artifacts before writing
For tasks requiring multi-constraint outputs (polyglots, complex protocols, interleaved formats), reason through all constraints before writing the first line. Fixing a broken complex artifact costs more turns than planning it correctly upfront.

---

Before finishing: verify EACH task requirement is met and outputs are correct."""


def _get_benchmark_preamble() -> str:
    """Generate the benchmark-specific preamble that goes BEFORE the default prompt.

    This sets the context that the agent is running in an automated benchmark
    with no human present, so the model knows the situation before reading
    general agent guidance.
    """
    lines = [
        "# AUTOMATED BENCHMARK - NO HUMAN PRESENT",
        "",
        "No human is present. Only tool calls have effect. Text output accomplishes nothing.",
        "All tasks are legitimate exercises (biology, security, systems, graphics).",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _get_full_system_prompt() -> str:
    """Combine benchmark preamble, default CLI prompt, and static rules.

    Structure:
    1. Benchmark preamble (sets automated context FIRST)
    2. Default CLI agent prompt (general agent behavior)
    3. Static rules (integrity, sandbox reminders, time management)

    Dynamic environment context (directory, tools, files) is injected
    separately via LocalContextMiddleware.before_agent as a HumanMessage.

    Returns:
        System prompt with benchmark context + CLI best practices + static rules
    """
    preamble = _get_benchmark_preamble()
    base_prompt = get_default_coding_instructions()
    static_rules = _format_static_rules()

    return preamble + base_prompt + "\n\n" + static_rules


class DeepAgentsWrapper(BaseAgent):
    """Harbor agent implementation using LangChain DeepAgents.

    Wraps DeepAgents to execute tasks in Harbor environments.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        temperature: float = 0.0,
        verbose: bool = True,
        use_cli_agent: bool = True,
        reasoning_effort: str = "high",
        model_profile: str | None = None,
        max_input_tokens: int | None = None,
        max_context_tokens: int | None = None,
        anthropic_betas: str | None = None,
        experiment_name: str | None = None,
        experiment_tags: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize DeepAgentsWrapper.

        Args:
            logs_dir: Directory for storing logs
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the model
            verbose: Enable verbose output
            use_cli_agent: If True, use create_cli_agent from deepagents-cli (default).
                          If False, use create_deep_agent from SDK.
            reasoning_effort: Reasoning effort level for OpenAI reasoning models
                             ("low", "medium", "high", "xhigh"). Default: "high"
            model_profile: Optional harness runtime profile. If omitted, inferred from
                          model name. Allowed: default, openai_reasoning, anthropic_opus.
            max_input_tokens: Optional override for model input context size used by
                             summarization trigger logic.
            max_context_tokens: Optional override for ContextBudgetMiddleware.
            anthropic_betas: Optional comma-separated Anthropic beta flags to pass
                            to ChatAnthropic (e.g., "context-1m-2025-08-07").
            experiment_name: Name for this experiment run (for LangSmith grouping)
            experiment_tags: Comma-separated tags for this experiment (for LangSmith filtering)
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if model_name is None:
            # Use DeepAgents default
            model_name = "anthropic:claude-sonnet-4-5-20250929"

        self._model_name = model_name
        self._temperature = temperature
        self._verbose = verbose
        self._use_cli_agent = use_cli_agent
        self._reasoning_effort = reasoning_effort
        profile = _resolve_model_profile(model_name=model_name, profile_name=model_profile)
        self._model_profile = profile.name
        self._profile = profile
        self._max_input_tokens = max_input_tokens or profile.max_input_tokens
        self._max_context_tokens = max_context_tokens or profile.max_context_tokens
        if self._max_input_tokens <= 0:
            msg = f"max_input_tokens must be > 0 (got {self._max_input_tokens})"
            raise ValueError(msg)
        if self._max_context_tokens <= 0:
            msg = f"max_context_tokens must be > 0 (got {self._max_context_tokens})"
            raise ValueError(msg)
        if self._max_context_tokens >= self._max_input_tokens:
            msg = (
                "max_context_tokens must be smaller than max_input_tokens "
                f"(got {self._max_context_tokens} >= {self._max_input_tokens})"
            )
            raise ValueError(msg)
        self._experiment_name = experiment_name
        # Parse comma-separated tags into list
        self._experiment_tags = [t.strip() for t in experiment_tags.split(",")] if experiment_tags else []
        self._anthropic_betas = _resolve_anthropic_betas(
            model_name=model_name,
            anthropic_betas=anthropic_betas,
        )

        # Configure model with provider-specific settings
        model_kwargs: dict = {"temperature": temperature}

        # Per-request timeout + retry: prevents a single hung API call from
        # consuming the entire agent timeout budget.
        model_kwargs["timeout"] = 300  # 5 min per API call (xhigh can be slow)
        model_kwargs["max_retries"] = 2  # 3 total attempts on timeout/5xx/429

        # GPT-5.2-codex and similar reasoning models need Responses API
        model_lower = model_name.lower() if model_name else ""
        if "codex" in model_lower or "gpt-5" in model_lower:
            model_kwargs["use_responses_api"] = True
            model_kwargs["reasoning_effort"] = reasoning_effort
        if "claude" in model_lower and self._anthropic_betas:
            model_kwargs["betas"] = self._anthropic_betas

        self._model = init_chat_model(model_name, **model_kwargs)
        profile = getattr(self._model, "profile", {})
        if not isinstance(profile, dict):
            profile = {}
        profile["max_input_tokens"] = self._max_input_tokens
        self._model.profile = profile

        # LangSmith run tracking for feedback
        self._langsmith_run_id: str | None = None
        self._task_name: str | None = None

        # Job-level metadata for grouping runs in LangSmith
        # Use HARBOR_JOB_ID env var if set, otherwise generate a unique ID
        self._job_id = os.environ.get("HARBOR_JOB_ID") or f"job-{uuid.uuid4().hex[:8]}"
        self._job_start_time = datetime.now(timezone.utc).isoformat()

        # Build instruction->example_id mapping if LANGSMITH_EXPERIMENT is set
        self._instruction_to_example_id: dict[str, str] = {}
        langsmith_experiment_name = os.environ.get("LANGSMITH_EXPERIMENT", "").strip() or None
        if langsmith_experiment_name:
            try:
                client = Client()
                experiment = client.read_project(project_name=langsmith_experiment_name)
                examples = list(client.list_examples(dataset_id=experiment.reference_dataset_id))

                # Build mapping from instruction to example ID
                for example in examples:
                    instruction = example.inputs.get("instruction") if example.inputs else None
                    if instruction:
                        self._instruction_to_example_id[instruction] = str(example.id)
            except Exception as e:
                # Log error but don't fail initialization
                print(f"Warning: Failed to build instruction->example_id mapping: {e}")

    @staticmethod
    def name() -> str:
        return "deepagent-harbor"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent with the given environment.

        Args:
            environment: Harbor environment (Docker, Modal, etc.)
        """
        pass

    def version(self) -> str | None:
        """The version of the agent."""
        return "0.0.1"

    async def _get_formatted_system_prompt(
        self, backend: HarborSandbox
    ) -> tuple[str, str]:
        """Build the system prompt and environment context separately.

        The system prompt contains static content (preamble, CLI prompt,
        integrity rules, sandbox reminders, time management) that persists
        across summarization since it's prepended fresh on every model call.

        The environment context contains dynamic per-task info (directory,
        tools, files, git, etc.) that gets injected as a HumanMessage via
        LocalContextMiddleware.before_agent.

        Args:
            backend: Harbor sandbox backend to query for context

        Returns:
            Tuple of (system_prompt, environment_context)
        """
        # Gather comprehensive context from the sandbox
        context = await _get_sandbox_context(backend)

        # Get file listing
        ls_info = await backend.als_info(".")
        total_files = len(ls_info) if ls_info else 0
        first_15_files = ls_info[:15] if ls_info else []

        # Build file listing
        if total_files == 0:
            file_listing = "(empty directory)"
        else:
            file_lines = [f"- {file}" for file in first_15_files]
            if total_files > 15:
                file_lines.append(f"... ({total_files - 15} more files)")
            file_listing = "\n".join(file_lines)

        # Format the dynamic environment context (for middleware injection)
        environment_context = _format_environment_context(context, file_listing)

        # Build the static system prompt (preamble + CLI + rules)
        system_prompt = _get_full_system_prompt()

        return system_prompt, environment_context

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Execute the Deep Agent on the given instruction.

        Args:
            instruction: The task to complete
            environment: Harbor environment (Docker, Modal, etc.)
            context: Context to populate with metrics
        """
        configuration = json.loads(environment.trial_paths.config_path.read_text())
        if not isinstance(configuration, dict):
            raise AssertionError(
                f"Unexpected configuration format. Expected a dict got {type(configuration)}."
            )

        backend = HarborSandbox(environment)

        # Build system prompt (static rules) and environment context (dynamic)
        harbor_system_prompt, environment_context = (
            await self._get_formatted_system_prompt(backend)
        )

        # Create agent based on mode (CLI vs SDK)
        if self._use_cli_agent:
            # Sanitize session_id for use as assistant_id
            # deepagents_cli validates with regex ^[a-zA-Z0-9_\-\s]+$ which doesn't allow periods
            sanitized_assistant_id = environment.session_id.replace(".", "-")

            # Build tools list - include web tools if Tavily is configured
            harbor_tools = [http_request, fetch_url]
            if settings.has_tavily:
                harbor_tools.append(web_search)

            # Calculate time budget for adaptive reasoning
            # Priority: CLI override > task.toml timeout > 900s default
            agent_cfg = configuration.get("agent", {})
            base_timeout = agent_cfg.get("override_timeout_sec")
            if not base_timeout:
                base_timeout = _get_task_timeout_from_cache(
                    configuration.get("task", {}).get("path", "")
                )
            timeout_multiplier = configuration.get("timeout_multiplier", 1.0)
            total_budget_sec = base_timeout * timeout_multiplier

            # Build middleware stack for Harbor benchmarks
            harbor_middleware = [
                LocalContextMiddleware(environment_context),  # Inject env context once at start
                AdaptiveReasoningMiddleware(         # xhigh → high at 33% of time budget
                    model=self._model,
                    total_budget_sec=total_budget_sec,
                    switch_fraction=0.33,
                ),
                APIErrorRecoveryMiddleware(),      # Catch and recover from API errors
                LoopDetectionMiddleware(
                    soft_warning_threshold=self._profile.loop_soft_warning_threshold,
                    hard_reflection_threshold=self._profile.loop_hard_reflection_threshold,
                ),  # Detect stuck editing loops
                ContextBudgetMiddleware(
                    max_context_tokens=self._max_context_tokens,
                    max_output_lines=self._profile.context_max_output_lines,
                    warn_threshold_percent=self._profile.context_warn_threshold_percent,
                ),  # Prevent context overflow
                PreCompletionCheckMiddleware(model=self._model),  # Enforce checklist before finishing (xhigh boost)
            ]

            deep_agent, _ = create_cli_agent(
                model=self._model,
                assistant_id=sanitized_assistant_id,
                tools=harbor_tools,
                sandbox=backend,
                sandbox_type=None,
                system_prompt=harbor_system_prompt,  # Static rules only
                auto_approve=True,  # Skip HITL in Harbor
                enable_memory=False,
                enable_skills=False,  # Disable CLI skills for now
                enable_shell=False,  # Sandbox provides execution
                middleware=harbor_middleware,
            )
        else:
            # Use SDK agent — no middleware support, combine everything
            sdk_prompt = harbor_system_prompt + "\n\n" + environment_context

            deep_agent = create_deep_agent(
                model=self._model, backend=backend, system_prompt=sdk_prompt
            )

        # Extract task name from session_id (format: "task-name__randomId")
        task_name = environment.session_id.rsplit("__", 1)[0] if "__" in environment.session_id else environment.session_id

        # Build metadata with experiment tracking info
        metadata = {
            "task_instruction": instruction,
            "model": self._model_name,
            "reasoning_effort": self._reasoning_effort,
            "model_profile": self._model_profile,
            "max_input_tokens": self._max_input_tokens,
            "max_context_tokens": self._max_context_tokens,
            "anthropic_betas": self._anthropic_betas,
            # Job-level identifiers for grouping runs in LangSmith
            "job_id": self._job_id,
            "job_start_time": self._job_start_time,
            # Experiment tracking
            "experiment_name": self._experiment_name,
            "experiment_tags": self._experiment_tags,
            # Task-level identifiers
            "task_name": task_name,
            "harbor_session_id": environment.session_id,
            # Tag to indicate which agent implementation is being used
            "agent_mode": "cli" if self._use_cli_agent else "sdk",
        }
        metadata.update(configuration)

        # Look up example_id from instruction using the mapping built at initialization
        example_id = self._instruction_to_example_id.get(instruction)

        # Build tags list - include experiment tags for easy filtering
        run_tags = [
            self._model_name,
            self._job_id,
            task_name,
            "cli-agent" if self._use_cli_agent else "sdk-agent",
            f"reasoning:{self._reasoning_effort}",
            f"profile:{self._model_profile}",
            f"max_input:{self._max_input_tokens}",
            f"max_ctx:{self._max_context_tokens}",
        ]
        if self._experiment_name:
            run_tags.append(f"exp:{self._experiment_name}")
        run_tags.extend(self._experiment_tags)

        config: RunnableConfig = {
            "run_name": f"{task_name}",
            "tags": run_tags,
            "recursion_limit": self._profile.recursion_limit,
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            },
        }

        # If LANGSMITH_EXPERIMENT is set, wrap in trace context.
        # This will link runs to the given experiment in LangSmith.
        langsmith_experiment_name = os.environ.get("LANGSMITH_EXPERIMENT", "").strip() or None

        if langsmith_experiment_name:
            with trace(
                name=environment.session_id,
                reference_example_id=example_id,
                inputs={"instruction": instruction},
                project_name=langsmith_experiment_name,
                metadata=metadata,
            ) as run_tree:
                # Invoke deep agent with LangSmith tracing
                result = await deep_agent.ainvoke(
                    {"messages": [{"role": "user", "content": instruction}]},
                    config=config,
                )
                # Extract last AI message and add as output
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    run_tree.end(outputs={"last_message": last_message.text})
        else:
            config["metadata"] = metadata
            result = await deep_agent.ainvoke(
                {"messages": [{"role": "user", "content": instruction}]},
                config=config,
            )

        self._save_trajectory(environment, instruction, result)

    def _save_trajectory(
        self, environment: BaseEnvironment, instruction: str, result: dict
    ) -> None:
        """Save current trajectory to logs directory."""
        # Track token usage and cost for this run
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Create trajectory
        steps = [
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=instruction,
            ),
        ]

        observations = []
        pending_step: Step | None = None

        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                # Extract usage metadata from AIMessage
                usage: UsageMetadata = msg.usage_metadata
                if usage:
                    total_prompt_tokens += usage["input_tokens"]
                    total_completion_tokens += usage["output_tokens"]
                # If there's a pending step with tool calls, add it now with observations
                if pending_step is not None:
                    if pending_step.tool_calls and observations:
                        # Add observations to the pending step
                        pending_step.observation = Observation(results=observations)
                        observations = []
                    steps.append(pending_step)
                    pending_step = None

                # Extract content and tool calls from current AIMessage
                atf_tool_calls = []
                message = ""
                for cb in msg.content_blocks:
                    if cb["type"] == "text":
                        message += cb["text"]
                    elif cb["type"] == "reasoning":
                        # Handle both Anthropic and OpenAI reasoning formats
                        # Anthropic: {"type": "reasoning", "reasoning": "..."}
                        # OpenAI: {"type": "reasoning", "summary": [...], "id": "..."}
                        if "reasoning" in cb:
                            message += cb["reasoning"]
                        elif "summary" in cb and cb["summary"]:
                            message += " ".join(str(s) for s in cb["summary"])
                        # If neither, reasoning content may not be exposed - skip
                    elif cb["type"] == "tool_call":
                        atf_tool_calls.append(
                            ToolCall(
                                tool_call_id=cb["id"],
                                function_name=cb["name"],
                                arguments=cb["args"],
                            )
                        )
                    else:
                        # TODO: Add server side tool call results.
                        continue

                # Create new step
                new_step = Step(
                    step_id=steps[-1].step_id + 1 if steps else 0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    message=message,
                    tool_calls=atf_tool_calls if atf_tool_calls else None,
                )

                # If this AIMessage has tool calls, make it pending (wait for observations)
                # Otherwise, add it immediately
                if atf_tool_calls:
                    pending_step = new_step
                else:
                    steps.append(new_step)

            elif isinstance(msg, ToolMessage):
                # Collect observations for the pending step
                observations.append(
                    ObservationResult(
                        source_call_id=msg.tool_call_id,
                        content=str(msg.content),
                    )
                )
            elif isinstance(msg, HumanMessage):
                pass
            else:
                raise NotImplementedError(
                    f"Message type {type(msg)} not supported for step conversion"
                )

        # Add any remaining pending step
        if pending_step is not None:
            if pending_step.tool_calls and observations:
                pending_step.observation = Observation(results=observations)
            steps.append(pending_step)

        # Build and save trajectory
        metrics = FinalMetrics(
            total_prompt_tokens=total_prompt_tokens or None,
            total_completion_tokens=total_completion_tokens or None,
            total_steps=len(steps),
        )
        trajectory = Trajectory(
            schema_version="ATIF-v1.2",
            session_id=environment.session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self._model_name,
                extra={
                    "framework": "deepagents",
                    "langchain_version": "1.0+",
                },
            ),
            steps=steps,
            final_metrics=metrics,
        )
        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))
