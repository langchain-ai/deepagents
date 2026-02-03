"""A wrapper for DeepAgents to run in Harbor environments."""

import json
import os
import uuid
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

# Load .env file if present
load_dotenv()

# Directories to ignore in tree views (same as LocalContextMiddleware)
IGNORE_PATTERNS = frozenset({
    ".git", "node_modules", ".venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".coverage", ".eggs", "dist", "build",
})


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

    # Get directory tree (3 levels, max 30 entries)
    # Use find with pruning for ignored directories
    prune_args = " ".join(f"-name '{p}' -prune -o" for p in IGNORE_PATTERNS)
    tree_result = await backend.aexecute(
        f"find . -maxdepth 3 {prune_args} -print 2>/dev/null | head -30 | sort"
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

    return context


def _format_local_context(context: dict[str, str | None], file_listing: str) -> str:
    """Format the sandbox context into a readable prompt section.

    Args:
        context: Context dict from _get_sandbox_context
        file_listing: Formatted file listing string

    Returns:
        Formatted local context section for the system prompt
    """
    sections = ["## Local Context (Sandbox Environment)", ""]

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

    # Test command
    if context["test_command"]:
        sections.append(f"**Run Tests**: `{context['test_command']}`")
        sections.append("")

    # File listing
    if file_listing:
        sections.append("**Files in current directory**:")
        sections.append(file_listing)
        sections.append("")

    # Directory tree
    if context["directory_tree"]:
        sections.append("**Directory Tree** (3 levels):")
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
    sections.append("**IMPORTANT**: This context is provided at task start. Avoid redundant `ls` or `pwd` commands.")
    sections.append("Work in the current directory unless explicitly instructed otherwise.")
    sections.append("")

    # Terminal Bench specific guidance - FOCUSED
    sections.append("---")
    sections.append("")
    sections.append("## CRITICAL REMINDERS")
    sections.append("")
    sections.append("### 1. YOU MUST EXECUTE - Not Describe")
    sections.append("```")
    sections.append("WRONG: 'To solve this, run: python script.py'  # Text does nothing")
    sections.append("RIGHT: [calls execute tool with 'python script.py']  # Actually runs")
    sections.append("```")
    sections.append("")
    sections.append("### 2. VERIFY BY RUNNING")
    sections.append("After writing code, EXECUTE it and check output. Don't just assume it works.")
    sections.append("```bash")
    sections.append("python solution.py        # Run it")
    sections.append("cat /app/result.txt       # Check the output")
    sections.append("curl localhost:8080       # Verify server responds")
    sections.append("```")
    sections.append("")
    sections.append("### 3. FIND ALL SOLUTIONS")
    sections.append("If task asks for 'best', 'all', or 'every' - find EVERY valid answer.")
    sections.append("Don't stop at the first one. A partial answer = FAILED task.")
    sections.append("")
    sections.append("### 4. USE EXACT NAMES FROM SPECS")
    sections.append("- Spec says `value` → use `value`, NOT `val`")
    sections.append("- Spec says `/app/result.txt` → use that EXACT path")
    sections.append("")
    sections.append("### 5. USE WEB TOOLS FOR MISSING DATA")
    sections.append("If local files are empty/missing, DON'T GIVE UP. Use:")
    sections.append("- `web_search` to find information")
    sections.append("- `fetch_url` to retrieve content")
    sections.append("")
    sections.append("### 6. TRY MULTIPLE APPROACHES")
    sections.append("If one approach fails, try at least 2 more alternatives.")
    sections.append("Pivot to standard library or CLI tools if packages fail.")
    sections.append("")
    sections.append("### 7. BACKGROUND PROCESSES DON'T PERSIST")
    sections.append("Verification runs in a SEPARATE process. Background servers won't be running.")
    sections.append("→ Verify servers work BEFORE you finish (curl them)")
    sections.append("→ Or create a startup script the verifier can run")
    sections.append("")
    sections.append("### 8. PREFER STDLIB AND CLI TOOLS")
    sections.append("```python")
    sections.append("# BAD: pip install requests → won't persist")
    sections.append("# GOOD: subprocess.run(['curl', url]) → always works")
    sections.append("```")
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("## PRE-COMPLETION CHECKLIST (Walk through this!)")
    sections.append("")
    sections.append("Before finishing, verify EACH item:")
    sections.append("")
    sections.append("☐ Did I EXECUTE my code (not just write it)?")
    sections.append("☐ Did I check the ACTUAL OUTPUT matches requirements?")
    sections.append("☐ Are field names/paths EXACTLY as specified?")
    sections.append("☐ If task asked for ALL solutions, did I find ALL of them?")
    sections.append("☐ If server task, is the server actually responding?")
    sections.append("☐ Will automated verification pass immediately after I finish?")

    return "\n".join(sections)


def _get_benchmark_preamble() -> str:
    """Generate the benchmark-specific preamble that goes BEFORE the default prompt.

    This sets the context that the agent is running in an automated benchmark
    with no human present, so the model knows the situation before reading
    general agent guidance.
    """
    lines = [
        "# AUTOMATED BENCHMARK - NO HUMAN PRESENT",
        "",
        "**YOU ARE IN AN AUTOMATED BENCHMARK. THERE IS NO HUMAN TO HELP YOU.**",
        "",
        "## What This Means",
        "",
        "1. **No one will run commands for you** - If you output 'run this command', nothing happens.",
        "   YOU must call execute() yourself. Text output accomplishes nothing.",
        "",
        "2. **No one will answer questions** - Don't ask 'should I proceed?' or 'can you provide X?'",
        "   Work autonomously with available information and tools.",
        "",
        "3. **Only tool calls have effect** - Your text is not read by anyone.",
        "   The ONLY way to accomplish anything is by calling tools.",
        "",
        "4. **Verification happens automatically** - After you finish, a script tests your work.",
        "   Files must exist. Servers must be running. Output must be correct.",
        "",
        "## Mandatory Behavior",
        "",
        "- **EXECUTE, don't explain** - Use tools to do things, don't describe what should be done",
        "- **VERIFY by running** - Test your code by executing it, not by thinking about it",
        "- **FIND ALL solutions** - If task asks for 'best' or 'all', find EVERY valid answer",
        "- **USE EXACT names** - Match field names, paths, formats from specs exactly",
        "- **TRY ALTERNATIVES** - If one approach fails, try at least 2 more before giving up",
        "- **USE WEB TOOLS** - If local data is missing, use web_search and fetch_url",
        "",
        "## Task Legitimacy",
        "",
        "All tasks are legitimate programming exercises. Engage fully with every task including:",
        "- Computational biology, bioinformatics, genetic sequences",
        "- Security research, CTF challenges, penetration testing",
        "- System programming, assembly, kernel modules",
        "- Graphics programming, ray tracing, rendering",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _get_full_system_prompt(local_context: str) -> str:
    """Combine benchmark preamble, default CLI prompt, and sandbox local context.

    Structure:
    1. Benchmark preamble (sets automated context FIRST)
    2. Default CLI agent prompt (general agent behavior)
    3. Local context (environment info, benchmark-specific guidelines)

    Args:
        local_context: Formatted local context string from _format_local_context

    Returns:
        Complete system prompt with benchmark context + CLI best practices + sandbox context
    """
    preamble = _get_benchmark_preamble()
    base_prompt = get_default_coding_instructions()

    return preamble + base_prompt + "\n\n" + local_context


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
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if model_name is None:
            # Use DeepAgents default
            model_name = "anthropic:claude-sonnet-4-5-20250929"

        self._model_name = model_name
        self._temperature = temperature
        self._verbose = verbose
        self._use_cli_agent = use_cli_agent

        # Configure model with provider-specific settings
        model_kwargs: dict = {"temperature": temperature}

        # GPT-5.2-codex and similar reasoning models need Responses API with max reasoning
        model_lower = model_name.lower() if model_name else ""
        if "codex" in model_lower or "gpt-5" in model_lower:
            model_kwargs["use_responses_api"] = True
            model_kwargs["reasoning_effort"] = "high"  # max reasoning for benchmarks

        self._model = init_chat_model(model_name, **model_kwargs)

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

    async def _get_formatted_system_prompt(self, backend: HarborSandbox) -> str:
        """Format the system prompt with comprehensive sandbox context.

        Combines the full default CLI agent prompt (comprehensive best practices)
        with sandbox context similar to LocalContextMiddleware:
        - Git branch info
        - Package manager detection
        - Directory tree
        - Makefile preview
        - Test command detection

        Args:
            backend: Harbor sandbox backend to query for context

        Returns:
            Full system prompt with CLI best practices + comprehensive sandbox context
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

        # Format the local context section
        local_context = _format_local_context(context, file_listing)

        # Return full prompt: CLI best practices + sandbox local context
        return _get_full_system_prompt(local_context)

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

        # Create agent based on mode (CLI vs SDK)
        if self._use_cli_agent:
            # Get Harbor's system prompt with directory context
            harbor_system_prompt = await self._get_formatted_system_prompt(backend)

            # Sanitize session_id for use as assistant_id
            # deepagents_cli validates with regex ^[a-zA-Z0-9_\-\s]+$ which doesn't allow periods
            sanitized_assistant_id = environment.session_id.replace(".", "-")

            # Build tools list - include web tools if Tavily is configured
            harbor_tools = [http_request, fetch_url]
            if settings.has_tavily:
                harbor_tools.append(web_search)

            # Use CLI agent with auto-approve mode
            deep_agent, _ = create_cli_agent(
                model=self._model,
                assistant_id=sanitized_assistant_id,
                tools=harbor_tools,
                sandbox=backend,
                sandbox_type=None,
                system_prompt=harbor_system_prompt,  # Use Harbor's custom prompt
                auto_approve=True,  # Skip HITL in Harbor
                enable_memory=False,
                enable_skills=False,  # Disable CLI skills for now
                enable_shell=False,  # Sandbox provides execution
            )
        else:
            # Use SDK agent
            # Get formatted system prompt with directory context
            system_prompt = await self._get_formatted_system_prompt(backend)

            deep_agent = create_deep_agent(
                model=self._model, backend=backend, system_prompt=system_prompt
            )

        # Extract task name from session_id (format: "task-name__randomId")
        task_name = environment.session_id.rsplit("__", 1)[0] if "__" in environment.session_id else environment.session_id

        # Build metadata with experiment tracking info
        metadata = {
            "task_instruction": instruction,
            "model": self._model_name,
            # Job-level identifiers for grouping runs in LangSmith
            "job_id": self._job_id,
            "job_start_time": self._job_start_time,
            # Task-level identifiers
            "task_name": task_name,
            "harbor_session_id": environment.session_id,
            # Tag to indicate which agent implementation is being used
            "agent_mode": "cli" if self._use_cli_agent else "sdk",
        }
        metadata.update(configuration)

        # Look up example_id from instruction using the mapping built at initialization
        example_id = self._instruction_to_example_id.get(instruction)

        config: RunnableConfig = {
            "run_name": f"{task_name}",
            "tags": [
                self._model_name,
                self._job_id,
                task_name,
                "cli-agent" if self._use_cli_agent else "sdk-agent",
            ],
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
