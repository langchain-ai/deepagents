"""Pier agent adapter for running dcode (Deep Agents Code) under Pier.

Pier (Datacurve's Harbor fork) drives air-gapped DeepSWE tasks. This module
implements a ``Dcode`` agent that Pier's ``AgentFactory`` loads dynamically via
``pier run --agent-import-path deepagents_pier.agent:Dcode``. The agent runs
the existing ``deepagents_harbor/langgraph_project`` LangGraph project inside
the task container (mirroring Harbor's installed ``LangGraph`` agent), so the
agent's shell/filesystem tools act on the task's own filesystem, and only the
model-API endpoint needs a network hole.

Design notes:

- ``Dcode`` subclasses Pier's ``BaseAgent`` directly rather than
  ``BaseInstalledAgent``. ``BaseInstalledAgent`` is for agents that are CLI
  binaries installed into the image at build time; dcode is a LangGraph graph
  constructed in-process, so the declarative ``install_spec``/``CliFlag``
  machinery does not fit.
- The graph runner (``langgraph_runner.py``) is read from the installed
  ``harbor`` package at setup time and uploaded into the container, avoiding a
  vendored copy that would drift as Harbor evolves.
- ``network_allowlist()`` returns the model-API host (the LangSmith proxy for
  our setup), which is the single egress hole DeepSWE's air-gapped
  ``allow_internet = false`` containers require.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import shlex
import shutil
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pier.agents.base import BaseAgent
from pier.models.agent.network import NetworkAllowlist
from pier.models.trial.paths import EnvironmentPaths

if TYPE_CHECKING:
    from pier.environments.base import BaseEnvironment
    from pier.models.agent.context import AgentContext

logger = logging.getLogger(__name__)

# Cap on the agent summary sidecar we read back from the environment
# (untrusted input).
_MAX_SUMMARY_BYTES = 1_000_000

# Default LangGraph project (the dcode harness graph + its local deps).
_DEFAULT_PROJECT_PATH = (
    Path(__file__).resolve().parent.parent / "deepagents_harbor" / "langgraph_project"
)

# Env vars forwarded from the host process to the agent container so the
# graph can make LLM calls and emit LangSmith traces without requiring each
# one to be passed explicitly.
_FORWARDED_ENV_VARS = (
    # Model provider keys
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "FIREWORKS_API_KEY",
    "OPENROUTER_API_KEY",
    "BASETEN_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "NVIDIA_API_KEY",
    "OLLAMA_API_KEY",
    "OLLAMA_HOST",
    # Base URLs / proxies that determine the egress host
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "LANGSMITH_ENDPOINT",
    # LangSmith tracing
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGSMITH_PROJECT",
    "LANGSMITH_PROFILE",
    "LANGSMITH_DATASET",
    # Harbor → LangGraph distributed-tracing headers
    "HARBOR_LANGSMITH_PARENT",
    "HARBOR_LANGSMITH_BAGGAGE",
)

# Environment variables that carry the model-API base URL / endpoint, in
# priority order. The first one set determines the allowlist domain.
_BASE_URL_ENV_VARS = (
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "LANGSMITH_ENDPOINT",
)

# Files/dirs excluded when staging the project into the container.
_IGNORE_NAMES = {
    ".env",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
}


class Dcode(BaseAgent):
    """Run the dcode LangGraph harness inside a Pier task container."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    _REMOTE_PROJECT_DIR = PurePosixPath("/installed-agent/langgraph-project")
    _REMOTE_RUNNER_PATH = PurePosixPath("/installed-agent/langgraph_runner.py")
    _REMOTE_VENV_DIR = PurePosixPath("/opt/dcode-langgraph-venv")
    _REMOTE_INSTRUCTION_PATH = PurePosixPath("/installed-agent/instruction.txt")
    _RESULT_FILENAME = "result.json"
    _OUTPUT_FILENAME = "dcode.txt"
    _SUMMARY_FILENAME = "summary.json"

    def __init__(
        self,
        *args: Any,
        project_path: str | Path | None = None,
        graph: str = "dcode",
        config: str = "langgraph.json",
        python_version: str | float = "3.12",
        model_kwargs: dict[str, Any] | None = None,
        configurable: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dcode Pier agent.

        Args:
            project_path: Path to the LangGraph project directory. Defaults to
                the bundled ``deepagents_harbor/langgraph_project``. Its
                ``.local_deps`` must already be staged (``make
                stage-harbor-local-deps``).
            graph: The graph name in ``langgraph.json`` to run.
            config: The LangGraph config filename within the project.
            python_version: Interpreter version for the in-container venv.
            model_kwargs: Optional model constructor kwargs forwarded to the graph.
            configurable: Optional ``configurable`` mapping forwarded to the graph.
        """
        super().__init__(*args, **kwargs)
        self.project_path = (
            Path(project_path).expanduser().resolve()
            if project_path is not None
            else _DEFAULT_PROJECT_PATH
        )
        self.graph = graph
        self.config = config
        self.model_kwargs = model_kwargs or {}
        self.configurable = configurable or {}
        self._python_version = str(python_version)

        if not self.project_path.is_dir():
            msg = f"dcode project_path does not exist: {self.project_path}"
            raise ValueError(msg)
        config_path = (self.project_path / self.config).resolve()
        if not config_path.is_file():
            msg = f"LangGraph config file not found: {config_path}"
            raise ValueError(msg)
        self.config = config_path.relative_to(self.project_path).as_posix()

    @staticmethod
    def name() -> str:
        """Return the agent name."""
        return "dcode"

    def version(self) -> str | None:
        """Return the installed ``deepagents-code`` package version."""
        try:
            return importlib.metadata.version("deepagents-code")
        except Exception:  # noqa: BLE001 - version detection is best-effort
            return None

    def network_allowlist(self) -> NetworkAllowlist:
        """Return the model-API host this agent needs egress to.

        DeepSWE task containers are air-gapped (``allow_internet = false``), so
        the only egress required is to the model API. For our setup that is the
        LangSmith proxy host (from ``OPENAI_BASE_URL``/``LANGSMITH_ENDPOINT``),
        not ``api.openai.com``.
        """
        for var in _BASE_URL_ENV_VARS:
            raw = os.environ.get(var)
            if not raw:
                continue
            host = urlparse(raw).hostname
            if host:
                return NetworkAllowlist(domains=[host])
        return NetworkAllowlist()

    def _staged_project_dir(self) -> Path:
        """Copy the project into the logs dir, excluding local env/VCS noise."""
        target = self.logs_dir / "langgraph_project"
        if target.exists():
            shutil.rmtree(target)
        ignore = shutil.ignore_patterns(*_IGNORE_NAMES, ".env.*")
        shutil.copytree(self.project_path, target, ignore=ignore)
        return target

    def _load_runner_script(self) -> Path:
        """Read Harbor's LangGraph runner from the installed harbor package."""
        try:
            # Lazy: harbor supplies the runner script but pier drives the run, so
            # only the setup path (not module import) should require harbor.
            from harbor.agents.installed import langgraph_runner  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "The harbor package is required to source langgraph_runner.py; "
                "install harbor in the dcode Pier environment"
            )
            raise RuntimeError(msg) from exc
        runner_path = Path(langgraph_runner.__file__).resolve()
        local_copy = self.logs_dir / "langgraph_runner.py"
        local_copy.write_text(runner_path.read_text())
        return local_copy

    def _normalized_model_name(self) -> str | None:
        """Normalize ``provider/model`` to the ``provider:model`` form dcode uses."""
        if not self.model_name:
            return None
        if ":" in self.model_name:
            return self.model_name
        if "/" in self.model_name:
            provider, model = self.model_name.split("/", maxsplit=1)
            return f"{provider}:{model}"
        return self.model_name

    async def setup(self, environment: BaseEnvironment) -> None:
        """Stage the project and runner, then build the in-container venv."""
        agent_user = str(environment.default_user or "root")
        quoted_agent_user = shlex.quote(agent_user)
        staged_project = self._staged_project_dir()
        runner_copy = self._load_runner_script()

        await environment.exec(
            f"rm -rf {shlex.quote(self._REMOTE_PROJECT_DIR.as_posix())} "
            f"{shlex.quote(self._REMOTE_RUNNER_PATH.as_posix())} "
            f"{shlex.quote(self._REMOTE_VENV_DIR.as_posix())} && "
            f"mkdir -p {shlex.quote(self._REMOTE_PROJECT_DIR.as_posix())} "
            f"{shlex.quote(self._REMOTE_VENV_DIR.as_posix())}",
            user="root",
        )
        await environment.upload_dir(
            staged_project, self._REMOTE_PROJECT_DIR.as_posix()
        )
        await environment.upload_file(
            runner_copy, self._REMOTE_RUNNER_PATH.as_posix()
        )
        await environment.exec(
            f"chown -R {quoted_agent_user}:{quoted_agent_user} "
            f"{shlex.quote(self._REMOTE_PROJECT_DIR.as_posix())} "
            f"{shlex.quote(self._REMOTE_VENV_DIR.as_posix())}",
            user="root",
        )

        project_dir = shlex.quote(self._REMOTE_PROJECT_DIR.as_posix())
        venv_dir = shlex.quote(self._REMOTE_VENV_DIR.as_posix())
        python_version = shlex.quote(self._python_version)
        install_program = (
            "import json, subprocess, sys\n"
            f"project_dir = {project_dir!r}\n"
            f"config_name = {self.config!r}\n"
            "installer = ['uv', 'pip', 'install', '--prerelease=if-necessary']\n"
            "with open(__import__('os').path.join(project_dir, config_name)) as f:\n"
            "    config = json.load(f)\n"
            "for dep in config.get('dependencies', []):\n"
            "    dep_path = __import__('os').path.join(project_dir, dep) "
            "if isinstance(dep, str) else None\n"
            "    if dep_path and __import__('os').path.isdir(dep_path):\n"
            "        subprocess.check_call([*installer, '-e', dep_path])\n"
            "    elif isinstance(dep, str):\n"
            "        subprocess.check_call([*installer, dep])\n"
        )
        result = await environment.exec(
            "set -euo pipefail; "
            "curl -LsSf https://astral.sh/uv/install.sh | sh; "
            'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi; '
            'export PATH="$HOME/.local/bin:$PATH"; '
            f"uv python install {python_version}; "
            f"uv venv {venv_dir} --python {python_version} --clear; "
            f". {venv_dir}/bin/activate; "
            "uv pip install langgraph python-dotenv; "
            f"python - <<'PY'\n{install_program}PY",
        )
        if result.return_code != 0:
            stderr = result.stderr or result.stdout or "dcode env install failed"
            raise RuntimeError(stderr)

    def _runtime_env(self, environment: BaseEnvironment, model: str | None) -> dict[str, str]:
        """Build the runtime env forwarded into the container."""
        env = {
            "HARBOR_SESSION_ID": environment.session_id,
            "HARBOR_MODEL_KWARGS_JSON": json.dumps(self.model_kwargs),
        }
        if model:
            env["HARBOR_MODEL"] = model
        for var in _FORWARDED_ENV_VARS:
            value = os.environ.get(var)
            if value is not None and var not in env:
                env[var] = value
        return env

    def _runtime_configurable_json(self) -> str:
        """Forward task-declared MCP servers into the graph configurable."""
        configurable = dict(self.configurable)
        if self.mcp_servers and "mcp_servers" not in configurable:
            configurable["mcp_servers"] = [
                server.model_dump(mode="json") for server in self.mcp_servers
            ]
        return json.dumps(configurable)

    def _runner_args(self, model: str | None) -> list[str]:
        """Build the runner CLI args."""
        args = [
            "--project-dir",
            self._REMOTE_PROJECT_DIR.as_posix(),
            "--graph",
            self.graph,
            "--graph-path",
            self._graph_path(),
            "--instruction-file",
            self._REMOTE_INSTRUCTION_PATH.as_posix(),
            "--result-path",
            (EnvironmentPaths.agent_dir / self._RESULT_FILENAME).as_posix(),
            "--output-path",
            (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix(),
            "--summary-path",
            (EnvironmentPaths.agent_dir / self._SUMMARY_FILENAME).as_posix(),
        ]
        if model:
            args.extend(("--model", model))
        args.extend(
            (
                "--model-kwargs-json",
                json.dumps(self.model_kwargs),
                "--configurable-json",
                self._runtime_configurable_json(),
            )
        )
        return args

    def _graph_path(self) -> str:
        """Resolve the graph's module:attribute path from langgraph.json."""
        config_path = self.project_path / self.config
        config = json.loads(config_path.read_text())
        graphs = config.get("graphs")
        if not isinstance(graphs, dict) or self.graph not in graphs:
            available = ", ".join(sorted(graphs)) if isinstance(graphs, dict) else ""
            msg = f"Unknown graph {self.graph!r}. Available graphs: {available}"
            raise ValueError(msg)
        return graphs[self.graph]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the dcode graph on the task instruction and populate context."""
        instruction_path = self.logs_dir / "instruction.txt"
        instruction_path.write_text(instruction)
        await environment.upload_file(
            instruction_path, self._REMOTE_INSTRUCTION_PATH.as_posix()
        )

        model = self._normalized_model_name()
        env = self._runtime_env(environment, model)
        python = (self._REMOTE_VENV_DIR / "bin" / "python").as_posix()
        runner_command = shlex.join(
            [python, self._REMOTE_RUNNER_PATH.as_posix(), *self._runner_args(model)]
        )
        log_path = (EnvironmentPaths.agent_dir / "dcode-run.log").as_posix()
        result = await environment.exec(
            f"{runner_command} 2>&1 | stdbuf -oL tee {shlex.quote(log_path)}",
            env=env,
        )
        if result.return_code != 0:
            stderr = result.stderr or result.stdout or "dcode agent run failed"
            raise RuntimeError(stderr)

        context.metadata = {
            **(context.metadata or {}),
            "dcode_graph": self.graph,
            "dcode_config": self.config,
            "dcode_project_path": str(self.project_path),
        }
        await self._apply_run_summary(environment, context)

    async def _apply_run_summary(
        self, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Populate ``AgentContext`` from the runner's ``summary.json`` sidecar.

        Treated as untrusted input: size-capped, type-validated, and never fatal.
        """
        remote = (EnvironmentPaths.agent_dir / self._SUMMARY_FILENAME).as_posix()
        local = self.logs_dir / self._SUMMARY_FILENAME
        try:
            await environment.download_file(remote, local)
            raw = local.read_text()
            if len(raw) > _MAX_SUMMARY_BYTES:
                logger.warning(
                    "dcode run summary %s exceeds %d bytes; skipping",
                    remote,
                    _MAX_SUMMARY_BYTES,
                )
                return
            summary = json.loads(raw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - sidecar is best-effort, never fatal
            logger.warning("Could not read dcode run summary %s: %s", remote, exc)
            return

        if not isinstance(summary, dict):
            logger.warning("dcode run summary %s is not a JSON object", remote)
            return

        answer = summary.get("answer_written")
        if isinstance(answer, str):
            context.metadata = {**(context.metadata or {}), "answer_written": answer}

        usage = summary.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            cache_tokens = usage.get("cache_read_tokens")
            if isinstance(input_tokens, int):
                context.n_input_tokens = input_tokens
            if isinstance(output_tokens, int):
                context.n_output_tokens = output_tokens
            if isinstance(cache_tokens, int):
                context.n_cache_tokens = cache_tokens
