"""Middleware for providing skills to an agent via BackendProtocol."""

import re
import warnings
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, TypedDict

import yaml
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime

from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol


class SkillMetadata(TypedDict):
    """Metadata for a skill."""

    name: str
    """Name of the skill."""

    description: str
    """Description of what the skill does."""

    path: str
    """Path to the SKILL.md file within the backend."""

    label: NotRequired[str | None]
    """Label for grouping in display (e.g., 'User Skills', 'Project Skills')."""


SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you know they exist (name + description above),
but you only read the full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches any skill's description
2. **Read the skill's full instructions**: The skill list above shows the exact path to use with read_file
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When you need specialized knowledge or structured workflows
- When a skill provides proven patterns for complex tasks

**Skills are Self-Documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it
- The skill list above shows the full path for each skill's SKILL.md file

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills above → See "web-research" skill with its full path
2. Read the skill using the path shown in the list
3. Follow the skill's research workflow (search → organize → synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills are tools to make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


def _strip_line_numbers(content: str) -> str:
    r"""Strip line number prefixes (e.g., '     1\t') from content."""
    return "\n".join(
        line.partition("\t")[2] if "\t" in line else line
        for line in content.split("\n")
    )


def _parse_skill_metadata(content: str) -> dict[str, str] | None:
    """Parse YAML frontmatter from SKILL.md content. Returns None if invalid."""
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return None

    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None

    if not isinstance(metadata, dict):
        return None
    if "name" not in metadata or "description" not in metadata:
        return None

    return {
        "name": str(metadata["name"]),
        "description": str(metadata["description"]),
    }


def _list_skills_from_backend(
    backend: BackendProtocol,
    skills_path: str,
) -> list[SkillMetadata]:
    """Scan directory for skills (subdirs with SKILL.md) and return metadata."""
    skills: list[SkillMetadata] = []
    normalized_path = skills_path.rstrip("/") or "/"

    try:
        items = backend.ls_info(normalized_path)
    except OSError:
        return []

    for item in items:
        if not item.get("is_dir"):
            continue

        item_path = item.get("path", "")
        if not item_path:
            continue

        base = item_path.rstrip("/") if item_path.startswith("/") else f"{normalized_path}/{item_path.rstrip('/')}"
        skill_md_path = f"{base}/SKILL.md"

        content = backend.read(skill_md_path)
        if content.startswith("Error:"):
            continue

        content = _strip_line_numbers(content)
        metadata = _parse_skill_metadata(content)
        if metadata is None:
            continue

        skills.append(
            SkillMetadata(
                name=metadata["name"],
                description=metadata["description"],
                path=skill_md_path,
            )
        )

    return skills


class _StateProxy:
    """Adapter to make request.state accessible as runtime.state for backends.

    In wrap_model_call, request.runtime is Runtime (no state attribute),
    but request.state contains the actual state. This proxy bridges that gap
    for backends which expect runtime.state or runtime.store.
    """

    def __init__(self, state: dict[str, Any], store: Any = None) -> None:
        self.state = state
        self.store = store


class SkillsMiddleware(AgentMiddleware):
    """Middleware for loading and exposing agent skills via BackendProtocol.

    Implements Anthropic's progressive disclosure pattern: injects skills metadata
    into system prompt, agent reads full SKILL.md only when relevant.

    Skills are loaded once on first model call and cached in the instance.

    Args:
        backend: Backend for skill storage. Must match FilesystemMiddleware's backend.
        skills_paths: Paths to scan for skills. Later paths override earlier ones.
        system_prompt_template: Custom prompt template (optional).

    Example:
        ```python
        backend = FilesystemBackend(root_dir="/")
        agent = create_deep_agent(
            backend=backend,
            middleware=[SkillsMiddleware(backend=backend, skills_paths=["/skills"])]
        )
        ```
    """

    def __init__(
        self,
        backend: BACKEND_TYPES,
        skills_paths: list[str | tuple[str, str]] | None = None,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize SkillsMiddleware.

        Args:
            backend: Backend for skill storage (required).
            skills_paths: Paths to scan for skills.
            system_prompt_template: Custom prompt template.
        """
        super().__init__()
        if backend is None:
            msg = (
                "backend is required for SkillsMiddleware.\n"
                "Use the same backend as FilesystemMiddleware.\n"
                "Example: SkillsMiddleware(backend=backend)"
            )
            raise TypeError(msg)
        self.backend = backend
        self._backend_checked = False
        self._cached_skills: list[SkillMetadata] | None = None
        if skills_paths is not None:
            self.skills_paths = self._normalize_paths(skills_paths)
        else:
            self.skills_paths = [("/skills", None)]
        self.system_prompt_template = system_prompt_template or SKILLS_SYSTEM_PROMPT

    @staticmethod
    def _normalize_paths(
        paths: list[str | tuple[str, str]]
    ) -> list[tuple[str, str | None]]:
        """Normalize paths to (path, label) tuples with validation."""
        result: list[tuple[str, str | None]] = []
        for item in paths:
            if isinstance(item, tuple):
                if len(item) != 2:  # noqa: PLR2004
                    msg = f"Path tuple must have 2 elements: {item}"
                    raise ValueError(msg)
                path, label = item[0], item[1] if item[1] and str(item[1]).strip() else None
            elif isinstance(item, str):
                path, label = item, None
            else:
                msg = f"Path must be str or (path, label) tuple: {type(item)}"
                raise TypeError(msg)

            if not path or not path.strip():
                msg = "Path cannot be empty"
                raise ValueError(msg)
            if not path.startswith("/"):
                msg = f"Path must be absolute: '{path}'"
                raise ValueError(msg)

            result.append((path, label))
        return result

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Resolve backend instance from direct reference or factory."""
        if isinstance(self.backend, BackendProtocol):
            return self.backend
        if callable(self.backend):
            return self.backend(runtime)
        msg = f"Invalid backend type: {type(self.backend)}"
        raise TypeError(msg)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills for system prompt."""
        if not skills:
            paths_str = ", ".join(f"`{p}`" for p, _ in self.skills_paths)
            return f"(No skills available yet. Skills can be added to {paths_str})"

        lines = []
        for skill in sorted(skills, key=lambda x: (x.get("label") or "", x["name"])):
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  → Read `{skill['path']}` for full instructions")
        return "\n".join(lines)

    def _check_backend_consistency(self, tools: list[Any]) -> None:
        """Warn if backends mismatch between SkillsMiddleware and FilesystemMiddleware."""
        fs_backend = None
        for tool in tools:
            if getattr(tool, "name", None) == "read_file":
                for cell in getattr(getattr(tool, "func", None), "__closure__", None) or []:
                    try:
                        content = cell.cell_contents
                        if isinstance(content, BackendProtocol) or (
                            hasattr(content, "read") and hasattr(content, "ls_info")
                        ):
                            fs_backend = content
                            break
                    except ValueError:
                        pass
                break

        if fs_backend is None or callable(self.backend) or callable(fs_backend):
            return

        if type(self.backend) is not type(fs_backend):
            warnings.warn(
                f"SkillsMiddleware uses {type(self.backend).__name__}, "
                f"but FilesystemMiddleware uses {type(fs_backend).__name__}. "
                "The agent may fail to read skill files.",
                UserWarning,
                stacklevel=3,
            )

    def _load_skills(self, runtime: ToolRuntime) -> list[SkillMetadata]:
        """Load and merge skills from all configured paths (later paths override)."""
        backend = self._get_backend(runtime)
        all_skills: dict[str, SkillMetadata] = {}

        for path, label in self.skills_paths:
            # Auto-generate label from path if not provided
            effective_label = label if label else path.rstrip("/").split("/")[-1].replace("-", " ").title()
            for skill in _list_skills_from_backend(backend, path):
                skill["label"] = effective_label
                all_skills[skill["name"]] = skill

        return list(all_skills.values())

    def _get_skills(self, runtime: ToolRuntime) -> list[SkillMetadata]:
        """Get skills with lazy loading and caching."""
        if self._cached_skills is None:
            self._cached_skills = self._load_skills(runtime)
        return self._cached_skills

    def _format_skills_locations(self) -> str:
        """Format skills paths for system prompt."""
        parts = [
            f"**{label or 'Skills'}**: `{path}`"
            for path, label in self.skills_paths
        ]
        return "\n".join(parts)

    def _prepare_skills_prompt(self, request: ModelRequest) -> str:
        """Build system prompt with skills documentation injected."""
        if not self._backend_checked:
            self._check_backend_consistency(request.tools)
            self._backend_checked = True

        # Use _StateProxy to bridge request.state/store → runtime for backends
        # In wrap_model_call, request.runtime is Runtime (no state), but request.state exists
        runtime_or_proxy = (
            _StateProxy(
                state=request.state,
                store=getattr(request.runtime, "store", None),
            )
            if hasattr(request, "state")
            else request.runtime
        )
        skills = self._get_skills(runtime_or_proxy)  # type: ignore[arg-type]
        skills_section = self.system_prompt_template.format(
            skills_locations=self._format_skills_locations(),
            skills_list=self._format_skills_list(skills),
        )

        if request.system_prompt:
            return request.system_prompt + "\n\n" + skills_section
        return skills_section

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject skills into system prompt before model call."""
        system_prompt = self._prepare_skills_prompt(request)
        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        system_prompt = self._prepare_skills_prompt(request)
        return await handler(request.override(system_prompt=system_prompt))
