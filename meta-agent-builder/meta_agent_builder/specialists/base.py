"""Base specialist class for all specialist agents."""

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from deepagents.middleware.subagents import SubAgent
from langchain_core.tools import BaseTool


class BaseSpecialist:
    """Base class for all specialist agents.

    This class provides common functionality for all specialist agents including:
    - Loading system prompts from files
    - Converting to SubAgent configuration
    - Standard initialization

    Attributes:
        name: Unique identifier for the specialist
        description: Description shown to the orchestrator
        system_prompt: Detailed instructions for the specialist
        tools: Tools available to the specialist
        model: Model to use (defaults to Claude Sonnet 4.5)
    """

    def __init__(
        self,
        name: str,
        description: str,
        prompt_file: str,
        tools: Optional[Sequence[BaseTool | Callable]] = None,
        model: str = "claude-sonnet-4-5-20250929",
        middleware: Optional[list[Any]] = None,
    ):
        """Initialize the specialist.

        Args:
            name: Specialist identifier (e.g., "documentation-specialist")
            description: What this specialist does (shown to orchestrator)
            prompt_file: Filename in prompts/specialists/ directory
            tools: List of tools for this specialist
            model: Model name or instance
            middleware: Optional additional middleware
        """
        self.name = name
        self.description = description
        self.system_prompt = self._load_prompt(prompt_file)
        self.tools = list(tools) if tools else []
        self.model = model
        self.middleware = middleware or []

    def _load_prompt(self, filename: str) -> str:
        """Load system prompt from file.

        Args:
            filename: Name of the prompt file

        Returns:
            Prompt content as string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Get path to prompts/specialists/ directory
        prompts_dir = Path(__file__).parent.parent / "prompts" / "specialists"
        prompt_path = prompts_dir / filename

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}\n"
                f"Expected location: prompts/specialists/{filename}"
            )

        return prompt_path.read_text()

    def to_subagent_config(self) -> SubAgent:
        """Convert to SubAgent configuration dictionary.

        Returns:
            SubAgent TypedDict configuration
        """
        config: SubAgent = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "model": self.model,
        }

        # Add middleware if present
        if self.middleware:
            config["middleware"] = self.middleware

        return config

    def add_tool(self, tool: BaseTool | Callable) -> None:
        """Add a tool to the specialist's toolset.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)

    def add_middleware(self, middleware: Any) -> None:
        """Add middleware to the specialist.

        Args:
            middleware: Middleware instance to add
        """
        self.middleware.append(middleware)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"tools={len(self.tools)}, "
            f"model='{self.model}'"
            f")"
        )
