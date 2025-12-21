"""Configuration management for ChATLAS agents."""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class LLMConfig(BaseModel):
    """Configuration for LLM backend."""

    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use")
    model: str = Field(default="gpt-5-mini ", description="Model name/identifier")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    streaming: bool = Field(default=False, description="Enable streaming responses")


class MCPServerConfig(BaseModel):
    """Configuration for ChATLAS MCP server connection."""

    url: str = Field(
        default="https://chatlas-mcp.app.cern.ch/mcp",
        description="MCP server URL",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")


class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent."""

    name: str = Field(..., description="Sub-agent name")
    description: str = Field(default="", description="Sub-agent description")
    enabled: bool = Field(default=True, description="Whether the sub-agent is enabled")
    llm: Optional[LLMConfig] = Field(default=None, description="Override LLM config for sub-agent")
    tools: List[str] = Field(default_factory=list, description="Tools available to sub-agent")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional sub-agent config")


class AgentConfig(BaseModel):
    """Configuration for a ChATLAS agent."""

    name: str = Field(default="chatlas-agent", description="Agent name")
    description: str = Field(
        default="ChATLAS AI assistant agent",
        description="Agent description",
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    mcp: MCPServerConfig = Field(
        default_factory=MCPServerConfig,
        description="MCP server configuration",
    )
    sub_agents: List[SubAgentConfig] = Field(
        default_factory=list,
        description="Sub-agents configuration",
    )
    tools: List[str] = Field(default_factory=list, description="Tools available to the agent")
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CHATLAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore',  # Ignore unknown environment variables
    )

    # LLM settings
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-5-mini")
    llm_api_key: Optional[str] = Field(default=None, description="API key for LLM provider")
    llm_base_url: Optional[str] = Field(default=None, description="Custom base URL for LLM API")
    llm_temperature: float = Field(default=0.7)

    # MCP server settings
    mcp_url: str = Field(default="https://chatlas-mcp.app.cern.ch/mcp")
    mcp_timeout: int = Field(default=60)

    # Agent settings
    agent_name: str = Field(default="chatlas-agent")
    agent_verbose: bool = Field(default=False)
    agent_max_iterations: int = Field(default=10)

    def to_agent_config(self) -> AgentConfig:
        """Convert settings to agent configuration."""
        return AgentConfig(
            name=self.agent_name,
            llm=LLMConfig(
                provider=LLMProvider(self.llm_provider),
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                temperature=self.llm_temperature,
            ),
            mcp=MCPServerConfig(
                url=self.mcp_url,
                timeout=self.mcp_timeout,
            ),
            max_iterations=self.agent_max_iterations,
            verbose=self.agent_verbose,
        )


def load_config_from_yaml(path: str) -> AgentConfig:
    """Load agent configuration from YAML file."""
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)


def load_config_from_env() -> AgentConfig:
    """Load agent configuration from environment variables."""
    import logging
    settings = Settings()
    
    logger = logging.getLogger(__name__)
    logger.debug(f"Loaded LLM provider: {settings.llm_provider}")
    logger.debug(f"Loaded LLM model: {settings.llm_model}")
    logger.debug(f"Loaded LLM API key: {'*' * 8 if settings.llm_api_key else 'NOT SET'}")
    
    return settings.to_agent_config()
