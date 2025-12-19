"""LLM backend factory for different providers."""

import logging
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from chatlas_agents.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances based on configuration."""

    @staticmethod
    def create_llm(config: LLMConfig) -> BaseChatModel:
        """Create an LLM instance based on the configuration.

        Args:
            config: LLM configuration

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If the provider is not supported or API key is missing
        """
        # Validate that API key is provided
        if not config.api_key:
            raise ValueError(
                f"API key is required for {config.provider.value} provider. "
                f"Please set CHATLAS_LLM_API_KEY environment variable."
            )
        
        common_kwargs = {
            "temperature": config.temperature,
            "streaming": config.streaming,
        }

        if config.max_tokens is not None:
            common_kwargs["max_tokens"] = config.max_tokens

        if config.provider == LLMProvider.OPENAI:
            logger.info(f"Creating OpenAI LLM with model: {config.model}")
            return ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                **common_kwargs,
            )
        elif config.provider == LLMProvider.ANTHROPIC:
            logger.info(f"Creating Anthropic LLM with model: {config.model}")
            return ChatAnthropic(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                **common_kwargs,
            )
        elif config.provider == LLMProvider.GROQ:
            logger.info(f"Creating Groq LLM with model: {config.model}")
            return ChatGroq(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")



def create_llm_from_config(config: LLMConfig) -> BaseChatModel:
    """Convenience function to create an LLM from configuration.

    Args:
        config: LLM configuration

    Returns:
        Configured LLM instance
    """
    return LLMFactory.create_llm(config)
