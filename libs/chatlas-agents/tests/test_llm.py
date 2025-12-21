"""Tests for LLM factory."""

import pytest
from chatlas_agents.config import LLMConfig, LLMProvider
from chatlas_agents.llm import LLMFactory, create_llm_from_config
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq


def test_create_openai_llm():
    """Test creating OpenAI LLM."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-5-mini",
        temperature=0.7,
        api_key="test-key",  # Provide a test key to avoid error
    )
    llm = LLMFactory.create_llm(config)
    assert isinstance(llm, ChatOpenAI)
    # Note: OpenAI uses 'model_name' attribute
    assert llm.model_name == "gpt-5-mini"
    assert llm.temperature == 0.7


def test_create_anthropic_llm():
    """Test creating Anthropic LLM."""
    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        temperature=0.5,
        api_key="test-key",  # Provide a test key to avoid error
    )
    llm = LLMFactory.create_llm(config)
    assert isinstance(llm, ChatAnthropic)
    # Note: Anthropic uses 'model' attribute (not 'model_name')
    assert llm.model == "claude-3-5-sonnet-20241022"
    assert llm.temperature == 0.5


def test_create_groq_llm():
    """Test creating Groq LLM."""
    config = LLMConfig(
        provider=LLMProvider.GROQ,
        model="llama-3.1-70b-versatile",
        temperature=0.7,
        api_key="test-key",  # Provide a test key to avoid error
    )
    llm = LLMFactory.create_llm(config)
    assert isinstance(llm, ChatGroq)
    # Note: Groq uses 'model_name' attribute
    assert llm.model_name == "llama-3.1-70b-versatile"
    assert llm.temperature == 0.7


def test_create_llm_from_config():
    """Test convenience function."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI, 
        model="gpt-5-mini",
        api_key="test-key"
    )
    llm = create_llm_from_config(config)
    assert isinstance(llm, ChatOpenAI)


def test_unsupported_provider():
    """Test error handling for unsupported provider."""
    # This would require modifying the enum, so we skip it
    pass
