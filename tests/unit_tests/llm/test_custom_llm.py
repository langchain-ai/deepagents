import os
import pytest
from unittest.mock import patch
from deepagents.llm.custom_llm import create_custom_llm, CustomLLMConfig


def test_create_from_explicit_args():
    llm = create_custom_llm(
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        model="qwen2.5-72b",
        temperature=0.0,
    )
    assert llm is not None


def test_create_from_env_vars():
    env = {
        "CUSTOM_LLM_BASE_URL": "http://my-api.com/v1",
        "CUSTOM_LLM_API_KEY": "env-key",
        "CUSTOM_LLM_MODEL": "deepseek-chat",
    }
    with patch.dict(os.environ, env):
        llm = create_custom_llm()
    assert llm is not None


def test_raises_when_no_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="base_url"):
            create_custom_llm(api_key="key", model="model")


def test_config_dataclass():
    cfg = CustomLLMConfig(
        base_url="http://x.com/v1",
        api_key="k",
        model="m",
    )
    assert cfg.base_url == "http://x.com/v1"
    assert cfg.temperature == 0.0  # 默认值
