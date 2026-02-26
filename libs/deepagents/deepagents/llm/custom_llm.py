"""
客制化 LLM API 适配器
支持任何兼容 OpenAI Chat Completions 格式的 API 端点。

典型场景：
  - DeepSeek:  base_url="https://api.deepseek.com/v1", model="deepseek-chat"
  - Qwen:      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-max"
  - 本地 vLLM: base_url="http://localhost:8000/v1", model="meta-llama/Llama-3-8B"
  - Ollama:    base_url="http://localhost:11434/v1", api_key="ollama", model="llama3"

环境变量（优先级低于显式参数）：
  CUSTOM_LLM_BASE_URL  — API 基础 URL
  CUSTOM_LLM_API_KEY   — API 密钥（无需鉴权时填任意字符串）
  CUSTOM_LLM_MODEL     — 模型名称
  CUSTOM_LLM_TEMP      — 温度（默认 0.0）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI


@dataclass
class CustomLLMConfig:
    """LLM 连接配置，支持从环境变量或显式参数构建。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 60


def create_custom_llm(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: int = 60,
) -> ChatOpenAI:
    """
    创建指向任意 OpenAI 兼容端点的 ChatOpenAI 实例。

    参数优先级：显式传参 > 环境变量 > 报错

    Args:
        base_url:    API 基础 URL，例如 "https://api.deepseek.com/v1"
        api_key:     API 密钥
        model:       模型名称
        temperature: 采样温度（0.0 = 确定性输出）
        max_tokens:  最大输出 token 数
        timeout:     请求超时（秒）

    Returns:
        配置好的 ChatOpenAI 实例（可直接传入 create_deep_agent）

    Raises:
        ValueError: 缺少必须参数时
    """
    resolved_base_url = base_url or os.environ.get("CUSTOM_LLM_BASE_URL")
    resolved_api_key = api_key or os.environ.get("CUSTOM_LLM_API_KEY", "")
    resolved_model = model or os.environ.get("CUSTOM_LLM_MODEL")
    resolved_temp = float(os.environ.get("CUSTOM_LLM_TEMP", temperature))

    if not resolved_base_url:
        raise ValueError(
            "base_url is required. Pass it explicitly or set CUSTOM_LLM_BASE_URL env var."
        )
    if not resolved_model:
        raise ValueError(
            "model is required. Pass it explicitly or set CUSTOM_LLM_MODEL env var."
        )

    return ChatOpenAI(
        base_url=resolved_base_url,
        api_key=resolved_api_key or "placeholder",  # OpenAI SDK 不允许空字符串
        model=resolved_model,
        temperature=resolved_temp,
        max_tokens=max_tokens,
        timeout=timeout,
    )
