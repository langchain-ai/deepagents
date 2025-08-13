from langchain.chat_models import init_chat_model
from typing import Optional


def get_model(
    model_name: str = 'claude-sonnet-4-20250514',
    model_provider: str = 'anthropic',
    max_tokens: int = 8192,
    temperature: Optional[float] = None,
):
    return init_chat_model(
        model=model_name,
        model_provider=model_provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )
