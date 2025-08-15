from langchain.chat_models import init_chat_model
from typing import Optional


def get_default_model(
    model: str = 'anthropic:claude-sonnet-4-20250514',
    max_tokens: int = 8192,
    temperature: Optional[float] = None,
):
    return init_chat_model(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

