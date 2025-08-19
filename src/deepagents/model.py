from langchain_anthropic import ChatAnthropic


def get_default_model():
    return ChatAnthropic(model_name="claude-sonnet-4-20250514", max_tokens=64000)
