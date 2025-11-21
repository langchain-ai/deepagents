import os
from os import getenv
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from langchain_core.callbacks.base import BaseCallbackHandler
from langsmith import Client


class OpenRouterPriceAndModelToLangSmith(BaseCallbackHandler):
    """On each LLM call, capture OpenRouter's actual model + usage cost,
    then tag + attach metadata to the LangSmith run."""

    def __init__(self) -> None:
        super().__init__()
        self.client = Client()

    def _extract_first_msg_metadata(self, response) -> Dict[str, Any]:
        """Get response_metadata from first generation if present."""
        try:
            if response.generations and response.generations[0]:
                msg = response.generations[0][0].message
                return getattr(msg, "response_metadata", {}) or {}
        except Exception:
            pass
        return {}

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs: Any) -> Any:
        meta = self._extract_first_msg_metadata(response)

        # OpenRouter usually surfaces the final resolved model as model_name (and/or model)
        model_used = meta.get("model_name") or meta.get("model")  # be liberal
        usage: Dict[str, Any] = meta.get("usage", {}) or {}

        # Costs (credits) are in usage.cost when usage accounting is enabled
        total_cost = usage.get("cost")
        upstream_cost = (usage.get("cost_details") or {}
                         ).get("upstream_inference_cost")

        # Build metadata payload for LangSmith
        ls_metadata = {
            "openrouter_model_used": model_used,
            "openrouter_usage": usage,  # full payload for later analysis
            "openrouter_total_cost": total_cost,
            "openrouter_upstream_inference_cost": upstream_cost,
        }

        # Prepare tags (keep them short)
        tags = []
        if model_used:
            tags.append(f"model:{model_used}")
        if total_cost is not None:
            tags.append(f"cost:{total_cost}")

        # Update the exact LLM run in LangSmith
        try:
            # You can also pass name=... or end_time=... if you need
            self.client.update_run(
                run_id=run_id, metadata=ls_metadata, tags=tags)
            pass
        except Exception:
            # Never break user code because of tracing issues
            pass


handler_openrouter_to_langsmith = OpenRouterPriceAndModelToLangSmith()

# Free
llm_openrouter_free = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="x-ai/grok-4-fast:free",
    temperature=0.1,
    model_kwargs={
        # "max_tokens": 4000,
    },
    default_headers={},
    extra_body={
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "high",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

# OpenRouter presets
# Works as langgraph agent
llm_openrouter_frontier = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="@preset/frontier",
    temperature=0.1,
    model_kwargs={
        # "max_tokens": 4000,
    },
    default_headers={},
    extra_body={
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "high",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

# Works as langgraph agent
llm_openrouter_frontier_affordable = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="@preset/frontier-affordable",
    temperature=0.1,
    default_headers={},
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "high",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

llm_openrouter_cheap_long_context = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="@preset/cheap-long-context",
    temperature=0.1,
    default_headers={},
    model_kwargs={
        # "max_tokens": 4000,
    },
)

# Works as langgraph agent
llm_openrouter_medium_fast = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="@preset/medium-gpt-oss-120b-high-throughput-w-fallback",
    temperature=0.1,
    default_headers={
    },
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "high",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

llm_openrouter_medium_fast_min_reasoning = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="@preset/medium-gpt-oss-120b-high-throughput-w-fallback",
    temperature=0.1,
    default_headers={
    },
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "low",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

llm_openrouter_flash = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="google/gemini-2.5-flash-preview-09-2025",
    temperature=0.1,
    timeout=15,
    max_retries=2,
    default_headers={
    },
    model_kwargs={
        # "max_tokens": 4000,
    },
    # extra_body={
    #     "reasoning": {
    #     # One of the following (not both):
    #     "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)
    #     # "max_tokens": 2000, # Specific token limit (Anthropic-style)
    #     # Optional: Default is false. All models support this.
    #     "exclude": False, # Set to true to exclude reasoning tokens from response
    #     # Or enable reasoning with the default parameters:
    #     # "enabled": True # Default: inferred from `effort` or `max_tokens`
    #     }
    # }
)

llm_openrouter_flash_reasoning = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="google/gemini-2.5-flash-preview-09-2025",
    temperature=0.1,
    default_headers={
    },
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            "enabled": True,
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "medium",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            # "exclude": False, # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

llm_openrouter_flash_lite = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="google/gemini-2.5-flash-lite-preview-09-2025",
    temperature=0.2,
    default_headers={
    },
    timeout=15,
    max_retries=2,
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "low",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

llm_openrouter_flash_lite_reasoning = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="google/gemini-2.5-flash-lite-preview-09-2025",
    temperature=0.1,
    default_headers={
    },
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            "enabled": True,
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "high",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            # "exclude": False, # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

llm_openrouter_gemini_pro = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="google/gemini-3-pro-preview",
    temperature=0.1,
    default_headers={
    },
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            "enabled": True,
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "low",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            # "exclude": False, # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    }
)

llm_openrouter_small_fast = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="@preset/small-fast-gpt-oss-20-b-high-throughput-w-fallback",
    temperature=0.1,
    default_headers={},
    model_kwargs={
        # "max_tokens": 4000,
    },
)

llm_openrouter_small_fast_min_reasoning = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="@preset/small-fast-gpt-oss-20-b-high-throughput-w-fallback",
    temperature=0.1,
    default_headers={},
    model_kwargs={
        # "max_tokens": 4000,
    },
    extra_body={
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style)
            "effort": "low",
            # "max_tokens": 2000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            # "enabled": True # Default: inferred from `effort` or `max_tokens`
        }
    },
)

llm_openrouter_test_model = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="openai/gpt-oss-20b",
    temperature=0.1,
    default_headers={},
    stream_usage=True,
    callbacks=[handler_openrouter_to_langsmith],
    # streaming=True,
    extra_body={
        "usage": {"include": True},
        "reasoning": {
            # One of the following (not both):
            # Can be "high", "medium", or "low" (OpenAI-style). Will be depend on model max reason tokens.
            "effort": "high",
            # "max_tokens": 10000, # Specific token limit (Anthropic-style)
            # Optional: Default is false. All models support this.
            "exclude": False,  # Set to true to exclude reasoning tokens from response
            # Or enable reasoning with the default parameters:
            "enabled": True  # Default: inferred from `effort` or `max_tokens`
        }
    }
)

# Google Gemini

model_fast_gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    thinking_budget=0,
    # convert_system_message_to_human=True, # Required for streaming?
    api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY", ""),
    # base_url=os.getenv("GOOGLE_AI_STUDIO_BASE_URL", ""),
)

model_fast_gemini_flash_thinking = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    thinking_budget=-1,
    # convert_system_message_to_human=True, # Required for streaming?
    api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY", ""),
    # base_url=os.getenv("GOOGLE_AI_STUDIO_BASE_URL", ""),
)

model_fast_gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    thinking_budget=-1,
    temperature=0.1,
    api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY", ""),
    # base_url=os.getenv("GOOGLE_AI_STUDIO_BASE_URL", ""),
)

# TODO add Gemini Flash 2.5 model with Google Search grounding
# from langchain_google_vertexai import ChatVertexAI
# from google.ai.generativelanguage_v1beta.types import Tool

# model_fast_gemini_flash_thinking_search_grounding = ChatVertexAI(
#     model="gemini-2.5-flash",        # Vertex model id  (GA as of 18 Jun 2025)
#     temperature=0.1,
#     tools=[Tool(google_search={})],  # identical to Gemini API
# )
"""
api-1    |   File "/usr/local/lib/python3.11/site-packages/google/cloud/aiplatform/initializer.py", line 368, in project
api-1    |     raise GoogleAuthError(project_not_found_exception_str) from exc
api-1    | google.auth.exceptions.GoogleAuthError: Unable to find your project. Please provide a project ID by:
api-1    | - Passing a constructor argument
api-1    | - Using vertexai.init()
api-1    | - Setting project using 'gcloud config set project my-project'
api-1    | - Setting a GCP environment variable
api-1    | - To create a Google Cloud project, please follow guidance at https://developers.google.com/workspace/guides/create-project
"""


# OpenAI/ Azure

# model_fast = AzureChatOpenAI(
#     # max_tokens=4000,
#     max_retries=3,
#     timeout=30,
#     azure_deployment=os.getenv(
#         "AZURE_CHAT_DEPLOYMENT_NAME_GPT_41_mini", "gpt-4.1-mini"),
#     openai_api_version=os.getenv(
#         "AZURE_OPENAI_API_VERSION_2", "2024-02-15-preview"),
#     temperature=0.1,
#     api_key=os.getenv("AZURE_OPENAI_API_KEY_2", None),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2", None),
#     streaming=True,
#     # cache=llm_cache,
# )


# model_fast_long = AzureChatOpenAI(
#     max_tokens=None,
#     max_retries=3,
#     timeout=30,
#     azure_deployment=os.getenv(
#         "AZURE_CHAT_DEPLOYMENT_NAME_GPT_41_mini", "gpt-4.1-mini"),
#     openai_api_version=os.getenv(
#         "AZURE_OPENAI_API_VERSION_2", "2024-02-15-preview"),
#     temperature=1,
#     api_key=os.getenv("AZURE_OPENAI_API_KEY_2", None),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2", None),
#     streaming=True,
# )

# model_gpt_41_nano = AzureChatOpenAI(
#     max_tokens=None,
#     max_retries=3,
#     timeout=30,
#     azure_deployment=os.getenv(
#         "AZURE_CHAT_DEPLOYMENT_NAME_GPT_41_nano", "gpt-4.1-nano"),
#     openai_api_version=os.getenv(
#         "AZURE_OPENAI_API_VERSION_2", "2024-02-15-preview"),
#     temperature=1,
#     api_key=os.getenv("AZURE_OPENAI_API_KEY_2", None),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2", None),
#     streaming=True,
# )

# model_smart = AzureChatOpenAI(
#     # max_tokens=2000,
#     max_retries=3,
#     timeout=30,
#     azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME_GPT_41", "gpt-4.1"),
#     openai_api_version=os.getenv(
#         "AZURE_OPENAI_API_VERSION_2", "2024-02-15-preview"),
#     temperature=1,
#     api_key=os.getenv("AZURE_OPENAI_API_KEY_2", None),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2", None),
#     streaming=True,
# )

# model_smart_long = AzureChatOpenAI(
#     max_tokens=None,
#     max_retries=3,
#     timeout=30,
#     azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME_GPT_41", "gpt-4.1"),
#     openai_api_version=os.getenv(
#         "AZURE_OPENAI_API_VERSION_2", "2024-02-15-preview"),
#     temperature=1,
#     api_key=os.getenv("AZURE_OPENAI_API_KEY_2", None),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2", None),
#     streaming=True,
# )
