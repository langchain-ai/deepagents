"""NVIDIA Deep Agent Skills.

General-purpose deep agent showcasing multi-model architecture:
- Frontier model as orchestrator and data processor
- NVIDIA Nemotron Super for research
- NVIDIA GPU skills (cuDF analytics, cuML ML, data visualization, document processing)
- Modal GPU sandbox for code execution with uploaded memory and skills

Inspired by NVIDIA's AIQ Blueprint. For the full blueprint with
NeMo Agent Toolkit, evaluation harnesses, knowledge layer, and frontend,
see: https://github.com/langchain-ai/aiq-blueprint
"""

import os
from datetime import datetime
from typing import Literal

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing_extensions import TypedDict

from src.backend import create_backend
from src.prompts import (
    DATA_PROCESSOR_INSTRUCTIONS,
    ORCHESTRATOR_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
)
from src.tools import tavily_search


class Context(TypedDict, total=False):
    """Runtime context passed via `context=` at invoke time.

    Controls sandbox configuration per-run. Defaults to GPU mode.
    """

    sandbox_type: Literal["gpu", "cpu"]

# Current date for prompt formatting
current_date = datetime.now().strftime("%Y-%m-%d")

# --- Models ---

# frontier model: Uses init_chat_model for model-agnostic configuration.
# Format: "provider:model_name" (e.g., "anthropic:claude-sonnet-4-6")
frontier_model = init_chat_model(
    os.environ.get("ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-4-6")
)

# Subagents: NVIDIA Nemotron Super via NIM
# Fast, efficient OSS model for research, data analysis, and optimization tasks.
nemotron_super = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b"
)

# --- Tools ---
tools = [tavily_search]

# --- Subagents ---

researcher_sub_agent = {
    "name": "researcher-agent",
    "description": (
        "Delegate research to this agent. Conducts web searches and gathers "
        "information on a topic. Give one focused research topic at a time."
    ),
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": tools,
    "model": nemotron_super,
}

data_processor_sub_agent = {
    "name": "data-processor-agent",
    "description": (
        "Delegate data analysis, ML, visualization, and document processing tasks. "
        "Handles large datasets (CSV analysis, statistical profiling, anomaly detection), "
        "ML model training (classification, regression, clustering), chart creation, "
        "and bulk document extraction using GPU-accelerated NVIDIA tools."
    ),
    "system_prompt": DATA_PROCESSOR_INSTRUCTIONS.format(date=current_date),
    "tools": tools,
    "model": frontier_model,
    "skills": ["/skills/"]
    # "interrupt_on": {"execute": True} # enable human in the loop for code execution
}

# --- Create Agent ---

agent = create_deep_agent(
    model=frontier_model,
    tools=tools,
    system_prompt=ORCHESTRATOR_INSTRUCTIONS.format(date=current_date),
    subagents=[researcher_sub_agent, data_processor_sub_agent],
    memory=["/memory/AGENTS.md"],
    backend=create_backend,
    context_schema=Context
    # interrupt_on={"execute": True}, # enable human in the loop for code execution
)
