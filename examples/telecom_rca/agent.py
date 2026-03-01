"""Telecom RCA Agent — LangGraph deployment entry point.

This module creates a deep research agent specialized for Root Cause Analysis (RCA)
of telecom network events (alarms, KPI degradations, outages).
"""

from datetime import datetime

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from rca_agent.prompts import (
    RCA_DELEGATION_INSTRUCTIONS,
    RCA_RESEARCHER_INSTRUCTIONS,
    RCA_WORKFLOW_INSTRUCTIONS,
)
from rca_agent.tools import classify_event, tavily_search, think_tool

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 2

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Combine orchestrator instructions
INSTRUCTIONS = (
    RCA_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + RCA_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# Create RCA research sub-agent
rca_researcher = {
    "name": "rca-researcher",
    "description": (
        "Research sub-agent for finding known root causes of a specific telecom event. "
        "Give it one specific event/symptom or domain at a time. "
        "It searches vendor docs, 3GPP specs, and operator communities."
    ),
    "system_prompt": RCA_RESEARCHER_INSTRUCTIONS,
    "tools": [tavily_search, think_tool],
}

# Model — gpt-4o-mini has 200k TPM (vs 30k for gpt-4o on Tier 1), required for large prompts
model = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)

# Create the agent
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool, classify_event],
    system_prompt=INSTRUCTIONS,
    subagents=[rca_researcher],
)
