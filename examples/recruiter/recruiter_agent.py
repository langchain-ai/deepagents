import os

from exa_py import Exa

from deepagents import create_deep_agent, SubAgent

exa = Exa(api_key = os.environ['EXA_API_KEY'])






# Search tool to use to do research
def linkedin_search(
    query: str,
    num_results: int = 5,
):
    """Run a linkedin search"""
    return exa.search_and_contents(
        query,
        text=True,
        num_results=num_results,
        type="auto",
        category="linkedin profile"
    )


sub_research_prompt = """You are a dedicated researcher. Your job is to source candidates for the role described.

Write down any new candidates in candidates.json

only include candidates if they look good

Respond to the user saying how many candidates you wrote down."""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to kick off in depth searches. Give it a brief to research.",
    "prompt": sub_research_prompt,
    "tools": ["linkedin_search"]
}

# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert sourcer

Use the research agent to run specific searches. It will write its results to candidates.json"""

# Create the agent
agent = create_deep_agent(
    [linkedin_search],
    research_instructions,
    subagents=[research_sub_agent],
).with_config({"recursion_limit": 1000})

from pydantic import BaseModel

class Config(BaseModel):
    instructions: str = research_instructions
    subagents: list[SubAgent] = [research_sub_agent]

from langchain_core.runnables import RunnableConfig

def create_agent(config: RunnableConfig):
    config = config.get('configurable', {})
    config_fields = {k: v for k,v in config.items() if k in ['instructions', 'subagents']}
    config = Config(**config_fields)
    return create_deep_agent(
        [linkedin_search],
        config.instructions,
        subagents=config.subagents,
        config_schema=Config,
    ).with_config({"recursion_limit": 1000})
