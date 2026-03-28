"""Parallel Subagent Coordination — Cookbook Example.

Demonstrates how to use Deep Agents' subagent system to delegate
independent research tasks to multiple subagents running in parallel,
then synthesize their results into a unified report.
"""

from deepagents import SubAgent, create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


@tool
def think(thought: str) -> str:
    """Use this tool to pause and reflect on findings before continuing.

    Think about what you've learned so far, what gaps remain, and what
    to investigate next. This helps you reason strategically between steps.

    Args:
        thought: Your reflection or reasoning about the current task.
    """
    return f"Thought recorded: {thought}"


# --- Subagent definitions ---

market_researcher: SubAgent = {
    "name": "market-researcher",
    "description": (
        "Research market trends, competitive landscape, and industry analysis "
        "for a given topic. Returns a structured summary with key findings."
    ),
    "system_prompt": (
        "You are a market research analyst. When given a topic:\n"
        "1. Use the think tool to plan your research approach\n"
        "2. Analyze market size, growth trends, and key players\n"
        "3. Identify opportunities and threats\n"
        "4. Return a structured report with sections: "
        "Overview, Key Players, Trends, Opportunities, Risks\n\n"
        "Be concise but thorough. Cite specific data points when possible."
    ),
    "tools": [think],
}

technical_analyst: SubAgent = {
    "name": "technical-analyst",
    "description": (
        "Analyze technical aspects of a technology or product — architecture, "
        "capabilities, limitations, and implementation considerations."
    ),
    "system_prompt": (
        "You are a technical analyst. When given a technology topic:\n"
        "1. Use the think tool to break down the technical components\n"
        "2. Evaluate architecture, scalability, and performance\n"
        "3. Identify technical strengths and limitations\n"
        "4. Return a structured report with sections: "
        "Architecture, Capabilities, Limitations, Recommendations\n\n"
        "Be precise and technical. Include concrete examples."
    ),
    "tools": [think],
}

user_researcher: SubAgent = {
    "name": "user-researcher",
    "description": (
        "Research user adoption, sentiment, and use cases for a given product "
        "or technology. Focuses on real-world usage patterns."
    ),
    "system_prompt": (
        "You are a user research specialist. When given a topic:\n"
        "1. Use the think tool to identify key user segments\n"
        "2. Analyze adoption patterns and user sentiment\n"
        "3. Document common use cases and pain points\n"
        "4. Return a structured report with sections: "
        "User Segments, Adoption, Use Cases, Pain Points, Satisfaction\n\n"
        "Focus on real-world evidence and user perspectives."
    ),
    "tools": [think],
}


# --- Orchestrator setup ---

ORCHESTRATOR_PROMPT = """You are a research coordinator. When the user asks you to
analyze a topic, you MUST delegate to your three specialist subagents in parallel:

1. **market-researcher** — market trends and competitive landscape
2. **technical-analyst** — technical architecture and capabilities
3. **user-researcher** — user adoption and real-world usage

Launch all three subagents simultaneously using separate task tool calls in a
single response. Each subagent should receive the same core topic but with
instructions specific to their domain.

After all subagents return, synthesize their findings into a unified executive
summary with clear sections and actionable insights."""


def create_parallel_research_agent():
    """Create a deep agent with three parallel research subagents."""
    model = init_chat_model(
        model="anthropic:claude-sonnet-4-5-20250929",
        temperature=0.0,
    )

    return create_deep_agent(
        model=model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=[market_researcher, technical_analyst, user_researcher],
        tools=[think],
    )


# --- Main ---


def main():
    """Run the parallel research agent on a sample query."""
    agent = create_parallel_research_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze the current state of AI coding assistants "
                        "(Cursor, GitHub Copilot, Claude Code, etc.) — cover "
                        "market positioning, technical approaches, and user adoption."
                    ),
                }
            ]
        }
    )

    # Print the final synthesized response
    final_message = result["messages"][-1]
    print(final_message.content if hasattr(final_message, "content") else str(final_message))


if __name__ == "__main__":
    main()
