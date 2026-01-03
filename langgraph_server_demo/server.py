"""Minimal LangGraph Server demo with deepagents.

This demonstrates how to expose your deepagents via HTTP API.
Run with: langgraph dev (for development) or langgraph up (for production)
"""

from deepagents.graph import create_deep_agent
from langgraph.checkpoint.sqlite import SqliteSaver


def create_agent():
    """Create and return the agent graph.

    LangGraph Server will call this function to get your graph.
    """
    # Use SQLite checkpointer for thread persistence
    # In production, you'd use PostgreSQL with AsyncPostgresSaver
    checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

    # Create your deepagent with all the middleware
    agent = create_deep_agent(
        # model defaults to Claude Sonnet 4.5
        # tools can be added here
        checkpointer=checkpointer,
        name="deepagent-server",
    )

    return agent


# This is what LangGraph Server looks for
graph = create_agent()
