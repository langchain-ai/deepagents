"""Ralph Mode agent graph for LangGraph / LangSmith Agent Server.

Each invoke is one Ralph iteration: fresh thread context, persistent filesystem
memory. Loop by calling the deployment repeatedly with a new `thread_id` and the
same task prompt (see README).
"""

from __future__ import annotations

import os

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

RALPH_INSTRUCTIONS = """\
You are running in Ralph mode: one focused iteration of a long-running build.

Rules:
- Inspect the filesystem first. Previous work is your memory.
- Make concrete progress on the TASK in the user message.
- Prefer writing/updating files over long explanations.
- Use git when the task asks for it (status, commits with clear messages).
- Do not ask clarifying questions; make reasonable assumptions and keep going.
- End with a short summary of what changed this iteration.
"""

model = init_chat_model(
    os.environ.get("RALPH_MODEL", "openai:gpt-5.5"),
)

agent = create_deep_agent(
    model=model,
    system_prompt=RALPH_INSTRUCTIONS,
)
