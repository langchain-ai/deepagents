"""Minimal test to check if Gemini works with create_deep_agent."""
import os
from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

agent = create_deep_agent(
    model=model,
    backend=LocalShellBackend(root_dir=os.getcwd(), inherit_env=True),
)

print("Invoking agent...")
result = agent.invoke(
    {"messages": [("user", "Read the file agent.py using the read_file tool and tell me how many lines it has.")]},
)

for msg in result["messages"]:
    print(f"\n--- {type(msg).__name__} ---")
    print(f"content: {repr(msg.content)[:500]}")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"tool_calls: {msg.tool_calls}")
