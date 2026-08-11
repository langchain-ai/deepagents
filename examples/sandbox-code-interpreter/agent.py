"""Host tool output -> sandbox file -> Python analysis inside the sandbox."""

import json
import random

from deepagents import create_deep_agent
from deepagents.backends.langsmith import LangSmithSandbox
from langchain.tools import tool
from langchain_quickjs import CodeInterpreterMiddleware
from langsmith.sandbox import SandboxClient

WORKDIR = "/root/workspace"

PTC_TOOLS = ["generate_readings", "write_file", "read_file", "ls", "execute"]

SYSTEM_PROMPT = f"""Use the js_eval tool to orchestrate work in one call.

Inside it, tools.generateReadings / tools.writeFile / tools.readFile / tools.ls
/ tools.execute are async functions. Each takes a single object argument, e.g.
tools.writeFile({{file_path: p, content: s}}).

generateReadings runs on the host, so the data it returns only reaches the
sandbox if you write it to a file there. Everything under {WORKDIR} is visible
to processes you start with tools.execute.

Return values are the tools' own text output, not JSON:

- tools.readFile returns line-numbered text; strip the leading number from each
  line before parsing the content.
- tools.ls returns a Python-style list literal, e.g. "['a.json', 'b.json']".
"""

QUESTION = f"""In a single js_eval call:

1. Call generateReadings with count 200 to get sensor data from the host.
2. Write the raw JSON to {WORKDIR}/readings.json.
3. Write a Python script to {WORKDIR}/analyze.py that loads that file and
   prints, per sensor, the reading count and the mean, min, and max value
   rounded to two decimals.
4. Run it with tools.execute and return its output.

Then report the per-sensor table."""


@tool
def generate_readings(count: int) -> str:
    """Generate synthetic sensor readings as a JSON list of objects."""
    rng = random.Random(7)
    sensors = ["kiln", "press", "dryer"]
    readings = [
        {
            "sensor": rng.choice(sensors),
            "value": round(rng.gauss(70, 12), 3),
            "ok": rng.random() > 0.05,
        }
        for _ in range(count)
    ]
    return json.dumps(readings)


def main() -> None:
    client = SandboxClient()
    raw_sandbox = client.create_sandbox(timeout=180, wait_for_ready=True)

    try:
        sandbox = LangSmithSandbox(raw_sandbox)
        print(f"sandbox: {sandbox.id}")
        sandbox.execute(f"mkdir -p {WORKDIR}")

        agent = create_deep_agent(
            model="openai:gpt-5.6-luna",
            tools=[generate_readings],
            backend=sandbox,
            middleware=[
                CodeInterpreterMiddleware(
                    tool_name="js_eval",
                    ptc=PTC_TOOLS,
                )
            ],
            system_prompt=SYSTEM_PROMPT,
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": QUESTION}]},
            {"configurable": {"thread_id": "sandbox-code-interpreter"}},
        )

        for message in result["messages"]:
            message.pretty_print()
    finally:
        client.delete_sandbox(raw_sandbox.name)


if __name__ == "__main__":
    main()
