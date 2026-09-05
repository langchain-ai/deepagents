"""Data crossing the host/sandbox boundary in both directions.

Host tool -> sandbox file -> Python in the sandbox -> sandbox file -> host tool.
"""

import json
import random

from deepagents import create_deep_agent
from deepagents.backends.langsmith import LangSmithSandbox
from langchain.tools import tool
from langchain_quickjs import CodeInterpreterMiddleware
from langsmith.sandbox import SandboxClient

WORKDIR = "/root/workspace"

PTC_TOOLS = [
    "generate_readings",
    "render_report",
    "write_file",
    "read_file",
    "ls",
    "execute",
]

SYSTEM_PROMPT = f"""Use the js_eval tool to orchestrate work in one call.

Inside it, tools.generateReadings / tools.renderReport / tools.writeFile /
tools.readFile / tools.ls / tools.execute are async functions. Each takes a
single object argument, e.g. tools.writeFile({{file_path: p, content: s}}).

generateReadings and renderReport run on the host; the rest act on the sandbox.
Data only crosses that boundary through files: write host data into {WORKDIR}
for the sandbox to see, and read a sandbox file back before passing it to a
host tool.

Return values are the tools' own text output, not JSON:

- tools.readFile returns line-numbered text; strip the leading number from each
  line before parsing the content.
- tools.ls returns a Python-style list literal, e.g. "['a.json', 'b.json']".
"""

QUESTION = f"""In a single js_eval call:

1. Call generateReadings with count 200 to get sensor data from the host.
2. Write the raw JSON to {WORKDIR}/readings.json.
3. Write a Python script to {WORKDIR}/analyze.py that loads that file and
   writes {WORKDIR}/summary.json: a JSON list of objects with keys sensor,
   count, mean, min, max, values rounded to two decimals.
4. Run it with tools.execute.
5. Read summary.json back and pass its contents to renderReport, which runs on
   the host.

Return renderReport's output verbatim."""


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


@tool
def render_report(summary_json: str) -> str:
    """Render a per-sensor summary, as written by the sandbox, into a table."""
    rows = json.loads(summary_json)
    header = f"{'SENSOR':<8}{'COUNT':>7}{'MEAN':>9}{'MIN':>9}{'MAX':>9}"
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: -float(r["mean"])):
        lines.append(
            f"{row['sensor']:<8}{row['count']:>7}"
            f"{float(row['mean']):>9.2f}{float(row['min']):>9.2f}"
            f"{float(row['max']):>9.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    client = SandboxClient()
    sandbox = client.create_sandbox()

    try:
        backend = LangSmithSandbox(sandbox)
        print(f"sandbox: {backend.id}")
        backend.execute(f"mkdir -p {WORKDIR}")

        agent = create_deep_agent(
            model="openai:gpt-5.6-luna",
            tools=[generate_readings, render_report],
            backend=backend,
            middleware=[
                CodeInterpreterMiddleware(
                    tool_name="js_eval",
                    ptc=PTC_TOOLS,
                    # One eval makes several round trips to the sandbox; the
                    # 5s default covers local compute, not remote orchestration.
                    timeout=120,
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
        client.delete_sandbox(sandbox.name)


if __name__ == "__main__":
    main()
