"""A Deep Agent whose slow tools run in the background on the Agent Server.

Each `@background_tool` is compiled into its own functional-API graph and
registered in `langgraph.json` next to the agent. When the agent calls one, the
call starts a run of that graph and returns a task ID right away, so the agent
can keep talking to the user while the work happens.
"""

import time

from langchain.chat_models import init_chat_model
from langgraph.func import task

from deepagents import background_tool, create_deep_agent

MODEL = "anthropic:claude-sonnet-4-6"


@task
def investigate(topic: str, angle: str) -> str:
    """Write notes on one angle of a topic."""
    model = init_chat_model(MODEL)
    prompt = f"In five bullet points, cover the {angle} of: {topic}"
    return f"## {angle.title()}\n{model.invoke(prompt).text}"


@background_tool
def research_report(topic: str) -> str:
    """Research a topic from several angles in parallel and return a written report.

    Takes a minute or more. Use for any request that needs a thorough report.
    """
    angles = ["history", "current state", "open problems"]
    futures = [investigate(topic, angle) for angle in angles]
    return f"# {topic}\n\n" + "\n\n".join(future.result() for future in futures)


@task
def benchmark_step(step: int) -> float:
    """Run one step of the benchmark and return how long it took."""
    started = time.perf_counter()
    time.sleep(2)
    return time.perf_counter() - started


@background_tool
def run_benchmark(steps: int = 10) -> dict[str, float]:
    """Run the performance benchmark and return timing statistics.

    Each step takes about two seconds.
    """
    timings = [benchmark_step(step).result() for step in range(steps)]
    return {"steps": steps, "total_seconds": round(sum(timings), 2), "mean_seconds": round(sum(timings) / steps, 2)}


agent = create_deep_agent(
    model=MODEL,
    tools=[research_report, run_benchmark],
    system_prompt=(
        "You help users with research and benchmarking. Slow tools run in the background: "
        "start them, tell the user what is running, and keep helping with other requests. "
        "When a task finishes, share its result."
    ),
)
