#!/usr/bin/env python3
"""
RLM Benchmark Runner for Deep Agents CLI

Runs the deepagents CLI against benchmarks from the RLM paper
(Zhang, Kraska, Khattab 2025) to test recursive decomposition
via subagents on long-context tasks.

Benchmarks:
  - s_niah:      Single Needle-in-a-Haystack (from RULER)
  - oolong:      Long-context aggregation (OOLONG trec_coarse)
  - browsecomp:  Multi-hop QA over documents (BrowseComp-Plus)

Usage:
    uv run python run_benchmark.py s_niah --num-tasks 10
    uv run python run_benchmark.py oolong --num-tasks 5
    uv run python run_benchmark.py browsecomp --num-tasks 5
    uv run python run_benchmark.py all --num-tasks 5
"""

import warnings

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
EXAMPLE_DIR = Path(__file__).parent


def parse_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip().split("\n")[-1].strip()


async def run_single_task(
    task_id: int,
    context: str,
    query: str,
    model_name: str | None,
    work_dir: Path,
) -> str:
    from deepagents_cli.agent import create_cli_agent
    from deepagents_cli.config import create_model
    from langgraph.checkpoint.memory import InMemorySaver

    task_dir = work_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    context_file = task_dir / "context.txt"
    context_file.write_text(context)

    output_file = task_dir / "answer.txt"
    output_file.write_text("")

    model_result = create_model(model_name)
    agent, backend = create_cli_agent(
        model=model_result.model,
        assistant_id=f"rlm-bench-{task_id}",
        tools=[],
        auto_approve=True,
        checkpointer=InMemorySaver(),
    )

    prompt = f"""## Task

{query}

## Context

The context for this task is stored in the file: {context_file}
It has {len(context.splitlines())} lines and {len(context)} characters.

## Instructions

1. Read the context file to understand its structure (use offset/limit to peek at sections).
2. Use subagents (the `task` tool) to process chunks of the context in parallel.
3. Combine the results and write your final answer to: {output_file}

Write your answer in the format: Answer: <your answer>
"""

    thread_id = f"rlm-bench-{task_id}"
    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(
        {"messages": [("human", prompt)]},
        config=config,
    )

    if output_file.exists():
        answer = output_file.read_text()
        if answer.strip():
            return answer

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")

    return ""


async def run_benchmark(
    benchmark: str,
    num_tasks: int,
    model_name: str | None,
    num_lines: int,
    num_documents: int,
    seed: int,
    results_dir: Path,
):
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]  Running: {benchmark.upper()}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    if benchmark == "s_niah":
        from benchmarks.s_niah import generate_dataset, evaluate

        tasks = generate_dataset(num_tasks=num_tasks, num_lines=num_lines, seed=seed)
        task_data = [
            {"context": t.context, "query": t.query, "answer": t.answer}
            for t in tasks
        ]
    elif benchmark == "oolong":
        from benchmarks.oolong import generate_synthetic_dataset, evaluate

        tasks = generate_synthetic_dataset(num_tasks=num_tasks, seed=seed)
        task_data = [
            {"context": t.context, "query": t.query, "answer": t.answer}
            for t in tasks
        ]
    elif benchmark == "browsecomp":
        from benchmarks.browsecomp import generate_synthetic_dataset, evaluate

        tasks = generate_synthetic_dataset(
            num_tasks=num_tasks, num_documents=num_documents, seed=seed
        )
        task_data = [
            {
                "context": "\n\n".join(t.documents),
                "query": t.query,
                "answer": t.answer,
            }
            for t in tasks
        ]
    else:
        console.print(f"[red]Unknown benchmark: {benchmark}[/red]")
        return {}

    results = []
    work_dir = Path(tempfile.mkdtemp(prefix=f"rlm-bench-{benchmark}-"))
    console.print(f"[dim]Work directory: {work_dir}[/dim]")
    console.print(f"[dim]Running {len(task_data)} tasks...[/dim]\n")

    for i, task in enumerate(task_data):
        console.print(f"\n[bold]--- Task {i+1}/{len(task_data)} ---[/bold]")
        console.print(f"[dim]Expected answer: {task['answer']}[/dim]")

        start_time = time.time()
        try:
            raw_output = await run_single_task(
                task_id=i,
                context=task["context"],
                query=task["query"],
                model_name=model_name,
                work_dir=work_dir,
            )
            elapsed = time.time() - start_time

            predicted = parse_answer(raw_output)
            score = evaluate(predicted, task["answer"])

            results.append({
                "task_id": i,
                "predicted": predicted,
                "expected": task["answer"],
                "score": score,
                "elapsed_seconds": round(elapsed, 1),
                "raw_output": raw_output[:500],
            })

            status = "[green]✓[/green]" if score >= 0.5 else "[red]✗[/red]"
            console.print(
                f"  {status} Predicted: {predicted[:80]} | "
                f"Score: {score:.2f} | Time: {elapsed:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            console.print(f"  [red]Error: {e}[/red]")
            results.append({
                "task_id": i,
                "predicted": "",
                "expected": task["answer"],
                "score": 0.0,
                "elapsed_seconds": round(elapsed, 1),
                "error": str(e),
            })

    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0
    avg_time = (
        sum(r["elapsed_seconds"] for r in results) / len(results) if results else 0
    )

    console.print(f"\n[bold]Results for {benchmark.upper()}:[/bold]")
    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Tasks", str(len(results)))
    table.add_row("Avg Score", f"{avg_score:.3f}")
    table.add_row("Total Correct (≥0.5)", f"{sum(1 for r in results if r['score'] >= 0.5)}/{len(results)}")
    table.add_row("Avg Time (s)", f"{avg_time:.1f}")
    console.print(table)

    results_file = results_dir / f"{benchmark}_results.json"
    results_file.write_text(json.dumps({
        "benchmark": benchmark,
        "model": model_name or "default",
        "num_tasks": len(results),
        "avg_score": avg_score,
        "results": results,
    }, indent=2))
    console.print(f"[dim]Results saved to {results_file}[/dim]")

    return {"benchmark": benchmark, "avg_score": avg_score, "num_tasks": len(results)}


def main():
    parser = argparse.ArgumentParser(
        description="Run RLM benchmarks with deepagents CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python run_benchmark.py s_niah --num-tasks 10
  uv run python run_benchmark.py oolong --num-tasks 5
  uv run python run_benchmark.py browsecomp --num-tasks 5 --num-documents 50
  uv run python run_benchmark.py all --num-tasks 3
  uv run python run_benchmark.py s_niah --model openai:gpt-4o
""",
    )
    parser.add_argument(
        "benchmark",
        choices=["s_niah", "oolong", "browsecomp", "all"],
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=10, help="Number of tasks to run (default: 10)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model to use (e.g., anthropic:claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--num-lines",
        type=int,
        default=500,
        help="S-NIAH: number of haystack lines (default: 500)",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=50,
        help="BrowseComp: number of documents per task (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    results_dir = EXAMPLE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    benchmarks = (
        ["s_niah", "oolong", "browsecomp"] if args.benchmark == "all" else [args.benchmark]
    )

    async def run_all():
        summaries = []
        for bench in benchmarks:
            summary = await run_benchmark(
                benchmark=bench,
                num_tasks=args.num_tasks,
                model_name=args.model,
                num_lines=args.num_lines,
                num_documents=args.num_documents,
                seed=args.seed,
                results_dir=results_dir,
            )
            summaries.append(summary)

        if len(summaries) > 1:
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print("[bold cyan]  Overall Summary[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
            table = Table()
            table.add_column("Benchmark", style="bold")
            table.add_column("Tasks")
            table.add_column("Avg Score")
            for s in summaries:
                table.add_row(
                    s["benchmark"],
                    str(s["num_tasks"]),
                    f"{s['avg_score']:.3f}",
                )
            console.print(table)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
