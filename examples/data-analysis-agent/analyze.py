#!/usr/bin/env python3
"""Data Analysis Agent

A CSV data analysis agent configured through files on disk:
- AGENTS.md defines analysis standards and reporting conventions
- skills/ provides specialized workflows (exploratory, statistical, anomaly detection)
- LocalShellBackend enables running Python/pandas scripts via the execute tool
- ToolStrategy returns a structured DataReport as the final output

Usage:
    uv run python analyze.py data/sample_sales.csv
    uv run python analyze.py data/sample_sales.csv "Which region has the highest revenue?"
    uv run python analyze.py data/sample_sales.csv "Find anomalies in daily revenue"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

load_dotenv()

EXAMPLE_DIR = Path(__file__).parent
console = Console()


class Finding(BaseModel):
    """A single finding from the data analysis."""

    title: str
    detail: str
    severity: Literal["info", "warning", "critical"]
    chart_path: str | None = None


class DataReport(BaseModel):
    """Structured report produced by the data analysis agent."""

    dataset_name: str
    row_count: int
    column_count: int
    summary: str
    findings: list[Finding]
    recommendations: list[str]


SEVERITY_STYLES = {
    "info": ("INFO", "cyan"),
    "warning": ("WARN", "yellow"),
    "critical": ("CRIT", "red bold"),
}


def display_report(report: DataReport) -> None:
    """Render a DataReport to the terminal with Rich formatting."""
    console.print()

    # Dataset overview
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold")
    overview.add_column()
    overview.add_row("Dataset", report.dataset_name)
    overview.add_row("Shape", f"{report.row_count} rows x {report.column_count} columns")
    console.print(Panel(overview, title="Dataset Overview", border_style="blue"))

    # Summary
    console.print(Panel(report.summary, title="Summary", border_style="green"))

    # Findings
    if report.findings:
        console.print()
        console.print("[bold]Findings[/bold]")
        console.print()
        for finding in report.findings:
            label, style = SEVERITY_STYLES.get(finding.severity, ("INFO", "cyan"))
            header = Text()
            header.append(f"  {label}  ", style=style)
            header.append(finding.title, style="bold")
            console.print(header)
            for line in finding.detail.split("\n"):
                console.print(f"         {line}")
            if finding.chart_path:
                console.print(f"         [dim]Chart: {finding.chart_path}[/dim]")
            console.print()

    # Recommendations
    if report.recommendations:
        rec_text = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(report.recommendations))
        console.print(Panel(rec_text, title="Recommendations", border_style="magenta"))


class AgentDisplay:
    """Manages the display of agent progress during streaming."""

    def __init__(self) -> None:
        self.printed_count = 0
        self.spinner = Spinner("dots", text="Thinking...")

    def update_status(self, status: str) -> None:
        self.spinner = Spinner("dots", text=status)

    def _truncate(self, text: str, length: int = 80) -> str:
        return text[:length] + "..." if len(text) > length else text

    def print_message(self, msg: object) -> None:
        """Print a message with formatting based on its type."""
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="You", border_style="blue"))

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            if content and content.strip():
                console.print(Panel(Markdown(content), title="Agent", border_style="green"))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    self._print_tool_call(name, args)

        elif isinstance(msg, ToolMessage):
            self._print_tool_result(msg)

    def _print_tool_call(self, name: str, args: dict) -> None:
        if name == "execute":
            cmd = args.get("command", "")
            console.print(f"  [bold cyan]>> Execute:[/] {self._truncate(cmd)}")
            self.update_status("Running script...")
        elif name == "write_file":
            path = args.get("file_path", "file")
            console.print(f"  [bold yellow]>> Write:[/] {path}")
        elif name == "read_file":
            path = args.get("file_path", "file")
            console.print(f"  [bold blue]>> Read:[/] {path}")
        elif name == "write_todos":
            todos = args.get("todos", [])
            console.print(f"  [bold magenta]>> Planning:[/] {len(todos)} steps")
            self.update_status("Planning analysis...")
        elif name == "task":
            desc = args.get("description", "working...")
            console.print(f"  [bold magenta]>> Subagent:[/] {self._truncate(desc, 60)}")
            self.update_status(f"Subagent: {self._truncate(desc, 40)}")
        elif name == "ls":
            path = args.get("path", ".")
            console.print(f"  [bold blue]>> List:[/] {path}")
        elif name == "glob" or name == "grep":
            pattern = args.get("pattern", "")
            console.print(f"  [bold blue]>> {name.title()}:[/] {self._truncate(pattern)}")
        else:
            console.print(f"  [bold dim]>> {name}[/]")

    def _print_tool_result(self, msg: ToolMessage) -> None:
        name = getattr(msg, "name", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        is_error = "error" in content.lower()[:100] or "traceback" in content.lower()[:100]

        if name == "execute":
            if is_error:
                preview = self._truncate(content.strip().split("\n")[-1], 80)
                console.print(f"  [red]✗ Script failed:[/] {preview}")
            else:
                lines = content.strip().split("\n")
                preview = self._truncate(lines[0], 80) if lines else ""
                suffix = f" [dim](+{len(lines) - 1} lines)[/dim]" if len(lines) > 1 else ""
                console.print(f"  [green]✓ Result:[/] {preview}{suffix}")
        elif name == "write_file":
            console.print(f"  [green]✓ File written[/]")
        elif name == "write_todos":
            console.print(f"  [green]✓ Plan created[/]")
        elif name == "task":
            console.print(f"  [green]✓ Subagent complete[/]")
        elif is_error:
            console.print(f"  [red]✗ {name} failed[/]")
        else:
            console.print(f"  [green]✓ {name}[/]")


def create_analysis_agent(csv_path: str) -> object:
    """Create a data analysis agent configured by filesystem files.

    Args:
        csv_path: Path to the CSV file being analyzed, included in the system prompt
            so the agent knows which file to operate on.
    """
    return create_deep_agent(
        system_prompt=(
            f"You are analyzing the CSV file at: {csv_path}\n"
            "Use the `execute` tool to run Python scripts with pandas and matplotlib. "
            "Each script must be self-contained (import dependencies and load the CSV fresh). "
            "Always use `matplotlib.use('Agg')` before importing pyplot. "
            "Create the output/ directory before saving charts."
        ),
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        backend=LocalShellBackend(
            root_dir=EXAMPLE_DIR,
            virtual_mode=False,
            inherit_env=True,
        ),
        response_format=ToolStrategy(schema=DataReport),
    )


async def main() -> None:
    """Run the data analysis agent with streaming output."""
    parser = argparse.ArgumentParser(
        description="Analyze CSV files using an AI agent with pandas and matplotlib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python analyze.py data/sample_sales.csv
  uv run python analyze.py data/sample_sales.csv "Which region has the highest revenue?"
  uv run python analyze.py data/sample_sales.csv "Find anomalies in daily revenue"
        """,
    )
    parser.add_argument("csv_file", help="Path to the CSV file to analyze")
    parser.add_argument(
        "question",
        nargs="?",
        default="Summarize this dataset",
        help='Analysis question (default: "Summarize this dataset")',
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        console.print(f"[red bold]Error:[/red bold] File not found: {csv_path}")
        sys.exit(1)

    console.print()
    console.print("[bold blue]Data Analysis Agent[/bold blue]")
    console.print(f"[dim]File: {csv_path}[/dim]")
    console.print(f"[dim]Question: {args.question}[/dim]")
    console.print()

    resolved_csv = os.path.relpath(csv_path.resolve(), EXAMPLE_DIR)
    agent = create_analysis_agent(resolved_csv)
    display = AgentDisplay()

    last_chunk = None

    try:
        with Live(display.spinner, console=console, refresh_per_second=10, transient=True) as live:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": args.question}]},
                config={"configurable": {"thread_id": "data-analysis-demo"}},
                stream_mode="values",
            ):
                last_chunk = chunk
                if "messages" in chunk:
                    messages = chunk["messages"]
                    if len(messages) > display.printed_count:
                        live.stop()
                        for msg in messages[display.printed_count :]:
                            display.print_message(msg)
                        display.printed_count = len(messages)
                        live.start()
                        live.update(display.spinner)

        report = last_chunk.get("structured_response") if last_chunk else None

        if isinstance(report, DataReport):
            display_report(report)

    except Exception as e:
        console.print(Panel(f"[red bold]Error:[/red bold]\n\n{e}", border_style="red"))
        sys.exit(1)

    console.print()
    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
