#!/usr/bin/env python3
"""
MuAPI Creative Agent

A generative-media Deep Agent configured through filesystem primitives:
  - AGENTS.md          — persistent context: capabilities, decision tree, quality rules
  - skills/            — loaded-on-demand workflows (generate-asset, run-skill)
  - subagents.yaml     — creative-specialist subagent for heavy multi-step work
  - MuapiCostCallback  — real-time credit tracking with optional budget cap

Usage:
    export MUAPI_API_KEY="..."             # or: muapi auth configure
    export ANTHROPIC_API_KEY="..."
    uv run python creative_agent.py "Generate a cinematic product photo of a sneaker"
    uv run python creative_agent.py "Make a 3-shot Instagram carousel for SunFizz mango water"
    uv run python creative_agent.py --budget 200 "Animate my logo"
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from deepagents import create_deep_agent
from muapi_langchain import (
    PLANNER_TOOLS,
    SPECIALIST_TOOLS,
    MuapiCostCallback,
)

EXAMPLE_DIR = Path(__file__).parent
console = Console()


def load_subagents(config_path: Path) -> list[dict]:
    """Load subagent definitions from YAML and wire up muapi specialist tools."""
    tool_map = {t.name: t for t in SPECIALIST_TOOLS}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    subagents = []
    for name, spec in config.items():
        subagent: dict[str, Any] = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }
        if "model" in spec:
            subagent["model"] = spec["model"]
        if "tools" in spec:
            subagent["tools"] = [tool_map[t] for t in spec["tools"] if t in tool_map]
        subagents.append(subagent)
    return subagents


def create_creative_agent(budget_credits: int = 500):
    cost_cb = MuapiCostCallback(
        budget_credits=budget_credits,
        on_event=lambda evt, payload: console.print(
            f"  [dim][{evt}] credits: {payload.get('credits', 0)} "
            f"(total: {payload.get('running_total', 0)})[/]"
        ),
    )
    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        memory=[str(EXAMPLE_DIR / "AGENTS.md")],
        skills=[str(EXAMPLE_DIR / "skills/")],
        tools=PLANNER_TOOLS,
        subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml"),
        interrupt_on={
            "muapi_creative_agent": {"allowed_decisions": ["approve", "edit", "reject"]},
        },
    )
    return agent, cost_cb


class AgentDisplay:
    def __init__(self):
        self.printed_count = 0
        self._spinner = Spinner("dots", text="Thinking…")

    def spinner(self, text: str = "Thinking…") -> Spinner:
        self._spinner = Spinner("dots", text=text)
        return self._spinner

    def render(self, msg: Any) -> None:
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="You", border_style="blue"))

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if content and content.strip():
                console.print(Panel(Markdown(content), title="Agent", border_style="green"))

            for tc in (msg.tool_calls or []):
                name = tc.get("name", "")
                args = tc.get("args", {})
                if name == "muapi_select":
                    console.print(f"  [bold cyan]→ Discovering models for:[/] {args.get('intent', '')[:60]}")
                    self.spinner("Discovering models…")
                elif name == "muapi_generate":
                    console.print(f"  [bold magenta]→ Generating {args.get('kind', 'asset')}:[/] {args.get('prompt', '')[:60]}")
                    self.spinner("Generating…")
                elif name == "task":
                    desc = args.get("description", "")
                    console.print(f"  [bold yellow]→ Delegating:[/] {desc[:70]}")
                    self.spinner("Delegating to specialist…")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content = str(msg.content)
            if name == "muapi_generate":
                if '"ok": true' in content or '"url":' in content:
                    import json
                    try:
                        data = json.loads(content)
                        url = data.get("url", "")
                        console.print(f"  [green]✓ Generated:[/] {url}")
                    except Exception:
                        console.print(f"  [green]✓ Generated[/]")
                else:
                    console.print(f"  [red]✗ Generation failed[/]")
            elif name == "muapi_select":
                console.print(f"  [green]✓ Models discovered[/]")
            elif name == "task":
                console.print(f"  [green]✓ Specialist done[/]")


async def run(brief: str, budget_credits: int = 500) -> None:
    agent, cost_cb = create_creative_agent(budget_credits)
    display = AgentDisplay()

    console.print()
    console.print("[bold blue]MuAPI Creative Agent[/]")
    console.print(f"[dim]Brief: {brief}[/]")
    console.print(f"[dim]Budget: {budget_credits} credits[/]")
    console.print()

    config = {"configurable": {"thread_id": "creative-agent-demo"}, "callbacks": [cost_cb]}

    with Live(display.spinner(), console=console, refresh_per_second=10, transient=True) as live:
        async for chunk in agent.astream(
            {"messages": [("user", brief)]},
            config=config,
            stream_mode="values",
        ):
            if "messages" not in chunk:
                continue
            messages = chunk["messages"]
            if len(messages) <= display.printed_count:
                continue
            live.stop()
            for msg in messages[display.printed_count:]:
                display.render(msg)
            display.printed_count = len(messages)
            live.start()
            live.update(display.spinner())

    summary = cost_cb.summary()
    console.print()
    console.print(
        f"[bold green]✓ Done[/] — "
        f"[dim]{summary['total_credits']} credits across {summary['calls']} calls[/]"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MuAPI Creative Agent")
    parser.add_argument("brief", nargs="*", help="Creative brief (or omit for demo)")
    parser.add_argument("--budget", type=int, default=500, help="Credit budget cap (default 500)")
    args = parser.parse_args()

    brief = " ".join(args.brief) if args.brief else (
        "Generate a cinematic product photo of a pair of minimalist white sneakers "
        "on a wet cobblestone street at night, neon reflections"
    )

    try:
        asyncio.run(run(brief, budget_credits=args.budget))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")


if __name__ == "__main__":
    main()
