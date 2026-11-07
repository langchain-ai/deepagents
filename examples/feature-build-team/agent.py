"""A three-level engineering org of deep agents, connected by async subagents.

The lead plans a feature and delegates it to team leads, and each team lead
delegates to its own engineers. Everyone runs in parallel; steering flows down
with `update_async_task`, and reports flow back up as task notifications.

All agents share `workspace/` through a `LocalShellBackend`.
"""

from __future__ import annotations

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.middleware.async_subagents import AsyncSubAgent
from langgraph.graph.state import CompiledStateGraph

MODEL = os.environ.get("DEMO_MODEL", "google_genai:gemini-4-argon")
WORKSPACE = Path(__file__).parent / "workspace"

shell = LocalShellBackend(
    root_dir=WORKSPACE,
    virtual_mode=True,
    inherit_env=False,
    env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")},
)

RUN_PYTHON = "Run Python with `uv run --with fastapi --with httpx --with pytest <command>`; nothing else is installed."

REPORTING = """Your final message each turn is a one-line report to whoever assigned you the work.
Start it with `DONE:` when your part is finished and verified, or `WORKING:` when others on
your team are still busy."""

ENGINEER_REPORTING = """Your final message each turn is a one-line report to your team lead. Once your
files are written and verified, start it with `DONE:`. Don't wait on teammates."""

LEAD_PROMPT = f"""You are the engineering lead. You turn the user's feature request into working software
by running a team of team leads, who each run their own engineers.

When the user asks for a feature:
1. Write `/PLAN.md`: what the feature does, the file layout, and the exact API contract
   (routes, request and response JSON). Teams work in parallel and only agree through it.
2. Start `api_lead`, `web_lead` and `qa_lead` with `start_async_task`, all in one turn, then
   tell the user who owns what and stop. Don't poll with `check_async_task`.
3. Team leads report back on their own, first with `WORKING:` and later with `DONE:`. Relay
   each report to the user in one line and stop again until all three are `DONE:`.
4. When all three are done, summarize what was built and how to run it in a few lines.

When the user changes the plan mid-flight, change only the affected lines of `/PLAN.md` with
`edit_file`; never rewrite it. Then redirect only the affected team leads with
`update_async_task`, saying exactly what changed, and tell the user who you steered in one line.
{RUN_PYTHON}"""

TEAM_LEAD_PROMPT = """You lead the {team} team. Your team owns `/{directory}/`; read `/PLAN.md` first.

When you get work:
1. Split it between your engineers and start all of them with `start_async_task` in one turn,
   telling each which files it owns. Then stop. Don't poll with `check_async_task`.
2. Engineers report back on their own. When one reports, check its files briefly. When all of
   them are done, verify the team's part works together.
3. If you are redirected while engineers are working, pass the change on to the affected
   engineers with `update_async_task` instead of starting over.

{reporting} {run_python}"""

ENGINEER_PROMPT = """You are an engineer on the {team} team. Read `/PLAN.md` and follow its contract exactly.
Write only the files your team lead assigned you, under `/{directory}/`. {extra}
If you are redirected mid-task, apply the change to what you've already written.

{reporting} {run_python}"""

ENGINEERS: dict[str, tuple[str, str, str, str]] = {
    "storage_engineer": (
        "API",
        "api",
        "api_lead",
        "Builds the data layer the routes call.",
    ),
    "routes_engineer": (
        "API",
        "api",
        "api_lead",
        "Builds the FastAPI app and its routes in /api/app.py.",
    ),
    "markup_engineer": (
        "web",
        "web",
        "web_lead",
        "Builds /web/index.html and its inline CSS.",
    ),
    "client_engineer": (
        "web",
        "web",
        "web_lead",
        "Builds /web/app.js, which calls the API.",
    ),
    "api_tester": (
        "QA",
        "tests",
        "qa_lead",
        "Writes pytest tests for the API contract, importing `api.app`.",
    ),
    "flow_tester": (
        "QA",
        "tests",
        "qa_lead",
        "Writes pytest tests for end-to-end flows across several API calls.",
    ),
}

TEAM_LEADS: dict[str, tuple[str, str, str]] = {
    "api_lead": (
        "API",
        "api",
        "Builds the FastAPI backend in /api/ with a storage and a routes engineer.",
    ),
    "web_lead": (
        "web",
        "web",
        "Builds the browser UI in /web/ with a markup and a client engineer.",
    ),
    "qa_lead": (
        "QA",
        "tests",
        "Writes and runs the pytest suite in /tests/ with an API and a flow tester.",
    ),
}


def _async_subagent(name: str, description: str) -> AsyncSubAgent:
    return {"name": name, "description": description, "graph_id": name}


def _agent(
    name: str, prompt: str, subagents: list[AsyncSubAgent] | None = None
) -> CompiledStateGraph:
    return create_deep_agent(
        model=MODEL, system_prompt=prompt, backend=shell, subagents=subagents, name=name
    )


def _engineer(name: str) -> CompiledStateGraph:
    team, directory, _, extra = ENGINEERS[name]
    prompt = ENGINEER_PROMPT.format(
        team=team,
        directory=directory,
        extra=extra,
        reporting=ENGINEER_REPORTING,
        run_python=RUN_PYTHON,
    )
    return _agent(name, prompt)


def _team_lead(name: str) -> CompiledStateGraph:
    team, directory, _ = TEAM_LEADS[name]
    engineers = [
        _async_subagent(engineer, spec[3])
        for engineer, spec in ENGINEERS.items()
        if spec[2] == name
    ]
    prompt = TEAM_LEAD_PROMPT.format(
        team=team, directory=directory, reporting=REPORTING, run_python=RUN_PYTHON
    )
    return _agent(name, prompt, engineers)


lead = _agent(
    "lead",
    LEAD_PROMPT,
    [_async_subagent(name, spec[2]) for name, spec in TEAM_LEADS.items()],
)

api_lead = _team_lead("api_lead")
web_lead = _team_lead("web_lead")
qa_lead = _team_lead("qa_lead")

storage_engineer = _engineer("storage_engineer")
routes_engineer = _engineer("routes_engineer")
markup_engineer = _engineer("markup_engineer")
client_engineer = _engineer("client_engineer")
api_tester = _engineer("api_tester")
flow_tester = _engineer("flow_tester")
