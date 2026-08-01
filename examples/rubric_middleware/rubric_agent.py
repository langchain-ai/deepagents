"""Run `RubricMiddleware` against real models with LangSmith tracing.

Mirrors `TestRubricMiddlewareEndToEnd` in the core test suite, but with real
models instead of fakes: the agent drafts a document, a grader model scores it
against `RUBRIC`, and the middleware loops the agent until every criterion is
verifiably satisfied or `MAX_ITERATIONS` is spent.

Credentials come from a gitignored `.env` (see `.env.example`). They are read
through `os.environ` only and are never printed.

Usage:
    uv run --with "deepagents" --with "langchain[anthropic]" --with python-dotenv \
        python examples/rubric_middleware/rubric_agent.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.run_helpers import trace

from deepagents import RubricMiddleware, create_deep_agent
from deepagents.middleware.rubric import RUBRIC_GRADER_MESSAGE_SOURCE, RubricEvaluation

AGENT_MODEL = "anthropic:claude-sonnet-4-6"
"""Default model that drafts and revises the document."""

GRADER_MODEL = "anthropic:claude-sonnet-4-6"
"""Default model that grades the transcript against the rubric.

Override with `--grader-model` to see how the middleware behaves when the
grader is too weak to enumerate the rubric reliably.
"""

MAX_ITERATIONS = 4
"""Grading passes allowed before the run terminates as `max_iterations_reached`."""

PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
"""Environment variable holding the API key for each supported provider prefix."""

LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "deepagents-rubric-example")

TASK = """Write an internal engineering brief on why our nightly batch job started
timing out after we moved it from a single worker to a four-worker pool. Save it to
`brief.md`. Keep it under 400 words."""

RUBRIC = """- The brief names at least two distinct plausible causes for the regression,
  and for each one states the specific evidence an on-call engineer could check to
  confirm or rule it out.
- The brief proposes a concrete mitigation for each cause it names, not a generic
  "add more monitoring" suggestion.
- The brief explicitly states what it does NOT know, so the reader can tell analysis
  apart from speculation.
- The brief is written for an engineer with no prior context on this job: every
  acronym or internal term it uses is defined on first use.
- The brief is saved to `brief.md` and is under 400 words."""


def _key_var_for(model: str) -> str:
    """Map a `provider:model` string to the env var holding its API key.

    Rejects unknown providers up front rather than letting the failure surface
    later as an opaque auth error from inside the graph.
    """
    provider, separator, _ = model.partition(":")
    if not separator or provider not in PROVIDER_KEY_VARS:
        supported = ", ".join(sorted(PROVIDER_KEY_VARS))
        print(f"Unsupported model identifier {model!r}. Use one of: {supported}.", file=sys.stderr)
        raise SystemExit(2)
    return PROVIDER_KEY_VARS[provider]


def _require_env(*names: str) -> None:
    """Exit with a readable message if any variable is unset.

    Only variable names are reported; values never reach stdout.
    """
    missing = [name for name in dict.fromkeys(names) if not os.environ.get(name)]
    if missing:
        print(f"Missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        print("Copy examples/rubric_middleware/.env.example to a .env file and fill it in.", file=sys.stderr)
        raise SystemExit(1)


def _print_evaluation(evaluation: RubricEvaluation) -> None:
    """Print one grader verdict as it happens."""
    verdict = evaluation["result"]
    if evaluation["unverified"]:
        verdict += " (downgraded: grading was incomplete)"
    print(f"\n--- iteration {evaluation['iteration']}: {verdict} ---")
    print(f"    {evaluation['explanation']}")
    for criterion in evaluation["criteria"]:
        mark = "PASS" if criterion["passed"] else "FAIL"
        print(f"    [{mark}] {criterion['name']}")
        gap = criterion.get("gap")
        if gap:
            print(f"           gap: {gap}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to the `.env` to load. Defaults to the nearest one at or above the working directory.",
    )
    parser.add_argument("--agent-model", default=AGENT_MODEL, help=f"Model that drafts the document. Default: {AGENT_MODEL}.")
    parser.add_argument("--grader-model", default=GRADER_MODEL, help=f"Model that grades against the rubric. Default: {GRADER_MODEL}.")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help=f"Grading passes allowed. Default: {MAX_ITERATIONS}.")
    parser.add_argument("--thread-id", default="rubric-example", help="Checkpointer thread id. Change it to start from a clean state.")
    return parser.parse_args()


def main() -> None:
    """Run the rubric loop once and report where the trace landed."""
    args = _parse_args()
    load_dotenv(args.env_file or find_dotenv(usecwd=True))
    _require_env(
        _key_var_for(args.agent_model),
        _key_var_for(args.grader_model),
        "LANGSMITH_API_KEY",
    )
    # Tracing is the point of this example, so turn it on rather than
    # depending on whichever value the .env happens to carry.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

    print(f"agent model:  {args.agent_model}")
    print(f"grader model: {args.grader_model}")

    agent = create_deep_agent(
        model=args.agent_model,
        middleware=[
            RubricMiddleware(
                model=args.grader_model,
                max_iterations=args.max_iterations,
                on_evaluation=_print_evaluation,
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: dict[str, Any] = {"configurable": {"thread_id": args.thread_id}}

    with trace(name="rubric-middleware-example", project_name=LANGSMITH_PROJECT) as run:
        result = agent.invoke(
            {"messages": [HumanMessage(content=TASK)], "rubric": RUBRIC},
            config=config,
        )
        trace_url = run.get_url()

    state = agent.get_state(config).values
    print("\n=== run summary ===")
    print(f"status:     {state.get('_rubric_status')}")
    print(f"iterations: {state.get('_rubric_iterations')}")
    print(f"criteria:   {len(state.get('_rubric_criteria') or [])} frozen after the first pass")
    for name in state.get("_rubric_criteria") or []:
        print(f"  - {name}")

    revisions = [m for m in result["messages"] if m.additional_kwargs.get("lc_source") == RUBRIC_GRADER_MESSAGE_SOURCE]
    print(f"\nrevision prompts injected: {len(revisions)}")
    for index, message in enumerate(revisions):
        print(f"\n--- revision prompt {index} ---\n{message.content}")

    # The default backend keeps files in graph state, so nothing is written to
    # the host filesystem. The agent picks its own path prefix, so match on the
    # basename rather than assuming one.
    files = state.get("files") or {}
    for path, data in files.items():
        if path.rsplit("/", 1)[-1] == "brief.md":
            print(f"\n--- {path} ---\n{data['content']}")
            break

    print(f"\nLangSmith trace: {trace_url}")


if __name__ == "__main__":
    main()
